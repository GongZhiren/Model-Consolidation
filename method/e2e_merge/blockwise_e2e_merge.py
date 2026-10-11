"""Block-wise (sequential) End-to-End Functional Merging for CLIP.

Motivation (rebuttal): full-model E2E optimizes the exact end-to-end objective but
(a) holds the whole network's activations/grads in memory and (b) looks like generic
distillation. Block-wise E2E fixes both. It partitions the encoder's L transformer
blocks into consecutive GROUPS and merges them one group at a time, from the input
side, feeding each group the TRUE merged activation the student actually produces
(teacher-forcing on merged activations, which is exactly what closed-form RegMean
CANNOT do -- RegMean uses the expert's activation). For each group we match the
expert's own output of that group on the same merged input; the final group matches
the accuracy-relevant normalized projected embedding. Peak memory is that of a single
group, not the whole network, so it scales to large models / many experts while
approaching full E2E accuracy.

group_size sweeps the spectrum:
  group_size = 1   -> most memory-frugal, closest in spirit to a sequential RegMean
  group_size = L   -> identical to full-model E2E (one group = whole encoder)
Partition can be uniform (group_size) or amplification-driven (partition="alpha",
cutting at low-amplification edges so high-sensitivity late blocks stay grouped).
"""
import json
import logging
import os
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.mixins import CLIPClassificationMixin, auto_register_config
from fusion_bench.dataset.clip_dataset import CLIPDataset

log = logging.getLogger(__name__)


@auto_register_config
class BlockwiseE2EMergingForCLIP(CLIPClassificationMixin, BaseAlgorithm):
    def __init__(
        self,
        init_scale: float = 0.3,
        group_size: int = 2,           # consecutive blocks per group (uniform partition)
        partition: str = "uniform",    # "uniform" | "alpha" (amplification-driven)
        alpha_json: str = None,        # path to amplification.json for partition="alpha"
        mem_budget_blocks: int = 3,    # max blocks/group for partition="alpha"
        group_steps: int = 1000,       # gradient steps per group
        cache_batches: int = 0,        # >0: cache this many prefix-activation batches/task per group
                                       #      (forward the frozen prefix ONCE, then train from cache;
                                       #      makes each step cost ~one block, no prefix re-forward)
        lr: float = 5e-6,
        weight_decay: float = 0.0,
        batch_size: int = 32,
        grad_clip: float = 1.0,
        seed: int = 0,
        save_partition_to: str = None,
        dataloader_kwargs: dict = None,
        **kwargs,
    ):
        self.init_scale = init_scale
        self.group_size = group_size
        self.partition = partition
        self.alpha_json = alpha_json
        self.mem_budget_blocks = mem_budget_blocks
        self.group_steps = group_steps
        self.cache_batches = cache_batches
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.seed = seed
        self.save_partition_to = save_partition_to
        self.dataloader_kwargs = dataloader_kwargs or {"batch_size": batch_size, "num_workers": 0}
        super().__init__(**kwargs)

    # ---- task-arithmetic init ----
    @torch.no_grad()
    def _task_arithmetic_init(self, pretrained, task_models):
        base = {k: v.detach().cpu().float() for k, v in pretrained.state_dict().items()}
        merged = {k: v.clone() for k, v in base.items()}
        for tm in task_models:
            sd = tm.state_dict()
            for k in merged:
                if merged[k].dtype.is_floating_point:
                    merged[k] += self.init_scale * (sd[k].detach().cpu().float() - base[k])
        return merged

    # ---- run encoder blocks [i..j] on a hidden state (inclusive) ----
    @staticmethod
    def _run_blocks(vision_model, hidden, i, j):
        for l in range(i, j + 1):
            hidden = vision_model.encoder.layers[l](hidden, None, None)[0]
        return hidden

    # ---- embeddings + pre_layrnorm -> hidden entering block 0 ----
    @staticmethod
    def _embed_input(vision_model, pix):
        h = vision_model.embeddings(pix)
        h = vision_model.pre_layrnorm(h)
        return h

    def _projected(self, vision_model, last_hidden):
        pooled = vision_model.post_layernorm(last_hidden[:, 0, :])
        e = self.visual_projection(pooled)
        return e / e.norm(p=2, dim=-1, keepdim=True)

    # ---- partition the L blocks into consecutive groups ----
    def _make_partition(self, L) -> List[List[int]]:
        if self.partition == "alpha" and self.alpha_json and os.path.isfile(self.alpha_json):
            # Sketch: keep the HIGH-amplification edges INSIDE a block (optimize the
            # error-sensitive transitions jointly) and cut only at LOW-amplification
            # edges, subject to a per-block size budget. Start fully split, then greedily
            # merge across the highest-alpha edge whenever the merged block stays <= budget.
            alpha = json.load(open(self.alpha_json))["alpha_profile"]
            B = self.mem_budget_blocks
            edges_desc = sorted(range(1, L), key=lambda e: -alpha[e - 1])  # high-alpha first
            cut = set(range(1, L))  # every edge cut => all singletons
            for e in edges_desc:
                bnds = [0] + sorted(cut - {e}) + [L]
                if all(bnds[k + 1] - bnds[k] <= B for k in range(len(bnds) - 1)):
                    cut.discard(e)  # safe to keep blocks joined across this high-alpha edge
            bnds = [0] + sorted(cut) + [L]
            groups = [list(range(bnds[k], bnds[k + 1])) for k in range(len(bnds) - 1)]
        else:
            groups = [list(range(s, min(s + self.group_size, L)))
                      for s in range(0, L, self.group_size)]
        return groups

    def run(self, modelpool: BaseModelPool):
        torch.manual_seed(self.seed)
        if not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(modelpool)
        self.modelpool = modelpool
        self.setup_zero_shot_classification_head()
        device = self.fabric.device
        names = modelpool.model_names
        g = torch.Generator().manual_seed(self.seed)

        pretrained = modelpool.load_pretrained_model()
        teachers, loaders, src = {}, {}, {}
        for name in names:
            tm = modelpool.load_model(name).to(device).eval()
            for p in tm.parameters():
                p.requires_grad_(False)
            teachers[name] = tm
            src[name] = modelpool.load_train_dataset(name)
            loaders[name] = iter(DataLoader(CLIPDataset(src[name], self.clip_processor),
                                            shuffle=True, drop_last=True, **self.dataloader_kwargs))

        def next_batch(name):
            try:
                return next(loaders[name])
            except StopIteration:
                loaders[name] = iter(DataLoader(CLIPDataset(src[name], self.clip_processor),
                                                shuffle=True, drop_last=True, **self.dataloader_kwargs))
                return next(loaders[name])

        # student init
        student = modelpool.load_pretrained_model().to(device)
        init_sd = self._task_arithmetic_init(pretrained, list(teachers.values()))
        student.load_state_dict({k: v.to(student.state_dict()[k].dtype) for k, v in init_sd.items()})
        svm = student.vision_model
        L = len(svm.encoder.layers)
        groups = self._make_partition(L)
        log.info("[BlockE2E] partition (%s): %s", self.partition, groups)
        if self.save_partition_to:
            os.makedirs(os.path.dirname(self.save_partition_to), exist_ok=True)
            json.dump({"partition": groups, "group_size": self.group_size,
                       "mode": self.partition}, open(self.save_partition_to, "w"), indent=2)

        peak_mem = 0.0
        for gi, grp in enumerate(groups):
            i, j = grp[0], grp[-1]
            is_last = (j == L - 1)
            # freeze everything, then unfreeze only this group's blocks
            for p in student.parameters():
                p.requires_grad_(False)
            params = []
            for l in range(i, j + 1):
                for p in svm.encoder.layers[l].parameters():
                    p.requires_grad_(True)
                    params.append(p)
            opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
            student.train()
            torch.cuda.reset_peak_memory_stats(device) if torch.cuda.is_available() else None

            # ---- cached mode: forward the FROZEN prefix once, then train from cache ----
            # (the prefix 0..i-1 is fixed while group i..j trains, so re-forwarding it every
            #  step is pure waste; caching makes each step cost ~one group forward+backward.)
            cache = None
            if self.cache_batches > 0:
                cache = {}
                with torch.no_grad():
                    for name in names:
                        hs = []
                        for _ in range(self.cache_batches):
                            pix = next_batch(name)[0].to(device)
                            h_in = self._embed_input(svm, pix)
                            if i > 0:
                                h_in = self._run_blocks(svm, h_in, 0, i - 1)
                            hs.append(h_in.detach().to("cpu"))  # keep cache off-GPU
                        cache[name] = hs

            for step in tqdm(range(self.group_steps), desc=f"BlockE2E g{gi}[{i}-{j}]", leave=False):
                name = names[torch.randint(0, len(names), (1,), generator=g).item()]
                tvm = teachers[name].vision_model
                with torch.no_grad():
                    if cache is not None:
                        pool = cache[name]
                        h_in = pool[torch.randint(0, len(pool), (1,), generator=g).item()].to(device)
                    else:
                        pix = next_batch(name)[0].to(device)
                        h_in = self._embed_input(svm, pix)
                        if i > 0:
                            h_in = self._run_blocks(svm, h_in, 0, i - 1)
                    if is_last:
                        target = self._projected(tvm, self._run_blocks(tvm, h_in, i, j))
                    else:
                        target = self._run_blocks(tvm, h_in, i, j)
                # student group forward (with grad); prefix is NOT re-run under cache
                out = self._run_blocks(svm, h_in, i, j)
                if is_last:
                    out = self._projected(svm, out)
                loss = F.mse_loss(out, target)
                opt.zero_grad()
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, self.grad_clip)
                opt.step()
                if step % 200 == 0:
                    log.info("[BlockE2E] g%d[%d-%d] step %d loss=%.5f (%s)", gi, i, j, step, loss.item(), name)
            if torch.cuda.is_available():
                peak_mem = max(peak_mem, torch.cuda.max_memory_allocated(device) / 1e9)
            del cache

        student.eval()
        for p in student.parameters():
            p.requires_grad_(False)
        log.info("[BlockE2E] done. peak group memory ~ %.2f GB", peak_mem)
        return student
