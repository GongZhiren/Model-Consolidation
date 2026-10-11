"""End-to-End Functional Merging (E2E) for CLIP vision models — FusionBench harness.

Thesis (clean novelty): RegMean / RegMean++ minimize per-LAYER activation discrepancy — a
LOCAL surrogate whose errors compound through depth. The quantity that actually matters is
the END-TO-END functional discrepancy: the merged model should reproduce each task model's
final representation on that task's own inputs. We optimize exactly that objective:

    min_θ  Σ_a  E_{x∼D_a}  || g_θ(x) − g_{θ_a}(x) ||²           (g = CLIP vision pooled output)

RegMean is the layer-wise first-order stationarity condition of a single layer of this
objective; solving it globally (gradient descent from a task-arithmetic init) is strictly
more expressive. Data-based but LABEL-FREE (matches task outputs, not labels).
"""
import logging
from copy import deepcopy
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.mixins import CLIPClassificationMixin, auto_register_config
from fusion_bench.dataset.clip_dataset import CLIPDataset

log = logging.getLogger(__name__)


@auto_register_config
class E2EFunctionalMergingForCLIP(CLIPClassificationMixin, BaseAlgorithm):
    def __init__(
        self,
        init_scale: float = 0.3,       # task-arithmetic init: θ0 + init_scale·Σ τ_a
        steps: int = 400,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        batch_size: int = 32,
        num_cal_batches: int = 8,      # calibration batches per task
        anchor: float = 0.0,           # optional L2 pull toward init (regularization)
        grad_clip: float = 1.0,
        match: str = "proj",           # "proj" = normalized projected embeds (accuracy-aligned); "pool" = raw pooled MSE; "logit" = KL
        init_from: str = None,         # path to a saved merged state_dict to init from (e.g. RegMean++) instead of task-arith
        seed: int = 0,                 # controls task-selection order + data shuffle (for multi-seed robustness)
        stream: bool = True,           # True = fresh batch every step; False = fixed calibration set of num_cal_batches (overfitting ablation)
        sam_rho: float = 0.0,          # >0 enables sharpness-aware minimization (faithful SAMerging re-impl. with match=logit)
        dataloader_kwargs: dict = None,
        **kwargs,
    ):
        self.sam_rho = sam_rho
        self.stream = stream
        self.seed = seed
        self.init_from = init_from
        self.init_scale = init_scale
        self.steps = steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_cal_batches = num_cal_batches
        self.anchor = anchor
        self.grad_clip = grad_clip
        self.match = match
        self.dataloader_kwargs = dataloader_kwargs or {"batch_size": batch_size, "num_workers": 0}
        super().__init__(**kwargs)

    def _embed(self, model, x):
        """final classification-relevant embedding: normalized visual_projection(pooled)."""
        h = model(x)[1]
        if self.match == "pool":
            return h
        e = self.visual_projection(h)
        return e / e.norm(p=2, dim=-1, keepdim=True)

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

    def run(self, modelpool: BaseModelPool):
        torch.manual_seed(self.seed)  # determinism across the whole run (loader shuffle + task order)
        if not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(modelpool)
        self.modelpool = modelpool
        device = self.fabric.device
        names = modelpool.model_names

        if self.match == "logit":
            self.setup_zero_shot_classification_head()
        else:
            # Representation matching needs the fixed visual projection, but
            # not the expensive per-task zero-shot text heads.
            source_name = "_pretrained_" if modelpool.has_pretrained else names[0]
            clip_model = modelpool.load_clip_model(source_name)
            self.visual_projection = deepcopy(clip_model.visual_projection)
            self.visual_projection.requires_grad_(False)
            self.visual_projection = self.fabric.to_device(self.visual_projection)
            del clip_model

        pretrained = modelpool.load_pretrained_model()
        # ---- keep all task models frozen on GPU (teachers) + streaming train loaders
        teachers, loaders, data_generators = {}, {}, {}
        for name in names:
            tm = modelpool.load_model(name).to(device).eval()
            for p in tm.parameters():
                p.requires_grad_(False)
            teachers[name] = tm
            # Use a dedicated generator for every task so that the calibration
            # subset depends only on ``seed`` rather than on unrelated model-loading
            # RNG consumption.  This also permits exactly paired subset-seed
            # comparisons with closed-form baselines.
            data_generators[name] = torch.Generator().manual_seed(self.seed)
            dl = DataLoader(CLIPDataset(modelpool.load_train_dataset(name), self.clip_processor),
                            shuffle=True, drop_last=True, generator=data_generators[name],
                            **self.dataloader_kwargs)
            loaders[name] = iter(dl)
            log.info("[E2E] teacher + stream loader ready for %s", name)
        self._loaders_src = {n: modelpool.load_train_dataset(n) for n in names}

        # fixed-calibration mode: pre-sample num_cal_batches per task and cycle only those
        fixed_cache, fixed_ptr = {}, {}
        if not self.stream:
            for name in names:
                bs = []
                for _ in range(self.num_cal_batches):
                    try:
                        bs.append(next(loaders[name]))
                    except StopIteration:
                        loaders[name] = iter(DataLoader(CLIPDataset(self._loaders_src[name], self.clip_processor),
                                                        shuffle=True, drop_last=True,
                                                        generator=data_generators[name],
                                                        **self.dataloader_kwargs))
                        bs.append(next(loaders[name]))
                fixed_cache[name] = bs
                fixed_ptr[name] = 0
            log.info("[E2E] FIXED calibration set: %d batches/task", self.num_cal_batches)

        def next_batch(name):
            if not self.stream:  # cycle through the fixed calibration set (tests overfitting)
                b = fixed_cache[name][fixed_ptr[name] % len(fixed_cache[name])]
                fixed_ptr[name] += 1
                return b
            try:
                return next(loaders[name])
            except StopIteration:
                dl = DataLoader(CLIPDataset(self._loaders_src[name], self.clip_processor),
                                shuffle=True, drop_last=True,
                                generator=data_generators[name],
                                **self.dataloader_kwargs)
                loaders[name] = iter(dl)
                return next(loaders[name])

        # ---- init merged model (student): task-arithmetic, or load a saved merged model (e.g. RegMean++)
        merged = modelpool.load_pretrained_model().to(device)
        if self.init_from:
            import os as _os
            if _os.path.isdir(self.init_from):  # HF model directory (FusionBench save_pretrained)
                from transformers import CLIPVisionModel
                src = CLIPVisionModel.from_pretrained(self.init_from)
                sd = src.state_dict()
            else:                                # torch.save state_dict / checkpoint
                sd = torch.load(self.init_from, map_location="cpu")
                sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
            merged.load_state_dict({k: v.to(merged.state_dict()[k].dtype) for k, v in sd.items()
                                    if k in merged.state_dict()}, strict=False)
            log.info("[E2E] initialized student from %s", self.init_from)
        else:
            init_sd = self._task_arithmetic_init(pretrained, list(teachers.values()))
            merged.load_state_dict({k: v.to(merged.state_dict()[k].dtype) for k, v in init_sd.items()})
        init_params = {k: v.detach().clone() for k, v in merged.named_parameters()} if self.anchor > 0 else None

        # ---- STREAMING distillation: fresh batch each step → matches the full task distribution (no memorization)
        merged.train()
        for p in merged.parameters():
            p.requires_grad_(True)
        opt = torch.optim.AdamW(merged.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        torch.manual_seed(self.seed)  # data-shuffle order (DataLoader global RNG)
        g = torch.Generator().manual_seed(self.seed)
        for step in tqdm(range(self.steps), desc="E2E stream"):
            name = names[torch.randint(0, len(names), (1,), generator=g).item()]
            x = next_batch(name)[0].to(device)
            with torch.no_grad():
                tgt = self.compute_logits(teachers[name], x, name) if self.match == "logit" \
                    else self._embed(teachers[name], x)
            def _fwd_loss():
                if self.match == "logit":
                    slog = self.compute_logits(merged, x, name)
                    l = F.kl_div(F.log_softmax(slog, -1), F.softmax(tgt, -1), reduction="batchmean")
                else:
                    out = self._embed(merged, x)
                    l = F.mse_loss(out, tgt)
                if self.anchor > 0:
                    l = l + self.anchor * sum(
                        ((p - init_params[k]) ** 2).sum() for k, p in merged.named_parameters())
                return l
            loss = _fwd_loss()
            opt.zero_grad(); loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(merged.parameters(), self.grad_clip)
            if self.sam_rho > 0:
                # SAM: ascend to the local worst case, recompute the gradient there, then step
                with torch.no_grad():
                    gnorm = torch.norm(torch.stack(
                        [p.grad.norm() for p in merged.parameters() if p.grad is not None]))
                    scale = self.sam_rho / (gnorm + 1e-12)
                    e_w = {}
                    for p in merged.parameters():
                        if p.grad is None:
                            continue
                        e = p.grad * scale
                        p.add_(e); e_w[p] = e
                opt.zero_grad()
                loss2 = _fwd_loss(); loss2.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(merged.parameters(), self.grad_clip)
                with torch.no_grad():
                    for p, e in e_w.items():
                        p.sub_(e)  # restore original weights before the optimizer step
            opt.step()
            if step % 100 == 0:
                log.info("[E2E] step %d  loss=%.5f  (%s)", step, loss.item(), name)
        merged.eval()
        for p in merged.parameters():
            p.requires_grad_(False)
        return merged
