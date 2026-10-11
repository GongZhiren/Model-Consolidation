"""End-to-End Functional Merging (E2E) for seq2seq LMs (Flan-T5) --- NLP generality.

Same principle as the CLIP method: minimize the end-to-end functional discrepancy
    min_theta  sum_a  E_{x~D_a} || g_theta(x) - g_{theta_a}(x) ||^2
where for a seq2seq classifier g_theta(x) is the decoder's first-step last-hidden
representation (the vector the LM head reads the label from) --- the seq2seq analog of
CLIP's pooled projected embedding. Label-free: the target is the frozen expert's
representation, never the gold label. Experts are LoRA adapters; we task-arithmetic
merge them into the base and refine by streaming gradient descent.
"""
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.autonotebook import tqdm

from transformers import AutoTokenizer

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.mixins import LightningFabricMixin, auto_register_config

log = logging.getLogger(__name__)


@auto_register_config
class Seq2SeqE2EMergingForFlanT5(LightningFabricMixin, BaseAlgorithm):
    def __init__(
        self,
        init_scale: float = 0.3,
        steps: int = 800,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        batch_size: int = 16,
        num_calibration_examples: int = None,
        match: str = "hidden",   # decoder/encoder hidden state or first-step logits
        init_from: str = None,
        grad_clip: float = 1.0,
        seed: int = 0,
        dataloader_kwargs: dict = None,
        **kwargs,
    ):
        self.init_scale = init_scale
        self.steps = steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_calibration_examples = num_calibration_examples
        self.match = match
        self.init_from = init_from
        self.grad_clip = grad_clip
        self.seed = seed
        self.dataloader_kwargs = dataloader_kwargs or {"batch_size": batch_size, "num_workers": 0}
        super().__init__(**kwargs)

    @torch.no_grad()
    def _task_arithmetic_init(self, pretrained_sd, expert_sds):
        merged = {k: v.clone().float() for k, v in pretrained_sd.items()}
        for sd in expert_sds:
            for k in merged:
                if merged[k].dtype.is_floating_point and k in sd:
                    merged[k] += self.init_scale * (sd[k].float() - pretrained_sd[k].float())
        return merged

    def _g(self, model, input_ids, attn):
        dsi = model.config.decoder_start_token_id
        dec = torch.full((input_ids.size(0), 1), dsi, dtype=torch.long, device=input_ids.device)
        needs_hidden = self.match.startswith("hidden")
        out = model(input_ids=input_ids, attention_mask=attn, decoder_input_ids=dec,
                    output_hidden_states=needs_hidden)
        if self.match in ("logit", "logit_kl"):
            return out.logits[:, 0, :]
        if self.match == "encoder_mean":
            h = out.encoder_last_hidden_state
            pooled = (h * attn.unsqueeze(-1)).sum(dim=1) / attn.sum(
                dim=1, keepdim=True
            ).clamp_min(1)
            return F.normalize(pooled, p=2, dim=-1)
        if self.match == "encoder_token":
            return F.normalize(out.encoder_last_hidden_state, p=2, dim=-1)
        h = out.decoder_hidden_states[-1][:, 0, :]  # post-final-layernorm: what the LM head reads
        if self.match in ("hidden_raw", "hidden_cosine"):
            return h
        return h / (h.norm(p=2, dim=-1, keepdim=True) + 1e-8)  # normalized (CLIP-analog), default

    def _merge_lora(self, m):
        # fold a LoRA/PEFT expert into a plain seq2seq model, return its state_dict
        try:
            m = m.merge_and_unload()
        except Exception:
            pass
        return {k: v.detach().cpu() for k, v in m.state_dict().items()}

    def run(self, modelpool: BaseModelPool):
        torch.manual_seed(self.seed)
        if not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(modelpool)
        self.modelpool = modelpool
        device = self.fabric.device
        names = modelpool.model_names
        g = torch.Generator().manual_seed(self.seed)

        pretrained = modelpool.load_pretrained_model()
        pretrained_sd = {k: v.detach().cpu() for k, v in pretrained.state_dict().items()}

        # frozen expert teachers (LoRA folded in) + their state_dicts for the init
        teachers, expert_sds = {}, []
        for name in names:
            em = modelpool.load_model(name)
            sd = self._merge_lora(em)
            expert_sds.append(sd)
            tm = modelpool.load_pretrained_model()
            tm.load_state_dict({k: v.to(tm.state_dict()[k].dtype) for k, v in sd.items() if k in tm.state_dict()},
                               strict=False)
            tm = tm.to(device).eval()
            for p in tm.parameters():
                p.requires_grad_(False)
            teachers[name] = tm
            log.info("[Seq2SeqE2E] teacher ready: %s", name)

        # streaming calibration loaders (unlabeled task inputs); pad variable-length
        # token sequences into batched tensors via the tokenizer.
        try:
            tokenizer = modelpool.load_tokenizer()
        except Exception:
            base = getattr(pretrained.config, "_name_or_path", "google/flan-t5-base")
            tokenizer = AutoTokenizer.from_pretrained(base)

        def collate(features):
            return tokenizer.pad(
                {"input_ids": [f["input_ids"] for f in features],
                 "attention_mask": [f["attention_mask"] for f in features]},
                padding=True, return_tensors="pt")

        src = {name: modelpool.load_test_dataset(name) for name in names}
        if self.num_calibration_examples is not None:
            capped = {}
            for name in names:
                dataset = src[name]
                count = min(self.num_calibration_examples, len(dataset))
                indices = torch.randperm(
                    len(dataset), generator=torch.Generator().manual_seed(self.seed)
                )[:count].tolist()
                capped[name] = Subset(dataset, indices)
            src = capped
            log.info(
                "[Seq2SeqE2E] fixed calibration pool: at most %d examples/task",
                self.num_calibration_examples,
            )
        loaders = {name: iter(DataLoader(src[name], shuffle=True, drop_last=True,
                                         collate_fn=collate, **self.dataloader_kwargs))
                   for name in names}

        def next_batch(name):
            try:
                return next(loaders[name])
            except StopIteration:
                loaders[name] = iter(DataLoader(src[name], shuffle=True, drop_last=True,
                                                collate_fn=collate, **self.dataloader_kwargs))
                return next(loaders[name])

        # student init: task-arithmetic merge of the experts
        student = modelpool.load_pretrained_model()
        if self.init_from:
            init_sd = torch.load(self.init_from, map_location="cpu")
            init_sd = init_sd.get("state_dict", init_sd) if isinstance(init_sd, dict) else init_sd
            log.info("[Seq2SeqE2E] initialized student from %s", self.init_from)
        else:
            init_sd = self._task_arithmetic_init(pretrained_sd, expert_sds)
        student.load_state_dict({k: v.to(student.state_dict()[k].dtype) for k, v in init_sd.items()})
        student = student.to(device)
        student.train()
        for p in student.parameters():
            p.requires_grad_(True)

        opt = torch.optim.AdamW(student.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for step in tqdm(range(self.steps), desc="Seq2SeqE2E"):
            name = names[torch.randint(0, len(names), (1,), generator=g).item()]
            batch = next_batch(name)
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            with torch.no_grad():
                tgt = self._g(teachers[name], input_ids, attn)
            out = self._g(student, input_ids, attn)
            if self.match == "logit_kl":
                # match the expert's first-step output distribution (what generation reads)
                loss = F.kl_div(F.log_softmax(out, dim=-1), F.softmax(tgt, dim=-1),
                                reduction="batchmean")
            elif self.match == "hidden_cosine":
                loss = (1.0 - F.cosine_similarity(out, tgt, dim=-1)).mean()
            elif self.match == "encoder_token":
                token_loss = F.mse_loss(out, tgt, reduction="none").mean(dim=-1)
                loss = (token_loss * attn).sum() / attn.sum().clamp_min(1)
            else:
                loss = F.mse_loss(out, tgt)
            opt.zero_grad()
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), self.grad_clip)
            opt.step()
            if step % 100 == 0:
                log.info("[Seq2SeqE2E] step %d loss=%.5f (%s)", step, loss.item(), name)

        student.eval()
        for p in student.parameters():
            p.requires_grad_(False)
        return student
