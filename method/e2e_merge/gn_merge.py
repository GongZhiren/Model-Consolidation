"""Gauss-Newton Functional Merge (GN-Merge) for CLIP --- a genuinely new merger.

The end-to-end functional objective is  F(theta)=sum_a E_x ||g_theta(x)-g_{theta_a}(x)||^2.
RegMean is (Prop. in the paper) the per-LAYER, first-order, ISOTROPIC-Jacobian
approximation of the Gauss-Newton step on F. GN-Merge instead takes the EXACT
Gauss-Newton step, globally over all weights, in closed form:

    (J^T J + lambda I) dtheta = J^T r,      J = d g_theta / d theta at the current merge,
                                            r = g_{expert}(x) - g_theta(x),

solved matrix-free by conjugate gradient (each CG mat-vec is one JVP + one VJP through
the whole encoder; the Jacobian is never formed). This is NOT gradient descent -- it is
one (optionally a few) closed-form second-order steps -- and it is not a stitch of parts:
it is the exact induced metric of which RegMean (isotropic) and Fisher (diagonal) are
approximations. A few GN steps recover most of what full gradient E2E reaches, with no
SGD training loop.
"""
import logging
import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp, vjp
from torch.utils.data import DataLoader

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.mixins import CLIPClassificationMixin, auto_register_config
from fusion_bench.dataset.clip_dataset import CLIPDataset

log = logging.getLogger(__name__)


def _force_eager(model):
    # forward-mode AD (JVP) is unsupported by the fused SDPA kernel; use eager attention.
    for mod in model.modules():
        if hasattr(mod, "config"):
            mod.config._attn_implementation = "eager"
        if hasattr(mod, "_attn_implementation"):
            mod._attn_implementation = "eager"
    if hasattr(model, "config"):
        model.config._attn_implementation = "eager"
    return model


@auto_register_config
class GaussNewtonMergingForCLIP(CLIPClassificationMixin, BaseAlgorithm):
    def __init__(
        self,
        init_scale: float = 0.3,
        n_cal_batches: int = 4,      # calibration batches per task
        cg_iters: int = 25,          # conjugate-gradient iterations per GN step
        lam: float = 1e-2,           # Levenberg-Marquardt damping (J^T J + lam I)
        n_gn_steps: int = 3,         # number of closed-form GN steps (re-linearize each)
        step_size: float = 1.0,      # scale on the GN update (LM trust)
        batch_size: int = 16,
        seed: int = 0,
        init_from: str = None,       # optional: start from a saved merged state_dict (e.g. RegMean++)
        dataloader_kwargs: dict = None,
        **kwargs,
    ):
        self.init_scale = init_scale
        self.n_cal_batches = n_cal_batches
        self.cg_iters = cg_iters
        self.lam = lam
        self.n_gn_steps = n_gn_steps
        self.step_size = step_size
        self.batch_size = batch_size
        self.seed = seed
        self.init_from = init_from
        self.dataloader_kwargs = dataloader_kwargs or {"batch_size": batch_size, "num_workers": 0}
        super().__init__(**kwargs)

    @torch.no_grad()
    def _task_arithmetic_init(self, pretrained, task_models):
        base = {k: v.detach().float() for k, v in pretrained.state_dict().items()}
        merged = {k: v.clone() for k, v in base.items()}
        for tm in task_models:
            sd = tm.state_dict()
            for k in merged:
                if merged[k].dtype.is_floating_point:
                    merged[k] += self.init_scale * (sd[k].detach().float() - base[k])
        return merged

    def _g(self, vm, params, pix):
        out = functional_call(vm, params, (pix,))
        e = self.visual_projection(out.pooler_output)
        return e / e.norm(p=2, dim=-1, keepdim=True)

    def run(self, modelpool: BaseModelPool):
        torch.manual_seed(self.seed)
        if not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(modelpool)
        self.modelpool = modelpool
        self.setup_zero_shot_classification_head()
        device = self.fabric.device
        names = modelpool.model_names

        pretrained = _force_eager(modelpool.load_pretrained_model()).to(device).eval()
        teachers, src = {}, {}
        for name in names:
            tm = _force_eager(modelpool.load_model(name)).to(device).eval()
            for p in tm.parameters():
                p.requires_grad_(False)
            teachers[name] = tm
            src[name] = modelpool.load_train_dataset(name)

        # student init: task-arithmetic, or refine a supplied merge (e.g. RegMean++), eager attention
        student = _force_eager(modelpool.load_pretrained_model()).to(device)
        if self.init_from:
            import os
            if os.path.isdir(self.init_from):
                from transformers import CLIPVisionModel
                sd = CLIPVisionModel.from_pretrained(self.init_from).state_dict()
            else:
                sd = torch.load(self.init_from, map_location="cpu")
                sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
            student.load_state_dict({k: v.to(student.state_dict()[k].dtype)
                                     for k, v in sd.items() if k in student.state_dict()}, strict=False)
            log.info("[GN] initialized student from %s", self.init_from)
        else:
            init_sd = self._task_arithmetic_init(pretrained, list(teachers.values()))
            student.load_state_dict({k: v.to(student.state_dict()[k].dtype) for k, v in init_sd.items()})
        svm = student.vision_model
        for p in student.parameters():
            p.requires_grad_(False)

        # ---- calibration set: (pixels, expert target embedding) ----
        log.info("[GN] building calibration set (%d batches/task) ...", self.n_cal_batches)
        cal = []
        for name in names:
            loader = iter(DataLoader(CLIPDataset(src[name], self.clip_processor),
                                     shuffle=True, drop_last=True, **self.dataloader_kwargs))
            tvm = teachers[name].vision_model
            for _ in range(self.n_cal_batches):
                try:
                    pix = next(loader)[0].to(device)
                except StopIteration:
                    break
                with torch.no_grad():
                    tgt = self._g(tvm, dict(tvm.named_parameters()), pix)
                cal.append((pix, tgt))
        log.info("[GN] calibration batches: %d", len(cal))

        # teachers are only needed for the (already-computed) targets; free them from GPU
        # so the GN solve (student + CG tangent vectors + JVP/VJP graphs) fits on large backbones.
        for name in list(teachers.keys()):
            teachers[name].to("cpu")
        del teachers, pretrained
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # parameter pytree we optimize: all float params of the student vision_model
        theta = {k: v.detach().clone() for k, v in svm.named_parameters()}
        fkeys = [k for k, v in theta.items() if v.is_floating_point()]

        def z():        return {k: torch.zeros_like(theta[k]) for k in fkeys}
        def dot(a, b):  return sum((a[k] * b[k]).sum() for k in fkeys)
        def axpy(al, x, y): return {k: al * x[k] + y[k] for k in fkeys}

        # ---- iterated Gauss-Newton: each step re-linearizes at the current theta ----
        for gn in range(self.n_gn_steps):
            d0 = z()
            def fn_for(pix):
                def fn(delta):
                    p = {k: (theta[k] + delta[k]) if k in set(fkeys) else theta[k] for k in theta}
                    return self._g(svm, p, pix)
                return fn

            # residual r = target - g_theta(x); RHS b = J^T r; init CG
            with torch.no_grad():
                res_before = 0.0
                r_list = []
                for pix, tgt in cal:
                    r = tgt - self._g(svm, theta, pix)
                    r_list.append(r); res_before += (r ** 2).sum().item()

            def JTJ(v):
                out = z()
                for pix, _ in cal:
                    fn = fn_for(pix)
                    _, Jv = jvp(fn, (d0,), (v,))
                    _, vjp_fn = vjp(fn, d0)
                    Jt = vjp_fn(Jv)[0]
                    for k in fkeys:
                        out[k] += Jt[k]
                return out

            b = z()
            for (pix, _), r in zip(cal, r_list):
                fn = fn_for(pix)
                _, vjp_fn = vjp(fn, d0)
                Jt = vjp_fn(r)[0]
                for k in fkeys:
                    b[k] += Jt[k]

            # CG solve (J^T J + lam I) x = b
            x = z(); rk = {k: b[k].clone() for k in fkeys}; pk = {k: rk[k].clone() for k in fkeys}
            rs_old = dot(rk, rk)
            for it in range(self.cg_iters):
                Hp = JTJ(pk)
                for k in fkeys:
                    Hp[k] = Hp[k] + self.lam * pk[k]
                al = rs_old / (dot(pk, Hp) + 1e-20)
                x = axpy(al, pk, x)
                rk = axpy(-al, Hp, rk)
                rs_new = dot(rk, rk)
                if rs_new.sqrt() < 1e-6:
                    break
                pk = axpy(rs_new / rs_old, pk, rk); rs_old = rs_new

            # apply damped GN update
            for k in fkeys:
                theta[k] = theta[k] + self.step_size * x[k]
            with torch.no_grad():
                res_after = sum(((tgt - self._g(svm, theta, pix)) ** 2).sum().item() for pix, tgt in cal)
            log.info("[GN] step %d/%d: functional residual %.4f -> %.4f (%.1f%%)",
                     gn + 1, self.n_gn_steps, res_before, res_after,
                     100 * (1 - res_after / max(res_before, 1e-9)))

        # write merged params back into the student
        svm.load_state_dict({k: v.to(svm.state_dict()[k].dtype) for k, v in theta.items()}, strict=False)
        student.eval()
        for p in student.parameters():
            p.requires_grad_(False)
        return student
