# When Does Refinement Pay? A Budgeted Functional View of Model Consolidation

Reference implementation of the consolidation methods and the label-free refinement
predictor from the paper **"When Does Refinement Pay? A Budgeted Functional View of Model
Consolidation."**

Model consolidation combines task-specific experts into one deployable model. We treat method
choice as deciding **how faithfully to preserve expert behavior within a resource budget**.
Closed-form merging, current-point Gauss–Newton, block-wise refinement, and end-to-end
refinement are then operating points on **one task-indexed functional objective**

```
F(θ) = Σ_a π_a · E_{x∼D_a} || g_θ(x) − g_{θ_a}(x) ||²
```

where `g_θ` is the merged model's final representation and `g_{θ_a}` is expert `a`'s. The
objective is **label-free** (it matches expert outputs, not gold labels) but data-assisted.

---

## What's here

All methods are implemented as **drop-in algorithms for
[FusionBench](https://github.com/tanganke/fusion_bench)**, the standard model-merging harness,
so every method runs under identical data loading, checkpoints, and evaluation. Closed-form
baselines (RegMean / RegMean++, Fisher, Iso-C, TSV-M, Task Arithmetic) are FusionBench's own
methods; this repository adds the refinement endpoints and the predictor.

```
Model-Consolidation/
├── method/e2e_merge/                 Consolidation methods (FusionBench algorithms)
│   ├── clip_e2e_merge.py             E2E-Merge: CLIP representation distillation,
│   │                                 streaming, target = {proj, pool, logit}, optional SAM
│   ├── gn_merge.py                   GN-Merge: matrix-free damped Gauss–Newton refinement
│   ├── blockwise_e2e_merge.py        Block-wise variant (bounded-memory operating points)
│   ├── seq2seq_e2e_merge.py          Flan-T5 GLUE variant (hidden / logit-KL targets)
│   └── __init__.py
├── configs/                          Hydra configs for each method
│   ├── clip_e2e_merge.yaml
│   ├── clip_blockwise_e2e.yaml
│   ├── clip_gn_merge.yaml
│   └── flan_t5_e2e.yaml
├── residual_predictor/               Label-free refine-or-stop signal (see its README)
│   ├── make_residual_configs.py      Enumerate the 70 four-task subsets as pool configs
│   ├── measure_residual.py           Held-out functional residual of a closed-form merge
│   └── analyze_residual.py           Correlate the residual with realized refinement gain
├── scripts/                          Reproduction drivers (edit env vars, then run)
├── requirements.txt
├── RESULTS.md                        Command-to-number map for the reported results
└── LICENSE
```

---

## The operating-point spectrum

| Operating point | File | Cost | What it restores |
|---|---|---|---|
| Closed-form (RegMean++) | *FusionBench* | statistics pass + solve | layer-local activation match |
| Current-point quadratic (**GN-Merge**) | `gn_merge.py` | a few matrix-free CG solves | current-point cross-layer geometry |
| Block-wise refinement | `blockwise_e2e_merge.py` | one block group in memory | partial joint scope, memory-controlled |
| End-to-end (**E2E-Merge**) | `clip_e2e_merge.py` | full training trajectory | the full nonlinear objective |

Every operating point returns **one shared encoder** with no learned router, no task-specific
module, and the same per-example inference cost.

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Register the methods with FusionBench, either by copying them into a checkout,

```bash
cp -r method/e2e_merge  <fusion_bench>/fusion_bench/method/
cp    configs/*.yaml     <fusion_bench>/config/method/e2e_merge/
```

or by pointing `FUSIONBENCH_ROOT` at a checkout that already contains them (see
`scripts/env.sh`). The public CLIP experts and datasets are downloaded by FusionBench on first
use.

---

## Quick start

Merge the eight ViT-B/32 CLIP experts with `E2E-Merge` and evaluate under the standard harness:

```bash
fusion_bench \
  method=e2e_merge/clip_e2e_merge \
  method.lr=5.0e-6 method.steps=8000 method.match=proj method.init_scale=0.3 \
  modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
  taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
  taskpool.base_model=openai/clip-vit-base-patch32
```

Key `E2E-Merge` options (`configs/clip_e2e_merge.yaml`):

| Option            | Meaning                                                               |
|-------------------|----------------------------------------------------------------------|
| `match`           | `proj` (normalized projected representation, default), `pool`, `logit` (KL) |
| `steps`           | number of student updates                                            |
| `init_scale`      | task-arithmetic initialization scale `α`                             |
| `stream`          | `true` = fresh minibatch per step; `false` = fixed calibration set   |
| `num_cal_batches` | fixed-set size per task when `stream=false` (data-budget sweep)      |
| `init_from`       | path to a saved merged model to refine (e.g. a closed-form solution) |
| `sam_rho`         | `>0` enables sharpness-aware minimization on the logit target        |
| `seed`            | controls task-sampling and data-shuffle order                       |

`GN-Merge` and the block-wise variant take `method=e2e_merge/clip_gn_merge` and
`method=e2e_merge/clip_blockwise_e2e` respectively; see their configs for options.

---

## Reproducing the paper

`RESULTS.md` maps every reported number to the command that produces it. The `scripts/`
drivers wrap those commands; edit `scripts/env.sh` (paths and GPU), then run, for example:

```bash
source scripts/env.sh
bash scripts/run_endpoints.sh          # accuracy endpoints
bash scripts/run_data_frontier.sh      # data-budget crossover
bash scripts/run_iteration_frontier.sh # update-budget curve
bash scripts/run_blockwise.sh          # memory frontier
bash scripts/run_target_control.sh     # target-by-init and SAM control
```

The label-free refinement predictor has its own pipeline; see
[`residual_predictor/README.md`](residual_predictor/README.md).

All experiments run on a single GPU.

---

## Notes

- The methods are **label-free but data-assisted**: they use each task's unlabeled training
  inputs and repeated frozen-teacher queries, never labels.
- All numbers share one FusionBench harness (data loading, checkpoints, evaluation).
- Runs are deterministic given a fixed `seed`.
