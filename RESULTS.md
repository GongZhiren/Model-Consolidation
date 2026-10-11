# Reproduction map

Every reported number mapped to the command that produces it. Commands run through the
FusionBench CLI (`fusion_bench ...`) after the methods and configs are registered (see
`README.md`). `B/32` uses `modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8`, `L/14`
uses `clip-vit-large-patch14_TA8`, and `TALL-20` uses the twenty-task pools; set
`taskpool.base_model` to the matching backbone. Closed-form references (RegMean++, Iso-C,
TSV-M, Fisher, Task Arithmetic) are FusionBench's own methods run under the same pools.

Common B/32 suffix (`<MP/TP B/32>`):

```
modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8 \
taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
taskpool.base_model=openai/clip-vit-base-patch32
```

## High-accuracy endpoints

| Setting | Closed-form ref | E2E-Merge | Δ | Command |
|---|---|---|---|---|
| ViT-B/32 · 8 tasks | 84.2 (RegMean++) | 89.9 (8k) / 90.2 (12k) | +5.7 | `method=e2e_merge/clip_e2e_merge method.match=proj method.lr=5.0e-6 method.steps=8000` (seeds 0–3), and `method.steps=12000` |
| ViT-B/16 · 8 tasks | 87.3 (RegMean++) | 91.9 (8k) | +4.6 | `... method.steps=8000` with the B/16 pools |
| ViT-L/14 · 8 tasks | 90.9 (RegMean++) | 94.2 (8k) / 94.3 | +3.3 | `... method.steps=8000` with `<MP/TP L/14>` |
| ViT-L/14 · 20 tasks | 83.4 (Iso-C) | 92.9 | +9.5 | `... method.steps=16000 method.batch_size=16` with the TALL-20 pools |

Closed-form references: `method=regmean_plusplus/clip_regmean_plusplus` (and `iso_c`, etc.) under each pool.

## Data-budget crossover (five paired subsets per point)

Fixed calibration set of `N` distinct examples per task (`stream=false`):

```
method=e2e_merge/clip_e2e_merge method.match=proj method.stream=false \
method.num_cal_batches=$((N/32)) method.steps=8000 method.lr=5.0e-6 <MP/TP B/32>
```

Sweep `N in {32,64,128,256,512,1024}`, five paired seeds each:

| N | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|
| RegMean++ | **82.5** | **83.5** | 84.0 | 84.4 | 84.3 | 84.5 |
| E2E-Merge | 78.7 | 82.3 | **84.6** | **86.4** | **87.7** | **88.6** |

The ordering reverses at `N = 128`; the E2E advantage grows `+0.68 → +2.14 → +3.42 → +4.20`
from 128 to 1024, with five paired wins at every point from 128 on.

## Compute frontier (same hardware, N = 256, seed 0)

```
method=e2e_merge/clip_e2e_merge method.match=proj method.lr=5.0e-6 method.steps=$K <MP/TP B/32>
```

| Point | Updates | Merge-only time | Peak GiB | Acc. |
|---|---|---|---|---|
| RegMean++ | 0 | 26 s | 5.01 | 84.20 |
| E2E-Merge | 500 | 56 s | 6.18 | 84.44 |
| E2E-Merge | 1k | 81 s | 6.18 | 85.21 |
| E2E-Merge | 2k | 148 s | 6.18 | 85.93 |
| E2E-Merge | 4k | 280 s | 6.18 | 86.17 |
| E2E-Merge | 8k | 533 s | 6.18 | 86.48 |

E2E clears RegMean++ at the first measured 500-step point; gains taper after 2k.

## Memory frontier (block-wise)

```
method=e2e_merge/clip_blockwise_e2e method.group_size=$G <MP/TP B/32>
```

B/32: `85.3% @ 3.7 GiB → 89.8% @ 5.4 GiB`, with a full-network run at `90.2% @ ~8 GiB`.
L/14: `G=1 → 92.0 (~13 GiB)`, `G=4 → 93.0 (~15 GiB)`, full block-wise `94.2 (~31 GiB)`.

## Current-point Gauss–Newton (GN-Merge)

```
method=e2e_merge/clip_gn_merge method.init_from=<RegMean++ merged dir> method.n_gn_steps=6 <MP/TP B/32>
```

Six matrix-free GN steps from RegMean++ raise `84.3 → 86.9`, between layer-local regression and
sustained E2E (fixed-protocol 12k E2E reaches `90.2`).

## Consolidation-difficulty axis

Merge `k` experts (subset pools `..._TA{k}`): the E2E − RegMean++ margin grows
`2.6, 3.3, 4.4, 6.0` at `k in {2,4,6,8}` and `13.7` at `k=20` on B/32. For the same twenty
TALL tasks across backbones the margin over the best baseline is `9.5` (L/14), `12.0` (B/16),
`13.7` (B/32).

## Target-by-initialization and SAM control

```
method=e2e_merge/clip_e2e_merge method.match={proj|logit} method.init_scale={0.0|0.3}
method=e2e_merge/clip_e2e_merge method.match=logit method.sam_rho={0.02|0.05|0.10}
```

Representation supervision adds `1.6–2.2` points over the strongest matched logit control; the
best logit+SAM point reaches `89.1` (B/32), `1.1` below the representation endpoint. The
restricted-coefficient gradient merger AdaMerging (`method=adamerging/...`) reaches `82.6`.
Refining 8k steps from either task-arithmetic or RegMean++ init converges to `90.1`.

## Flan-T5 GLUE sweep (three paired seeds)

```
method=e2e_merge/flan_t5_e2e method.match=logit_kl   # iterative E2E output matching
method=regmean/flan_t5_regmean                        # closed-form RegMean
```

Across 32/128/256/512 matched inputs per task, closed-form RegMean wins all 12 paired runs
(mean advantage `0.76–0.94`). Both functional methods beat task arithmetic (`79.0`) and TIES
(`80.2`). Seven GLUE tasks on Flan-T5-base; a complementary regime where more input coverage
does not make refinement preferable.

## Label-free refinement predictor

See [`residual_predictor/README.md`](residual_predictor/README.md). Across all 70 four-task
subsets the held-out RegMean++ residual tracks the eventual E2E gain at Spearman `ρ = 0.956`
(`0.630` after additive task adjustment). Leave-one-task-out calibration predicts the 6k-step
gain at `0.265`-point RMSE and makes `92.9%` of refine-or-stop decisions at a 2.5-point
threshold (vs `73.9%` from initial accuracy, `57.5%` from task-vector cosine). The signal holds
with as few as 16 examples per task and across nine backbone/task-count settings (`ρ = 0.992`).

Final normalized-projection residuals for Task Arithmetic / RegMean++ / E2E are
`0.93 / 0.60 / 0.20`, matching their accuracy order `67.6 / 84.2 / 90.2`.
