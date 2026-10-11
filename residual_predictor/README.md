# Label-free refinement predictor

A closed-form merge's **held-out functional residual** predicts how much end-to-end refinement
will help, *before any refinement budget is spent and without any labels*. This turns the
budgeted regime condition into an objective-aligned **refine-or-stop triage rule**.

The study enumerates all **70 four-task subsets** of the ViT-B/32 eight-task pool. For each
subset we fit RegMean++ on 256 examples per task, measure its squared normalized-projection
residual on the next disjoint 256 examples (no labels), then run a fixed 6k-step E2E refinement
and record the accuracy gain over RegMean++.

Headline result: Spearman **ρ = 0.96** between the residual and the eventual gain (0.63 after
additive task adjustment). Under leave-one-task-out calibration the residual predicts the gain
at **0.265-point RMSE** and makes **92.9%** of refine-or-stop decisions correctly at a 2.5-point
utility threshold, versus 73.9% from initial accuracy and 57.5% from task-vector cosine.

## Pipeline

```bash
# REPO = a FusionBench checkout that already has the e2e_merge method + configs registered
# OUT  = a working directory for the generated configs, per-subset residuals, and summary

# 1) enumerate all 70 four-task subsets as FusionBench pool configs
python make_residual_configs.py "$REPO" "$OUT"

# 2) for each subset s4_NN: fit RegMean++, save the merged model, then measure its residual
#    (RegMean++ is FusionBench's own method; save with report/model save options)
python measure_residual.py "$REPO" s4_00 /path/to/regmeanpp_s4_00 "$OUT/residual_s4_00.json"
#    ... repeat for s4_00 .. s4_69, and run the 6k-step E2E per subset for its realized gain

# 3) aggregate: correlations, in-sample R², leave-one-task-out RMSE and decision accuracy
python analyze_residual.py "$OUT"
```

`make_residual_configs.py` writes a `manifest.json` listing the 70 subsets and their task sets.
`measure_residual.py` holds out the 256 examples immediately following RegMean++'s seed-0
calibration prefix, so the residual never sees the data the merge was fit on. `analyze_residual.py`
reads the per-subset residual and accuracy JSONs under `OUT` and writes `summary.json` +
`results.csv`.

The residual is a measurable proxy; `make_residual_configs.py` enumerates the subsets up front
so no subset is selected after seeing its gain.
