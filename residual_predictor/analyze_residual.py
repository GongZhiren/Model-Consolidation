"""Aggregate the preregistered 70-setting residual-prediction experiment."""
import csv
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr


def accuracy(path: Path) -> float:
    return 100.0 * json.loads(path.read_text())["average"]["accuracy"]


def correlation_dict(result) -> dict[str, float]:
    return {"statistic": float(result.statistic), "pvalue": float(result.pvalue)}


def r_squared(y: np.ndarray, prediction: np.ndarray) -> float:
    denominator = np.square(y - y.mean()).sum()
    return float(1.0 - np.square(y - prediction).sum() / denominator)


def loo_rmse(design: np.ndarray, y: np.ndarray) -> float:
    predictions = np.empty_like(y)
    for index in range(len(y)):
        keep = np.arange(len(y)) != index
        coefficients = np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]
        predictions[index] = design[index] @ coefficients
    return float(np.sqrt(np.square(y - predictions).mean()))


def linear_fit_predict(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray
) -> np.ndarray:
    """Ordinary least-squares calibration with an explicit intercept."""
    train_design = np.column_stack([np.ones(len(train_x)), train_x])
    test_design = np.column_stack([np.ones(len(test_x)), test_x])
    coefficients = np.linalg.lstsq(train_design, train_y, rcond=None)[0]
    return test_design @ coefficients


def balanced_accuracy(target: np.ndarray, prediction: np.ndarray) -> float:
    recalls = []
    for label in (False, True):
        mask = target == label
        recalls.append(float((prediction[mask] == label).mean()))
    return float(np.mean(recalls))


def leave_one_task_out(
    rows: list[dict], task_names: list[str], y: np.ndarray
) -> dict:
    """Test calibration on subsets containing a task absent from all fit rows."""
    predictors = {
        "residual": np.array([[row["residual"]] for row in rows]),
        "initial_accuracy": np.array(
            [[row["regmeanpp_accuracy"]] for row in rows]
        ),
        "task_vector_cosine": np.array(
            [[row["mean_abs_task_vector_cosine"]] for row in rows]
        ),
        "residual_plus_initial_accuracy": np.array(
            [[row["residual"], row["regmeanpp_accuracy"]] for row in rows]
        ),
    }
    task_sets = [set(row["tasks"].split(";")) for row in rows]
    useful_gain = 2.5
    output = {"useful_gain_threshold_points": useful_gain, "predictors": {}}
    for name, values in predictors.items():
        fold_rmse = []
        fold_accuracy = []
        fold_balanced_accuracy = []
        for task in task_names:
            test = np.array([task in tasks for tasks in task_sets])
            prediction = linear_fit_predict(values[~test], y[~test], values[test])
            fold_rmse.append(float(np.sqrt(np.square(y[test] - prediction).mean())))
            target_class = y[test] >= useful_gain
            predicted_class = prediction >= useful_gain
            fold_accuracy.append(float((target_class == predicted_class).mean()))
            fold_balanced_accuracy.append(
                balanced_accuracy(target_class, predicted_class)
            )
        output["predictors"][name] = {
            "mean_fold_rmse": float(np.mean(fold_rmse)),
            "mean_fold_decision_accuracy": float(np.mean(fold_accuracy)),
            "mean_fold_balanced_accuracy": float(
                np.mean(fold_balanced_accuracy)
            ),
            "fold_rmse": dict(zip(task_names, fold_rmse)),
            "fold_decision_accuracy": dict(zip(task_names, fold_accuracy)),
        }
    return output


def main(root: Path) -> None:
    pair_cosines = json.loads((root / "task_vector_cosines.json").read_text())
    rows = []
    for item in json.loads((root / "manifest.json").read_text()):
        subset_id = item["subset_id"]
        rm = accuracy(root / "reports" / f"{subset_id}_rmpp.json")
        e2e = accuracy(root / "reports" / f"{subset_id}_e2e.json")
        residual = json.loads((root / "residuals" / f"{subset_id}.json").read_text())[
            "squared_l2"
        ]
        cosine_values = [
            abs(pair_cosines["|".join(sorted(pair))])
            for pair in combinations(item["tasks"], 2)
        ]
        rows.append(
            {
                "subset_id": subset_id,
                "tasks": ";".join(item["tasks"]),
                "residual": residual,
                "mean_abs_task_vector_cosine": sum(cosine_values) / len(cosine_values),
                "regmeanpp_accuracy": rm,
                "e2e_accuracy": e2e,
                "e2e_gain": e2e - rm,
            }
        )
    x = np.array([row["residual"] for row in rows])
    y = np.array([row["e2e_gain"] for row in rows])
    rm_accuracy = np.array([row["regmeanpp_accuracy"] for row in rows])
    e2e_accuracy = np.array([row["e2e_accuracy"] for row in rows])
    cosine = np.array([row["mean_abs_task_vector_cosine"] for row in rows])
    task_names = sorted({task for item in json.loads((root / "manifest.json").read_text()) for task in item["tasks"]})
    task_design = np.array(
        [[float(task in row["tasks"].split(";")) for task in task_names] for row in rows]
    )
    # Every row has four active task indicators, so their sum already spans an
    # intercept.  Add the prospective diagnostics one at a time.
    residual_design = np.column_stack([np.ones(len(rows)), x])
    task_residual_design = np.column_stack([task_design, x])
    task_cosine_design = np.column_stack([task_design, cosine])
    task_rm_design = np.column_stack([task_design, rm_accuracy])
    task_rm_residual_design = np.column_stack([task_design, rm_accuracy, x])
    task_prediction = task_design @ np.linalg.lstsq(task_design, y, rcond=None)[0]
    task_residual_prediction = task_residual_design @ np.linalg.lstsq(
        task_residual_design, y, rcond=None
    )[0]
    task_cosine_prediction = task_cosine_design @ np.linalg.lstsq(
        task_cosine_design, y, rcond=None
    )[0]
    task_rm_prediction = task_rm_design @ np.linalg.lstsq(
        task_rm_design, y, rcond=None
    )[0]
    task_rm_residual_prediction = task_rm_residual_design @ np.linalg.lstsq(
        task_rm_residual_design, y, rcond=None
    )[0]
    x_residualized = x - task_design @ np.linalg.lstsq(task_design, x, rcond=None)[0]
    y_residualized = y - task_design @ np.linalg.lstsq(task_design, y, rcond=None)[0]
    rm_residualized = rm_accuracy - task_design @ np.linalg.lstsq(
        task_design, rm_accuracy, rcond=None
    )[0]
    e2e_residualized = e2e_accuracy - task_design @ np.linalg.lstsq(
        task_design, e2e_accuracy, rcond=None
    )[0]
    x_after_rm = x - task_rm_design @ np.linalg.lstsq(
        task_rm_design, x, rcond=None
    )[0]
    y_after_rm = y - task_rm_design @ np.linalg.lstsq(
        task_rm_design, y, rcond=None
    )[0]
    within_task = {}
    for task in task_names:
        mask = np.array([task in row["tasks"].split(";") for row in rows])
        within_task[task] = correlation_dict(spearmanr(x[mask], y[mask]))
    within_statistics = np.array(
        [within_task[task]["statistic"] for task in task_names]
    )
    order = np.argsort(x)
    quartile_size = len(rows) // 4
    loo_residual_only = loo_rmse(residual_design, y)
    loo_tasks = loo_rmse(task_design, y)
    loo_tasks_residual = loo_rmse(task_residual_design, y)
    loo_tasks_cosine = loo_rmse(task_cosine_design, y)
    loo_tasks_rm = loo_rmse(task_rm_design, y)
    loo_tasks_rm_residual = loo_rmse(task_rm_residual_design, y)
    bottom_quartile = float(y[order[:quartile_size]].mean())
    top_quartile = float(y[order[-quartile_size:]].mean())

    # Direction-only checks respect the combinatorial structure more directly
    # than treating the 70 overlapping subsets as independent replications.
    task_sets = [frozenset(row["tasks"].split(";")) for row in rows]
    subset_lookup = {tasks: index for index, tasks in enumerate(task_sets)}
    universe = frozenset(task_names)
    complement_pairs = []
    seen_pairs = set()
    one_swap_pairs = []
    for index, tasks in enumerate(task_sets):
        complement_index = subset_lookup[universe - tasks]
        pair = tuple(sorted((index, complement_index)))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            complement_pairs.append(pair)
        for other_index in range(index):
            if len(tasks.symmetric_difference(task_sets[other_index])) == 2:
                one_swap_pairs.append((other_index, index))

    def directional_count(pairs: list[tuple[int, int]]) -> int:
        return int(
            sum(
                np.sign(x[left] - x[right]) == np.sign(y[left] - y[right])
                for left, right in pairs
            )
        )

    summary = {
        "n": len(rows),
        "protocol": "all 4-of-8 subsets; ViT-B/32; RM++ N=256; E2E 6k steps",
        "spearman_residual_gain": correlation_dict(spearmanr(x, y)),
        "pearson_residual_gain": correlation_dict(pearsonr(x, y)),
        "spearman_residual_regmeanpp_accuracy": correlation_dict(
            spearmanr(x, rm_accuracy)
        ),
        "spearman_residual_e2e_accuracy": correlation_dict(
            spearmanr(x, e2e_accuracy)
        ),
        "spearman_cosine_gain": correlation_dict(spearmanr(cosine, y)),
        "spearman_after_additive_task_residualization": correlation_dict(
            spearmanr(x_residualized, y_residualized)
        ),
        "spearman_task_adjusted_residual_regmeanpp_accuracy": correlation_dict(
            spearmanr(x_residualized, rm_residualized)
        ),
        "spearman_task_adjusted_residual_e2e_accuracy": correlation_dict(
            spearmanr(x_residualized, e2e_residualized)
        ),
        "spearman_after_tasks_and_regmeanpp_accuracy": correlation_dict(
            spearmanr(x_after_rm, y_after_rm)
        ),
        "in_sample_r2_additive_tasks": r_squared(y, task_prediction),
        "in_sample_r2_additive_tasks_plus_residual": r_squared(
            y, task_residual_prediction
        ),
        "in_sample_r2_additive_tasks_plus_cosine": r_squared(
            y, task_cosine_prediction
        ),
        "in_sample_r2_additive_tasks_plus_regmeanpp_accuracy": r_squared(
            y, task_rm_prediction
        ),
        "in_sample_r2_additive_tasks_plus_regmeanpp_accuracy_plus_residual": r_squared(
            y, task_rm_residual_prediction
        ),
        "loo_rmse_residual_only": loo_residual_only,
        "loo_rmse_additive_tasks": loo_tasks,
        "loo_rmse_additive_tasks_plus_residual": loo_tasks_residual,
        "loo_rmse_additive_tasks_plus_cosine": loo_tasks_cosine,
        "loo_rmse_additive_tasks_plus_regmeanpp_accuracy": loo_tasks_rm,
        "loo_rmse_additive_tasks_plus_regmeanpp_accuracy_plus_residual": loo_tasks_rm_residual,
        "loo_relative_reduction_tasks_to_tasks_plus_residual": float(
            1.0 - loo_tasks_residual / loo_tasks
        ),
        "bottom_residual_quartile_gain_mean": bottom_quartile,
        "top_residual_quartile_gain_mean": top_quartile,
        "top_minus_bottom_residual_quartile_gain": top_quartile - bottom_quartile,
        "within_task_spearman": within_task,
        "within_task_nonnegative_count": int((within_statistics >= 0).sum()),
        "within_task_spearman_median": float(np.median(within_statistics)),
        "complement_pair_count": len(complement_pairs),
        "complement_pair_directional_agreement": directional_count(complement_pairs),
        "one_swap_pair_count": len(one_swap_pairs),
        "one_swap_pair_directional_agreement": directional_count(one_swap_pairs),
        "gain_mean": float(y.mean()),
        "gain_min": float(y.min()),
        "gain_max": float(y.max()),
        "leave_one_task_out": leave_one_task_out(rows, task_names, y),
    }
    with (root / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: analyze_residual.py OUTPUT_ROOT")
    main(Path(sys.argv[1]))
