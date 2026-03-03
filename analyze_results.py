"""Analyze tuning results and export one summary CSV.

Preferred format:
- Single merged file: {dataset}_{task}.json
- Each result entry has "source" (e.g., pretrain | predict | only_predict | predict_vote)
"""
import json
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from relbench.base import TaskType
from relbench.tasks import get_task


def _select_metric_value(test_metrics, task_type):
    if task_type == TaskType.REGRESSION:
        if "mae" in test_metrics:
            return test_metrics["mae"]
        if "r2" in test_metrics:
            return test_metrics["r2"]
    else:
        if "roc_auc" in test_metrics:
            return test_metrics["roc_auc"]
        if "multilabel_auprc_macro" in test_metrics:
            return test_metrics["multilabel_auprc_macro"]
    return list(test_metrics.values())[0]


def _process_one_results(results, stage, task_type, higher_is_better):
    """Build summary rows from one results dict under a fixed stage label."""
    setting_results = defaultdict(lambda: defaultdict(list))
    for setting_str, timestamps in results.items():
        for timestamp, result in timestamps.items():
            seed = result["seed"]
            test_metrics = result["test_metrics"]
            metric_value = _select_metric_value(test_metrics, task_type)
            setting_results[setting_str][seed].append(metric_value)

    rows = []
    for setting_str, seed_dict in setting_results.items():
        setting = json.loads(setting_str)
        seed_to_mean = {seed: np.mean(values) for seed, values in seed_dict.items()}
        seed_means = list(seed_to_mean.values())
        if higher_is_better:
            best_seed = max(seed_to_mean.keys(), key=lambda s: seed_to_mean[s])
        else:
            best_seed = min(seed_to_mean.keys(), key=lambda s: seed_to_mean[s])
        best_seed_metric = seed_to_mean[best_seed]
        row = {
            "stage": stage,
            "backbone": setting.get("backbone", "relgnn"),
            "mode": setting.get("mode", "recent"),
            "tag": setting.get("tag", "default"),
            "lr": setting.get("lr"),
            "weight_decay": setting.get("weight_decay"),
            "num_heads": setting.get("num_heads"),
            "num_layers": setting.get("num_layers"),
            "dropout": setting.get("dropout"),
            "window_size": setting.get("window_size"),
            "top_k": setting.get("top_k"),
            "ref_baseline": setting.get("ref_baseline"),
            "mean_metric": np.mean(seed_means),
            "std_metric": np.std(seed_means, ddof=1) if len(seed_means) > 1 else 0.0,
            "num_seeds": len(seed_means),
            "best_seed": best_seed,
            "best_seed_metric": best_seed_metric,
        }
        rows.append(row)
    return rows


def _process_merged_results(results, task_type, higher_is_better):
    """Build summary rows from merged results dict using entry['source'] as stage."""
    # Group key: (stage, setting_str) -> seed -> list(metric)
    grouped = defaultdict(lambda: defaultdict(list))
    for setting_str, timestamps in results.items():
        for _timestamp, result in timestamps.items():
            stage = result.get("source")
            if stage is None:
                # Backward-compat: try stage from hyperparam blob, else pretrain default.
                try:
                    parsed = json.loads(setting_str)
                    stage = parsed.get("stage", "pretrain")
                except Exception:
                    stage = "pretrain"
            seed = result["seed"]
            test_metrics = result["test_metrics"]
            metric_value = _select_metric_value(test_metrics, task_type)
            grouped[(stage, setting_str)][seed].append(metric_value)

    rows = []
    for (stage, setting_str), seed_dict in grouped.items():
        setting = json.loads(setting_str)
        seed_to_mean = {seed: np.mean(values) for seed, values in seed_dict.items()}
        seed_means = list(seed_to_mean.values())
        if higher_is_better:
            best_seed = max(seed_to_mean.keys(), key=lambda s: seed_to_mean[s])
        else:
            best_seed = min(seed_to_mean.keys(), key=lambda s: seed_to_mean[s])
        best_seed_metric = seed_to_mean[best_seed]
        row = {
            "stage": stage,
            "backbone": setting.get("backbone", "relgnn"),
            "mode": setting.get("mode", "recent"),
            "tag": setting.get("tag", "default"),
            "lr": setting.get("lr"),
            "weight_decay": setting.get("weight_decay"),
            "num_heads": setting.get("num_heads"),
            "num_layers": setting.get("num_layers"),
            "dropout": setting.get("dropout"),
            "window_size": setting.get("window_size"),
            "top_k": setting.get("top_k"),
            "ref_baseline": setting.get("ref_baseline"),
            "mean_metric": np.mean(seed_means),
            "std_metric": np.std(seed_means, ddof=1) if len(seed_means) > 1 else 0.0,
            "num_seeds": len(seed_means),
            "best_seed": best_seed,
            "best_seed_metric": best_seed_metric,
        }
        rows.append(row)
    return rows


def analyze_results(results_dir, dataset, task):
    """Analyze merged results from one file."""
    results_dir = Path(results_dir)
    task_obj = get_task(dataset, task, download=True)
    higher_is_better = task_obj.task_type != TaskType.REGRESSION
    base_name = f"{dataset}_{task}"
    pretrain_path = results_dir / f"{base_name}.json"

    rows = []
    if pretrain_path.exists():
        with open(pretrain_path, "r") as f:
            pretrain_results = json.load(f)
        rows.extend(
            _process_merged_results(
                pretrain_results,
                task_type=task_obj.task_type,
                higher_is_better=higher_is_better,
            )
        )

    if not rows:
        print(f"Error: No result files found for {dataset}/{task}.")
        print(f"  Looked for: {pretrain_path}")
        exit(1)

    df = pd.DataFrame(rows)
    # Sort by stage (pretrain first) then by metric direction per task type.
    df = df.sort_values(
        ["stage", "mean_metric"],
        ascending=[True, not higher_is_better],
        kind="stable",
    ).reset_index(drop=True)

    output_csv = results_dir / f"{base_name}_summary.csv"
    df.to_csv(output_csv, index=False, float_format="%.6f")
    print(f"Results saved to: {output_csv}")

    # Print by stage
    for stage in df["stage"].unique():
        subset = df[df["stage"] == stage]
        print(f"\n--- {stage.upper()} (top 10) ---")
        print(subset.head(10).to_string(index=False))

    print(f"\n{'='*80}")
    print("BEST PER STAGE:")
    print(f"{'='*80}")
    for stage in df["stage"].unique():
        stage_df = df[df["stage"] == stage]
        if stage_df.empty:
            continue
        best = stage_df.iloc[0]
        print(f"\n[{stage.upper()}]")
        print(f"  tag: {best['tag']}, mode: {best['mode']}, backbone: {best['backbone']}")
        if best.get("ref_baseline") is not None:
            print(f"  ref_baseline: {best['ref_baseline']}")
        print(f"  lr: {best['lr']}, weight_decay: {best['weight_decay']}")
        print(f"  num_heads: {best['num_heads']}, num_layers: {best['num_layers']}, dropout: {best['dropout']}, window_size: {best['window_size']}")
        print(f"  mean_metric: {best['mean_metric']:.6f} ± {best['std_metric']:.6f} (seeds: {int(best['num_seeds'])})")
        print(f"  best_seed: {int(best['best_seed'])}, best_seed_metric: {best['best_seed_metric']:.6f}")
    print(f"{'='*80}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze hyperparameter tuning results")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., rel-f1)")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., driver-top3)")
    parser.add_argument("--results_path", type=str, default="results", help="Results directory (default: results)")
    args = parser.parse_args()

    analyze_results(args.results_path, args.dataset, args.task)
