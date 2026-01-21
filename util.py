import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
from relbench.base import TaskType


def analyze_by_sequence_length(all_preds, all_targets, all_seq_lengths, task_type):
    """
    Analyze performance by sequence length quartiles.
    Samples are sorted by sequence length and divided into 4 equal groups.
    
    Args:
        all_preds: Predictions array
        all_targets: Ground truth targets array
        all_seq_lengths: Sequence lengths array
        task_type: Task type (BINARY_CLASSIFICATION, REGRESSION, etc.)
    
    Returns:
        Dict with quartile analysis results
    """
    # Sort by sequence length
    sorted_indices = np.argsort(all_seq_lengths)
    sorted_preds = all_preds[sorted_indices]
    sorted_targets = all_targets[sorted_indices]
    sorted_seq_lengths = all_seq_lengths[sorted_indices]
    
    # Divide into 4 equal groups
    n_samples = len(all_preds)
    q1_end = n_samples // 4
    q2_end = n_samples // 2
    q3_end = 3 * n_samples // 4
    
    # Create quartile indices
    q1_indices = np.arange(0, q1_end)
    q2_indices = np.arange(q1_end, q2_end)
    q3_indices = np.arange(q2_end, q3_end)
    q4_indices = np.arange(q3_end, n_samples)
    
    quartiles = [
        ("Q1 (Shortest)", q1_indices),
        ("Q2", q2_indices),
        ("Q3", q3_indices),
        ("Q4 (Longest)", q4_indices),
    ]
    
    results = {}
    
    print("\n" + "="*80)
    print("Performance by Sequence Length Quartiles (Equal-sized groups)")
    print("="*80)
    
    for quartile_name, indices in quartiles:
        if len(indices) == 0:
            continue
        
        q_preds = sorted_preds[indices]
        q_targets = sorted_targets[indices]
        q_seq_lengths = sorted_seq_lengths[indices]
        
        # Calculate metric by quartile
        if task_type == TaskType.BINARY_CLASSIFICATION:
            # Use ROC-AUC for binary classification
            # Note: ROC-AUC is undefined if there is only one class present
            if len(np.unique(q_targets)) < 2:
                metric_name = "roc_auc"
                metric_value = float("nan")
            else:
                metric_name = "roc_auc"
                metric_value = roc_auc_score(q_targets, q_preds)
        elif task_type == TaskType.REGRESSION:
            # For regression, use accuracy based on threshold (or use MAE)
            # For now, we'll use a simple threshold-based accuracy
            # You might want to adjust this based on your needs
            threshold = np.median(all_targets)
            metric_name = "accuracy"
            metric_value = accuracy_score(
                (q_targets > threshold).astype(int),
                (q_preds > threshold).astype(int)
            )
        else:
            # For multilabel, use accuracy
            metric_name = "accuracy"
            metric_value = accuracy_score(q_targets, (q_preds > 0.5).astype(int))
        
        quartile_metrics = {
            metric_name: metric_value,
            "seq_min": float(q_seq_lengths.min()) if len(q_seq_lengths) > 0 else float("nan"),
            "seq_max": float(q_seq_lengths.max()) if len(q_seq_lengths) > 0 else float("nan"),
            "seq_avg": float(q_seq_lengths.mean()) if len(q_seq_lengths) > 0 else float("nan"),
        }
        results[quartile_name] = quartile_metrics
        
        print(f"\n{quartile_name}:")
        print(f"  Samples: {len(q_preds)} ({100*len(q_preds)/n_samples:.1f}%)")
        print(f"  Sequence length: {q_seq_lengths.min():.0f} - {q_seq_lengths.max():.0f} (avg: {q_seq_lengths.mean():.2f})")
        print(f"  {metric_name.upper()}: {metric_value:.4f}")
    
    print("="*80 + "\n")
    
    return results


def analyze_cold_start_gap(all_preds, all_targets, all_seq_lengths, task_type, top_k: int = 0):
    """
    Print combined analysis:
    - Cold-start definition based on top_k (or seq_len == 0)
    - Cold vs non-cold gap
    - Length buckets: Cold / Short (1-5) / Long (>5)
    """
    if len(all_preds) == 0:
        print("No samples available for cold-start analysis.")
        return

    if top_k > 0:
        cold_mask = (all_seq_lengths == top_k)
        cold_def = f"seq_len == top_k ({top_k})"
    else:
        cold_mask = (all_seq_lengths == 0)
        cold_def = "seq_len == 0"

    non_cold_mask = ~cold_mask

    def _compute_metric(preds, targets):
        if task_type == TaskType.BINARY_CLASSIFICATION:
            if len(np.unique(targets)) < 2:
                return "roc_auc", float("nan")
            return "roc_auc", roc_auc_score(targets, preds)
        elif task_type == TaskType.REGRESSION:
            threshold = np.median(all_targets)
            return "accuracy", accuracy_score(
                (targets > threshold).astype(int),
                (preds > threshold).astype(int)
            )
        else:
            return "accuracy", accuracy_score(targets, (preds > 0.5).astype(int))

    print("\n" + "=" * 80)
    print("Cold-Start vs Non-Cold-Start Performance")
    print("=" * 80)
    print(f"Cold-start definition: {cold_def}")

    cold_count = int(cold_mask.sum())
    non_cold_count = int(non_cold_mask.sum())
    print(f"Cold-start samples: {cold_count} / {len(all_preds)}")
    print(f"Non-cold-start samples: {non_cold_count} / {len(all_preds)}")

    if task_type == TaskType.BINARY_CLASSIFICATION:
        if cold_count > 0:
            cold_pos_ratio = float(np.mean(all_targets[cold_mask] > 0.5))
            print(f"Cold-start positive ratio: {cold_pos_ratio:.4f}")
        else:
            print("Cold-start positive ratio: None")
        if non_cold_count > 0:
            non_cold_pos_ratio = float(np.mean(all_targets[non_cold_mask] > 0.5))
            print(f"Non-cold-start positive ratio: {non_cold_pos_ratio:.4f}")
        else:
            print("Non-cold-start positive ratio: None")

    if cold_count == 0 or non_cold_count == 0:
        metric_name, _ = _compute_metric(
            all_preds[cold_mask] if cold_count > 0 else all_preds[non_cold_mask],
            all_targets[cold_mask] if cold_count > 0 else all_targets[non_cold_mask]
        )
        if cold_count == 0:
            print(f"Cold-start {metric_name.upper()}: None")
        if non_cold_count == 0:
            print(f"Non-cold-start {metric_name.upper()}: None")
        print("Gap (non-cold - cold): None")
    else:
        cold_metric_name, cold_metric_value = _compute_metric(
            all_preds[cold_mask], all_targets[cold_mask]
        )
        non_cold_metric_name, non_cold_metric_value = _compute_metric(
            all_preds[non_cold_mask], all_targets[non_cold_mask]
        )

        if cold_metric_name != non_cold_metric_name:
            print("Metric mismatch between groups; skipping gap report.")
        else:
            gap = non_cold_metric_value - cold_metric_value
            print(f"Cold-start {cold_metric_name.upper()}: {cold_metric_value:.4f}")
            print(f"Non-cold-start {non_cold_metric_name.upper()}: {non_cold_metric_value:.4f}")
            print(f"Gap (non-cold - cold): {gap:.4f}")

    print("=" * 80 + "\n")


def plot_quartile_results(quartile_results, save_path):
    """
    Create a simple line plot showing the metric trend across quartiles.
    Only uses the single metric present in quartile_results entries.
    """
    # Extract metric name and values in quartile order
    ordered = ["Q1 (Shortest)", "Q2", "Q3", "Q4 (Longest)"]
    x_labels = []
    y_values = []
    metric_name = None
    
    for name in ordered:
        if name not in quartile_results:
            continue
        metrics = quartile_results[name]
        if not metric_name:
            # Get the sole metric key
            metric_name = next(k for k in metrics.keys() if not k.startswith("seq_"))
        value = metrics.get(metric_name, float("nan"))
        seq_min = metrics.get("seq_min", float("nan"))
        seq_max = metrics.get("seq_max", float("nan"))
        label = f"{name.split()[0]} ({int(seq_min)}-{int(seq_max)})"
        value_str = f"{value:.4f}" if np.isfinite(value) else "nan"
        x_labels.append(f"{label}\n{value_str}")  # e.g., Q1 (0-0)\n0.8421
        y_values.append(value)
    
    if not x_labels or metric_name is None:
        print("No quartile results to plot.")
        return
    
    plt.figure(figsize=(6, 4))
    plt.plot(x_labels, y_values, marker="o")
    plt.title(f"{metric_name.upper()} by Sequence Length Quartile")
    plt.xlabel("Quartile")
    plt.ylabel(metric_name.upper())
    plt.ylim(0.0, 1.0) if metric_name == "roc_auc" else None
    plt.grid(True, linestyle="--", alpha=0.5)
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Quartile line plot saved to: {save_path}")