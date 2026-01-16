import argparse
import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
from datetime import datetime
import pandas as pd
import torch
from torch.nn import BCEWithLogitsLoss, L1Loss
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from tqdm import tqdm
import os
from huggingface_hub import hf_hub_download
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from relbench.base import Dataset, EntityTask, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

from relgnn.relgnn_model import RelGNN_Model
from relgnn.text_embedder import GloveTextEmbedding
from relgnn.utils import get_configs
from relgnn.atomic_routes import get_atomic_routes

from model import RelTS_Model, MLP_Head, EntityMeanBaseline
from dataset import EntityTimeSeriesBuilder, create_ar_dataloaders, create_strict_ar_dataloaders, create_random_ar_dataloaders

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-f1")
parser.add_argument("--task", type=str, default="driver-top3")
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument(
    "--cache_dir",
    type=str,
    default=os.path.expanduser("/data/starlab/relbench_examples"),
)
parser.add_argument(
    "--backbone",
    type=str,
    default="relgnn",
    choices=["rdl", "relgnn", "relgt"],
    help="Backbone model type: 'rdl', 'relgnn', or 'relgt'"
)
parser.add_argument("--results_path", type=str, default="./results")
parser.add_argument(
    "--index_path",
    type=str,
    default="/data/relts/snapshots",
    help="Root path for saving snapshot indices"
)
parser.add_argument(
    "--window_size",
    type=int,
    default=32,
    help="Window size for auto-regressive samples"
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=512,
    help="Batch size for AR DataLoader"
)
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
parser.add_argument("--max_steps_per_epoch", type=int, default=2000, help="Maximum number of steps per epoch")
parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
parser.add_argument("--ff_dim", type=int, default=512, help="Feedforward dimension")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
parser.add_argument(
    "--model",
    type=str,
    default="relts",
    choices=["relts", "snapshot", "entity_mean"],
    help="Model type: 'relts' (full temporal model), 'snapshot' (baseline, snapshot only), or 'entity_mean' (baseline, mean of historical labels)"
)
parser.add_argument(
    "--mode",
    type=str,
    default="recent",
    choices=["recent", "random", "strict"],
    help="Sampling mode: 'recent' (standard temporal, default), 'random' (random samples per epoch), 'strict' (strict temporal boundaries)"
)
parser.add_argument("--verbose", action="store_true", help="Show detailed statistics (sequence stats and quartile analysis)")
parser.add_argument("--save", action="store_true", help="Save results")
parser.add_argument("--tag", type=str, default="default", help="Tag for the experiment")
parser.add_argument("--random_embedding", action="store_true", help="Use random embeddings instead of pretrained embeddings (for ablation)")

args = parser.parse_args()

# Construct checkpoint path: /data/relts/ckpts/{backbone}/{dataset}_{task}.pth
checkpoint_path = Path(f"/data/relts/ckpts/{args.backbone}") / f"{args.dataset}_{args.task}.pth"
if not checkpoint_path.exists():
    # Fallback to HuggingFace Hub download for relgnn
    if args.backbone == "relgnn":
        checkpoint_path = Path(hf_hub_download(repo_id="tianlangchen/RelGNN", filename=f"{args.dataset}_{args.task}.pth", cache_dir=f"/data/relts/ckpts/{args.backbone}"))
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please ensure the checkpoint is available for backbone={args.backbone}")
assert checkpoint_path.exists(), f"Checkpoint not found at {checkpoint_path}. Please download the checkpoint first."

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)

dataset: Dataset = get_dataset(args.dataset, download=True)
task: EntityTask = get_task(args.dataset, args.task, download=True)

model_config, loader_config = get_configs(args.dataset, args.task, args.backbone)

stypes_cache_path = Path(f"{args.cache_dir}/{args.dataset}/stypes.json")
try:
    with open(stypes_cache_path, "r") as f:
        col_to_stype_dict = json.load(f)
    for table, col_to_stype in col_to_stype_dict.items():
        for col, stype_str in col_to_stype.items():
            col_to_stype[col] = stype(stype_str)
except FileNotFoundError:
    col_to_stype_dict = get_stype_proposal(dataset.get_db())
    Path(stypes_cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(stypes_cache_path, "w") as f:
        json.dump(col_to_stype_dict, f, indent=2, default=str)

data, col_stats_dict = make_pkey_fkey_graph(
    dataset.get_db(),
    col_to_stype_dict=col_to_stype_dict,
    text_embedder_cfg=TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=device), batch_size=256
    ),
    cache_dir=f"{args.cache_dir}/{args.dataset}/materialized",
)

clamp_min, clamp_max = None, None
if task.task_type == TaskType.BINARY_CLASSIFICATION:
    out_channels = 1
    loss_fn = BCEWithLogitsLoss()
    tune_metric = "roc_auc"
    higher_is_better = True
elif task.task_type == TaskType.REGRESSION:
    out_channels = 1
    loss_fn = L1Loss()
    tune_metric = "mae"
    higher_is_better = False
    # Get the clamp value at inference time
    train_table = task.get_table("train")
    clamp_min, clamp_max = np.percentile(
        train_table.df[task.target_col].to_numpy(), [2, 98]
    )
elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
    out_channels = task.num_labels
    loss_fn = BCEWithLogitsLoss()
    tune_metric = "multilabel_auprc_macro"
    higher_is_better = True
else:
    raise ValueError(f"Task type {task.task_type} is unsupported")

# Get train table for entity history building
train_table = task.get_table("train")
val_table = task.get_table("val")
test_table = task.get_table("test", mask_input_cols=False)

# Create loaders for train, val and test
loader_dict: Dict[str, NeighborLoader] = {}
for split in ["train", "val", "test"]:
    table = task.get_table(split)
    table_input = get_node_train_table_input(table=table, task=task)
    entity_table = table_input.nodes[0]
    loader_dict[split] = NeighborLoader(
        data,
        num_neighbors=[int(loader_config['num_neighbors'] / 2**i) for i in range(loader_config['num_layers'])],
        time_attr="time",
        input_nodes=table_input.nodes,
        input_time=table_input.time,
        transform=table_input.transform,
        subgraph_type=loader_config['subgraph_type'],
        batch_size=loader_config['batch_size'],
        temporal_strategy="last",  
        shuffle=split == "train", 
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

builder = EntityTimeSeriesBuilder(
    index_path=args.index_path,
    dataset_name=args.dataset,
    task_name=args.task,
    task=task,
    backbone=args.backbone,
    use_random_embedding=args.random_embedding,
)

# Get channel dimension from snapshot embeddings
channels = None
for split in ['train', 'val', 'test']:
    sequences = builder.entity_sequences[split]
    if len(sequences) > 0:
        first_entity_seq = next(iter(sequences.values()))
        if len(first_entity_seq) > 0:
            channels = first_entity_seq[0][1].shape[0]
            break

if channels is None:
    raise ValueError("Could not determine embedding dimension from snapshots")

if args.random_embedding:
    print(f"\n⚠️  Using RANDOM embeddings instead of pretrained embeddings!")
    print(f"Model embedding dimension: {channels}")
    print("="*80 + "\n")
else:
    print(f"\nModel embedding dimension (from snapshot): {channels}")

# Print sequence length statistics (only if verbose)
if args.verbose:
    print("\n" + "="*80)
    print("Entity Sequence Statistics")
    print("="*80)
    for split in ['train', 'val', 'test']:
        sequences = builder.entity_sequences[split]
        seq_lengths = [len(seq) for seq in sequences.values()]
        
        if len(seq_lengths) > 0:
            print(f"\n{split.upper()} Split:")
            print(f"  Number of entities: {len(sequences)}")
            print(f"  Average sequence length: {np.mean(seq_lengths):.2f}")
            print(f"  Median sequence length: {np.median(seq_lengths):.0f}")
            print(f"  Min sequence length: {np.min(seq_lengths)}")
            print(f"  Max sequence length: {np.max(seq_lengths)}")
            print(f"  Std sequence length: {np.std(seq_lengths):.2f}")
    print("="*80 + "\n")

# Get entity sequences and split indices from builder
# Note: builder already has .entity_sequences and .split_indices as attributes
# Choose dataloader creation function based on mode
if args.mode == "recent":
    print("Using 'recent' mode (standard temporal setting):")
    print("  - Input from all previous history")
    ar_loader_dict = create_ar_dataloaders(
        entity_sequences=builder.entity_sequences,
        split_indices=builder.split_indices,
        window_size=args.window_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        min_input_length=0, 
    )
elif args.mode == "random":
    print("Using 'random' mode (random sampling per epoch):")
    print("  - Each epoch generates different random samples")
    print("  - Input from all previous history")
    ar_loader_dict = create_random_ar_dataloaders(
        entity_sequences=builder.entity_sequences,
        split_indices=builder.split_indices,
        window_size=args.window_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        min_input_length=0,
        samples_per_epoch=None,  # Use all possible samples
        use_strict=False,  # Use standard temporal boundaries
        seed=args.seed,
    )
elif args.mode == "strict":
    print("Using 'strict' mode (strict temporal split boundaries):")
    print("  - Train: input from train history only")
    print("  - Val: input from train split only")
    print("  - Test: input from train + val splits only")
    ar_loader_dict = create_strict_ar_dataloaders(
        entity_sequences=builder.entity_sequences,
        split_indices=builder.split_indices,
        window_size=args.window_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        min_input_length=0, 
    )
else:
    raise ValueError(f"Unknown mode: {args.mode}")

# Create model based on type
if args.model == 'relts':
    model = RelTS_Model(
        channels=channels,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        num_classes=out_channels,
    ).to(device)
    print(f"  Model: RelTS (Temporal Sequence Model)")
    print(f"    Embedding dim: {channels}")
elif args.model == 'snapshot':
    # Use backbone model's head structure for direct parameter loading
    model = MLP_Head(
        channels=channels,
        num_classes=out_channels,
    ).to(device)
    print(f"  Model: Snapshot-Only Baseline (with {args.backbone} head structure)")
    print(f"    Embedding dim: {channels}")
    
    # Load pre-trained head parameters from backbone checkpoint
    print(f"  Loading pre-trained head from: {checkpoint_path}")
    backbone_state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Extract head parameters (head.lins.0.weight, head.lins.0.bias)
    head_state_dict = {k: v for k, v in backbone_state_dict.items() if k.startswith('head.')}
    model.load_state_dict(head_state_dict, strict=False)
    print(f"  ✓ Loaded {len(head_state_dict)} head parameters from pre-trained {args.backbone}")
elif args.model == 'entity_mean':
    # Simple baseline: no learnable parameters, just uses mean of historical labels
    model = EntityMeanBaseline(
        channels=channels,
        num_classes=out_channels,
    ).to(device)
    print(f"  Model: Entity Mean Baseline (no training needed)")
    print(f"    Embedding dim: {channels}")
else:
    raise ValueError(f"Unknown model type: {args.model}")

print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Optimizer (only for models that need training)
optimizer = None
if args.model in ['relts', 'snapshot']:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

# Training function
def train(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    total_samples = 0
    steps = 0
    total_steps = min(len(loader), args.max_steps_per_epoch)
    
    for batch in tqdm(loader, desc="Training", total=total_steps):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # Forward pass
        logits = model(batch)
        target = batch['target_label']
        
        # Compute loss
        if task.task_type == TaskType.BINARY_CLASSIFICATION:
            loss = loss_fn(logits.squeeze(-1), target)
        elif task.task_type == TaskType.REGRESSION:
            loss = loss_fn(logits.squeeze(-1), target)
        else:  # MULTILABEL_CLASSIFICATION
            loss = loss_fn(logits, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * batch['target_label'].size(0)
        total_samples += batch['target_label'].size(0)
        
        steps += 1
        if steps >= args.max_steps_per_epoch:
            break
    
    return total_loss / total_samples

# Evaluation function
@torch.no_grad()
def evaluate(model, loader, loss_fn, device, return_sequence_lengths=False, return_preds_targets=False):
    model.eval()
    total_loss = 0
    total_samples = 0
    all_preds = []
    all_targets = []
    all_seq_lengths = []  # Track sequence lengths
    
    for batch in tqdm(loader, desc="Evaluating"):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # Forward pass
        logits = model(batch)
        target = batch['target_label']
        
        # Track sequence lengths (number of valid inputs)
        if return_sequence_lengths:
            seq_lengths = batch['input_mask'].sum(dim=1).cpu().numpy()  # (batch_size,)
            all_seq_lengths.append(seq_lengths)
        
        # Compute loss
        if task.task_type == TaskType.BINARY_CLASSIFICATION:
            loss = loss_fn(logits.squeeze(-1), target)
            preds = torch.sigmoid(logits.squeeze(-1))
        elif task.task_type == TaskType.REGRESSION:
            loss = loss_fn(logits.squeeze(-1), target)
            preds = logits.squeeze(-1)
            if clamp_min is not None:
                preds = preds.clamp(clamp_min, clamp_max)
        else:  # MULTILABEL_CLASSIFICATION
            loss = loss_fn(logits, target)
            preds = torch.sigmoid(logits)
        
        total_loss += loss.item() * target.size(0)
        total_samples += target.size(0)
        all_preds.append(preds.cpu())
        all_targets.append(target.cpu())
    
    avg_loss = total_loss / total_samples
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    # Compute metrics using sklearn
    metrics_dict = {}
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        metrics_dict['roc_auc'] = roc_auc_score(all_targets, all_preds)
        metrics_dict['accuracy'] = accuracy_score(all_targets, (all_preds > 0.5).astype(int))
        metrics_dict['f1'] = f1_score(all_targets, (all_preds > 0.5).astype(int))
    elif task.task_type == TaskType.REGRESSION:
        metrics_dict['mae'] = mean_absolute_error(all_targets, all_preds)
        metrics_dict['rmse'] = np.sqrt(mean_squared_error(all_targets, all_preds))
        metrics_dict['r2'] = r2_score(all_targets, all_preds)
    
    # Return values based on flags
    return_values = [avg_loss, metrics_dict]
    if return_sequence_lengths:
        all_seq_lengths = np.concatenate(all_seq_lengths)
        return_values.append(all_seq_lengths)
    if return_preds_targets:
        return_values.extend([all_preds, all_targets])
    
    if len(return_values) == 2:
        return avg_loss, metrics_dict
    else:
        return tuple(return_values)


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
        x_labels.append(label)  # e.g., Q1 (0-0)
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

# Training loop (skip for models that don't need training)
if args.model in ['entity_mean']:
    print("\n" + "="*80)
    if args.model == 'snapshot':
        print("Snapshot model: Skipping training, evaluating pre-trained model directly")
    elif args.model == 'entity_mean':
        print("Entity Mean Baseline: Skipping training, using mean of historical labels")
    print("="*80)
    best_val_metric = None
    best_epoch = 0
else:
    print(f"\nStarting training for {args.epochs} epochs...")
    print("="*80)
    print("Early stopping patience: 5 epochs")
    
    best_val_metric = -float('inf') if higher_is_better else float('inf')
    best_epoch = 0
    patience = 5
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 80)
        
        # Train
        train_loss = train(model, ar_loader_dict['train'], optimizer, loss_fn, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss, val_metrics = evaluate(model, ar_loader_dict['val'], loss_fn, device)
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Metrics: {val_metrics}")
        
        # Get primary metric
        val_metric = val_metrics[tune_metric]
        print(f"Val {tune_metric}: {val_metric:.4f}")
        
        # Track best
        is_best = (higher_is_better and val_metric > best_val_metric) or \
                  (not higher_is_better and val_metric < best_val_metric)
        
        if is_best:
            best_val_metric = val_metric
            best_epoch = epoch
            patience_counter = 0
            print(f"✓ New best model! (epoch {epoch}, {tune_metric}={val_metric:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s)")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
            print(f"Best model was at epoch {best_epoch} with {tune_metric}={best_val_metric:.4f}")
            break
    
    print("\n" + "="*80)
    print(f"Training completed!")
    print(f"Best epoch: {best_epoch}")
    if best_val_metric is not None:
        print(f"Best val {tune_metric}: {best_val_metric:.4f}")
    print("="*80)

# Evaluate on test set
print("\nEvaluating on test set...")
if args.verbose:
    test_loss, test_metrics, test_seq_lengths, test_preds, test_targets = evaluate(
        model, ar_loader_dict['test'], loss_fn, device, 
        return_sequence_lengths=True, return_preds_targets=True
    )
    # Analyze by sequence length quartiles
    quartile_results = analyze_by_sequence_length(
        test_preds, test_targets, test_seq_lengths, task.task_type
    )
    # Plot quartile trend as a line plot
    plot_path = Path(args.results_path) / f"{args.dataset}_{args.task}_quartiles.png"
    plot_quartile_results(quartile_results, plot_path)
else:
    test_loss, test_metrics = evaluate(
        model, ar_loader_dict['test'], loss_fn, device
    )
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Metrics: {test_metrics}")
print(f"Test {tune_metric}: {test_metrics[tune_metric]:.4f}")
print("="*80)


if args.save:
    results_path = Path(args.results_path) / f"{args.dataset}_{args.task}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    # Load existing results if file exists
    if results_path.exists():
        try:
            with open(results_path, "r") as f:
                content = f.read().strip()
                if not content:
                    # Empty file
                    all_results = {}
                else:
                    all_results = json.loads(content)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Failed to parse existing results file {results_path}: {e}")
            print("Starting with empty results dictionary.")
            all_results = {}
    else:
        all_results = {}

    # Add new result with timestamp as key
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hyperparams = json.dumps({
        "model": args.model,
        "backbone": args.backbone,
        "mode": args.mode,
        "tag": args.tag,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "window_size": args.window_size,
    })
    result_entry = {
        "seed": args.seed,
        "best_epoch": best_epoch,
        "test_metrics": test_metrics,
    }
    if best_val_metric is not None:
        result_entry["best_val_metric"] = best_val_metric
    
    all_results.setdefault(str(hyperparams), {})[timestamp] = result_entry

    # Save back to file
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_path}")