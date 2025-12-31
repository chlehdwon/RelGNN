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

from model import RelTS_Model, RelGNN_Head
from dataset import EntityTimeSeriesBuilder, create_ar_dataloaders, create_strict_ar_dataloaders

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-f1")
parser.add_argument("--task", type=str, default="driver-top3")
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument(
    "--cache_dir",
    type=str,
    default=os.path.expanduser("/data/starlab/relbench_examples"),
)
parser.add_argument("--checkpoint_dir", type=str, default="/data/starlab/ckpts/relgnn/")
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
    choices=["relts", "snapshot"],
    help="Model type: 'relts' (full temporal model) or 'snapshot' (baseline, snapshot only)"
)
parser.add_argument(
    "--strict_temporal",
    action="store_true",
    help="Use strict temporal split boundaries (train uses train history, val uses train only, test uses train+val only)"
)
parser.add_argument("--save", action="store_true", help="Save results")

args = parser.parse_args()

checkpoint_path = Path(args.checkpoint_dir) / f"{args.dataset}_{args.task}.pth"
if not checkpoint_path.exists():
    checkpoint_path = Path(hf_hub_download(repo_id="tianlangchen/RelGNN", filename=f"{args.dataset}_{args.task}.pth", cache_dir=args.checkpoint_dir))
assert checkpoint_path.exists(), "Checkpoint not found. Please download the checkpoint first."

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_num_threads(1)
seed_everything(args.seed)

dataset: Dataset = get_dataset(args.dataset, download=True)
task: EntityTask = get_task(args.dataset, args.task, download=True)

model_config, loader_config = get_configs(args.dataset, args.task)

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

print(f"\nModel embedding dimension (from snapshot): {channels}")

# Print sequence length statistics
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
# Choose dataloader creation function based on strict_temporal flag
if args.strict_temporal:
    print("Using strict temporal split boundaries:")
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
    print("Using standard temporal setting (input from all previous history)")
    ar_loader_dict = create_ar_dataloaders(
        entity_sequences=builder.entity_sequences,
        split_indices=builder.split_indices,
        window_size=args.window_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        min_input_length=0, 
    )

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
    # Use RelGNN's head structure for direct parameter loading
    model = RelGNN_Head(
        channels=channels,
        num_classes=out_channels,
        dropout=args.dropout,
        use_relgnn_head=True,
    ).to(device)
    print(f"  Model: Snapshot-Only Baseline (with RelGNN head structure)")
    print(f"    Embedding dim: {channels}")
    
    # Load pre-trained head parameters from RelGNN checkpoint
    print(f"  Loading pre-trained head from: {checkpoint_path}")
    relgnn_state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Extract head parameters (head.lins.0.weight, head.lins.0.bias)
    head_state_dict = {k: v for k, v in relgnn_state_dict.items() if k.startswith('head.')}
    print(f"  ✓ Loaded {len(head_state_dict)} head parameters from pre-trained RelGNN")
else:
    raise ValueError(f"Unknown model type: {args.model}")

print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Optimizer
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
    
    for batch in tqdm(loader, desc="Training"):
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
    
    return total_loss / total_samples

# Evaluation function
@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    all_preds = []
    all_targets = []
    
    for batch in tqdm(loader, desc="Evaluating"):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # Forward pass
        logits = model(batch)
        target = batch['target_label']
        
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
    
    return avg_loss, metrics_dict

# Training loop
print(f"\nStarting training for {args.epochs} epochs...")
print("="*80)

best_val_metric = -float('inf') if higher_is_better else float('inf')
best_epoch = 0

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
        print(f"✓ New best model! (epoch {epoch}, {tune_metric}={val_metric:.4f})")

print("\n" + "="*80)
print(f"Training completed!")
print(f"Best epoch: {best_epoch}")
print(f"Best val {tune_metric}: {best_val_metric:.4f}")
print("="*80)

# Evaluate on test set
print("\nEvaluating on test set...")
test_loss, test_metrics = evaluate(model, ar_loader_dict['test'], loss_fn, device)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Metrics: {test_metrics}")
print(f"Test {tune_metric}: {test_metrics[tune_metric]:.4f}")
print("="*80)


if args.save:
    results_path = Path(args.results_path) / f"{args.dataset}_{args.task}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    # Load existing results if file exists
    if results_path.exists():
        with open(results_path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    # Add new result with timestamp as key
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hyperparams = json.dumps({
        "model": args.model,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "window_size": args.window_size,
    })
    all_results.setdefault(str(hyperparams), {})[timestamp] = {
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_metric": best_val_metric,
        "test_metrics": test_metrics,
    }

    # Save back to file
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_path}")