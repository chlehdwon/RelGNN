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
from torch_geometric.seed import seed_everything
from tqdm import tqdm
import os
from huggingface_hub import hf_hub_download
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from relbench.base import Dataset, EntityTask, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

from relgnn.text_embedder import GloveTextEmbedding

from model import RelTS_Model, MLP_Head, EntityMeanBaseline
from util import analyze_by_sequence_length, plot_quartile_results, analyze_cold_start_gap
from retrieval_sup import RetrievalManager as SupRetrievalManager
from retrieval_unsup import RetrievalManager as UnsupRetrievalManager
from dataset import EntityTimeSeriesBuilder, create_ar_dataloaders, create_random_ar_dataloaders
from dataset_ret import RetrievalDataset, retrieval_collate_fn

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
    choices=["recent", "random"],
    help="Sampling mode: 'recent' (standard temporal, default), 'random' (random samples per epoch)"
)
parser.add_argument("--verbose", action="store_true", help="Show detailed statistics (sequence stats and quartile analysis)")
parser.add_argument("--save", action="store_true", help="Save results")
parser.add_argument("--tag", type=str, default="default", help="Tag for the experiment")
parser.add_argument("--random_embedding", action="store_true", help="Use random embeddings instead of pretrained embeddings (for ablation)")
parser.add_argument(
    "--use_entity_embedding",
    action=argparse.BooleanOptionalAction,
    help="Use entity embeddings in model and retrieval"
)

# Retrieval related arguments
parser.add_argument("--retrieval_epochs", type=int, default=5, help="Number of epochs for retrieval pre-training")
parser.add_argument("--top_k", type=int, default=5, help="Number of retrieved contexts")
parser.add_argument("--retrieval_lr", type=float, default=1e-3, help="Learning rate for retrieval pre-training")
parser.add_argument("--retrieval_batch_size", type=int, default=2048, help="Batch size for retrieval operations")
parser.add_argument("--random_retrieval", action="store_true", help="Use random retrieval instead of similarity search")
parser.add_argument(
    "--ret_type",
    type=str,
    default="sup",
    choices=["unsup", "sup"],
    help="Retrieval training type: 'unsup' (InfoNCE) or 'sup' (SupCon)",
)
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
    print(f"\nUsing RANDOM embeddings instead of pretrained embeddings!")
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

retrieval_manager = None
if args.top_k > 0:
    retrieval_dataset = RetrievalDataset(builder.entity_sequences['train'])
    retrieval_loader = torch.utils.data.DataLoader(
        retrieval_dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        collate_fn=retrieval_collate_fn,
        num_workers=0
    )

    retrieval_cls = SupRetrievalManager if args.ret_type == "sup" else UnsupRetrievalManager
    entity_embed_dim = builder.entity_embeddings.shape[1]
    retrieval_input_dim = channels + entity_embed_dim if args.use_entity_embedding else channels
    retrieval_manager = retrieval_cls(
        input_dim=retrieval_input_dim,
        device=device,
        lr=args.retrieval_lr,
        embed_dim=128,
        use_random_retrieval=args.random_retrieval,
        use_entity_embedding=args.use_entity_embedding
    )

    if args.retrieval_epochs > 0:
        for r_epoch in range(1, args.retrieval_epochs + 1):
            r_loss = retrieval_manager.train_epoch(retrieval_loader)
            print(f" [Retrieval] Epoch {r_epoch} | Loss: {r_loss:.4f}")

    retrieval_loader = torch.utils.data.DataLoader(
        retrieval_dataset, 
        batch_size=args.batch_size * 4,
        shuffle=True, 
        collate_fn=retrieval_collate_fn
    )
    retrieval_manager.build_index(retrieval_loader)

if args.mode == "recent":
    print("Using 'recent' mode (standard temporal setting):")
    ar_loader_dict = create_ar_dataloaders(
        entity_sequences=builder.entity_sequences,
        split_indices=builder.split_indices,
        window_size=args.window_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        min_input_length=0,
        retrieval_manager=retrieval_manager,
        top_k=args.top_k,
        retrieval_batch_size=args.retrieval_batch_size,
    )
elif args.mode == "random":
    print("Using 'random' mode (random sampling per epoch):")
    ar_loader_dict = create_random_ar_dataloaders(
        entity_sequences=builder.entity_sequences,
        split_indices=builder.split_indices,
        window_size=args.window_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        min_input_length=0,
        samples_per_epoch=None,  # Use all possible samples
        seed=args.seed,
        retrieval_manager=retrieval_manager,
        top_k=args.top_k,
        retrieval_batch_size=args.retrieval_batch_size,
    )
else:
    raise ValueError(f"Unknown mode: {args.mode}")

# Create model based on type
if args.model == 'relts':
    entity_embed_dim = builder.entity_embeddings.shape[1] if args.use_entity_embedding else None
    model = RelTS_Model(
        channels=channels,
        entity_embed_dim=entity_embed_dim,
        use_entity_embedding=args.use_entity_embedding,
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
    analyze_cold_start_gap(
        test_preds, test_targets, test_seq_lengths, task.task_type, top_k=args.top_k
    )
    # Plot quartile trend as a line plot
    plot_path = Path(args.results_path) / f"{args.dataset}_{args.task}_quartiles_{args.top_k}.png"
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
        "top_k": args.top_k,
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