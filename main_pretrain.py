import os
# OMP_NUM_THREADS: openmp, OPENBLAS_NUM_THREADS: openblas, MKL_NUM_THREADS: mkl, VECLIB_MAXIMUM_THREADS: accelerate, NUMEXPR_NUM_THREADS: numexpr
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=6
# os.environ["OPENBLAS_NUM_THREADS"] = "2" # export OPENBLAS_NUM_THREADS=4
# os.environ["VECLIB_MAXIMUM_THREADS"] = "2" # export VECLIB_MAXIMUM_THREADS=4

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import gc
import math

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss, L1Loss
import torch.nn.functional as F
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_geometric.seed import seed_everything
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import matplotlib
matplotlib.use("Agg")

from relbench.base import Dataset, EntityTask, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

from relgnn.text_embedder import GloveTextEmbedding

from model import RelTS_Model
from util import analyze_by_sequence_length, plot_quartile_results, analyze_cold_start_gap
from dataset import EntityTimeSeriesBuilder, create_ar_dataloaders, create_random_ar_dataloaders

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
parser.add_argument(
    "--index_path",
    type=str,
    default="/data/relts/snapshots",
    help="Root path for saving snapshot indices"
)
parser.add_argument("--results_path", type=str, default="/data/relts/ckpts")
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
parser.add_argument(
    "--scheduler",
    type=str,
    default="none",
    choices=["none", "cosine"],
    help="Learning-rate scheduler type",
)
parser.add_argument(
    "--warmup_ratio",
    type=float,
    default=0.0,
    help="Warmup ratio for scheduler, applied over total optimization steps",
)
parser.add_argument(
    "--min_lr_ratio",
    type=float,
    default=0.1,
    help="Final LR ratio for cosine decay",
)
parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
parser.add_argument("--ff_dim", type=int, default=512, help="Feedforward dimension")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
parser.add_argument(
    "--loss_reweighting",
    type=str,
    default="none",
    choices=["none", "balanced"],
    help="Binary classification loss reweighting mode",
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
parser.add_argument("--report", action="store_true", help="Report results")
parser.add_argument("--report_path", type=str, default="./results")
parser.add_argument(
    "--ignore_train_config",
    action="store_true",
    help="If set, ignore train_config from relgnn.utils and use CLI arguments as-is.",
)
parser.add_argument("--tag", type=str, default="default", help="Tag for the experiment")
parser.add_argument(
    "--use_entity_embedding",
    action=argparse.BooleanOptionalAction,
    help="Use entity embeddings in model"
)

 
args = parser.parse_args()


def cli_flag_provided(flag_name: str) -> bool:
    option = f"--{flag_name}"
    return any(arg == option or arg.startswith(f"{option}=") for arg in sys.argv[1:])

# Use train_config from relgnn.utils.get_configs when available.
train_config = None
if not args.ignore_train_config:
    try:
        from relgnn.utils import get_configs
        res = get_configs(args.dataset, args.task, args.backbone)
        if res is not None and len(res) == 3:
            train_config = res[2]
    except Exception:
        pass

if train_config:
    explicit_cli = {
        "lr": cli_flag_provided("lr"),
        "weight_decay": cli_flag_provided("weight_decay"),
        "window_size": cli_flag_provided("window_size"),
        "seed": cli_flag_provided("seed"),
    }
    if "lr" in train_config and not explicit_cli["lr"]:
        args.lr = float(train_config["lr"])
    if "weight_decay" in train_config and not explicit_cli["weight_decay"]:
        args.weight_decay = float(train_config["weight_decay"])
    if "window_size" in train_config and not explicit_cli["window_size"]:
        args.window_size = int(train_config["window_size"])
    if "seed" in train_config and not explicit_cli["seed"]:
        args.seed = int(train_config["seed"])

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
    use_random_embedding=False,
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

if args.mode == "recent":
    print("Using 'recent' mode (standard temporal setting):")
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
    ar_loader_dict = create_random_ar_dataloaders(
        entity_sequences=builder.entity_sequences,
        split_indices=builder.split_indices,
        window_size=args.window_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        min_input_length=0,
        samples_per_epoch=None,  # Use all possible samples
        seed=args.seed,
    )
else:
    raise ValueError(f"Unknown mode: {args.mode}")

# Create model based on type
entity_embed_dim = builder.entity_embeddings.shape[1] if args.use_entity_embedding else None
model = RelTS_Model(
    channels=channels,
    task_type=task.task_type,
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

print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Optimizer (only for models that need training)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay,
)

steps_per_epoch = min(len(ar_loader_dict["train"]), args.max_steps_per_epoch)
total_optimization_steps = max(1, steps_per_epoch * args.epochs)
scheduler = None
if args.scheduler == "cosine":
    warmup_steps = int(total_optimization_steps * args.warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = (current_step - warmup_steps) / float(max(1, total_optimization_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

binary_class_weights = None
if task.task_type == TaskType.BINARY_CLASSIFICATION and args.loss_reweighting == "balanced":
    train_targets_np = train_table.df[task.target_col].to_numpy(dtype=np.float32)
    pos_count = float(train_targets_np.sum())
    neg_count = float(train_targets_np.shape[0] - pos_count)
    pos_weight = neg_count / max(pos_count, 1.0)
    neg_weight = pos_count / max(neg_count, 1.0)
    binary_class_weights = (float(neg_weight), float(pos_weight))
    print(
        "Using balanced BCE reweighting:",
        f"neg_weight={neg_weight:.6f}, pos_weight={pos_weight:.6f}",
    )


def compute_loss(logits, target):
    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        logits_flat = logits.squeeze(-1)
        if binary_class_weights is None:
            return loss_fn(logits_flat, target)
        raw_loss = F.binary_cross_entropy_with_logits(logits_flat, target.float(), reduction="none")
        neg_weight, pos_weight = binary_class_weights
        sample_weight = torch.where(
            target > 0.5,
            torch.full_like(target, pos_weight),
            torch.full_like(target, neg_weight),
        )
        return (raw_loss * sample_weight).mean()
    if task.task_type == TaskType.REGRESSION:
        return loss_fn(logits.squeeze(-1), target)
    return loss_fn(logits, target)

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
        loss = compute_loss(logits, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
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
            loss = compute_loss(logits, target)
            preds = torch.sigmoid(logits.squeeze(-1))
        elif task.task_type == TaskType.REGRESSION:
            loss = compute_loss(logits, target)
            preds = logits.squeeze(-1)
            if clamp_min is not None:
                preds = preds.clamp(clamp_min, clamp_max)
        else:  # MULTILABEL_CLASSIFICATION
            loss = compute_loss(logits, target)
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
print(f"\nStarting training for {args.epochs} epochs...")
print("="*80)
print("Early stopping patience: 5 epochs")
print(
    f"Scheduler: {args.scheduler}, warmup_ratio={args.warmup_ratio}, "
    f"min_lr_ratio={args.min_lr_ratio}"
)

best_val_metric = -float('inf') if higher_is_better else float('inf')
best_epoch = 0
best_state_dict = None
patience = 5
patience_counter = 0

for epoch in range(1, args.epochs + 1):
    print(f"\nEpoch {epoch}/{args.epochs}")
    print("-" * 80)
    current_lr = optimizer.param_groups[0]["lr"]
    print(f"Current LR: {current_lr:.8f}")
    
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
        best_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
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
if best_state_dict is not None:
    print(f"Loading best checkpoint from epoch {best_epoch} before test evaluation...")
    model.load_state_dict(best_state_dict)
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


if args.report:
    results_path = Path(args.report_path) / f"{args.dataset}_{args.task}.json"
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
        "backbone": args.backbone,
        "mode": args.mode,
        "tag": args.tag,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "warmup_ratio": args.warmup_ratio,
        "min_lr_ratio": args.min_lr_ratio,
        "loss_reweighting": args.loss_reweighting,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "window_size": args.window_size,
    })
    # Ensure JSON-serializable types (cast numpy scalars to Python float/int)
    metrics_for_json = {k: float(v) for k, v in test_metrics.items()}
    result_entry = {
        "seed": int(args.seed),
        "best_epoch": int(best_epoch),
        "test_metrics": metrics_for_json,
    }
    if best_val_metric is not None:
        result_entry["best_val_metric"] = float(best_val_metric)
    
    all_results.setdefault(str(hyperparams), {})[timestamp] = result_entry

    # Save back to file
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_path}")

if args.save:
    # Save best sequence model weights only
    ckpt_root = os.path.join(args.results_path, "transformers")
    os.makedirs(ckpt_root, exist_ok=True)
    ckpt_name = f"{args.dataset}_{args.task}_{args.backbone}.pth"
    ckpt_path = os.path.join(ckpt_root, ckpt_name)
    if best_state_dict is None:
        best_state_dict = model.state_dict()
    torch.save(best_state_dict, ckpt_path)
    print(f"Saved model checkpoint to: {ckpt_path}")

    # Extract CLS embeddings for retrieval (sorted by target timestamp)
    print("Extracting CLS embeddings for retrieval...")
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    model.eval()

    all_entity_ids = []
    all_timestamps = []
    all_cls = []
    with torch.no_grad():
        for split in ["train", "val", "test"]:
            for batch in tqdm(ar_loader_dict[split], desc=f"CLS encoding ({split})"):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                cls_emb = model.encode_cls(batch)
                all_entity_ids.append(batch["entity_id"].detach().cpu())
                all_timestamps.append(batch["target_timestamp"].detach().cpu())
                all_cls.append(cls_emb.detach().cpu())

    entity_ids = torch.cat(all_entity_ids, dim=0)
    timestamps = torch.cat(all_timestamps, dim=0)
    cls_embeddings = torch.cat(all_cls, dim=0)

    order = torch.from_numpy(np.lexsort((entity_ids.numpy(), timestamps.numpy())))
    entity_ids = entity_ids[order]
    timestamps = timestamps[order]
    cls_embeddings = cls_embeddings[order]

    retrieval_root = os.path.join(args.index_path, args.backbone, args.dataset, args.task)
    os.makedirs(retrieval_root, exist_ok=True)
    np.save(
        os.path.join(retrieval_root, "cls_embeddings.npy"),
        cls_embeddings.numpy().astype(np.float32, copy=False),
    )
    cls_meta = {
        "entity_ids": entity_ids.long(),
        "timestamps": timestamps.long(),
    }
    torch.save(cls_meta, os.path.join(retrieval_root, "cls_meta.pt"))
    key = (cls_meta["entity_ids"].numpy().astype(np.uint64) << np.uint64(32)) | (
        cls_meta["timestamps"].numpy().astype(np.uint64) & np.uint64(0xFFFFFFFF)
    )
    order_key = np.argsort(key, kind="mergesort")
    key_sorted = key[order_key]
    row_ids_sorted = np.arange(len(key), dtype=np.int64)[order_key]
    np.savez_compressed(
        os.path.join(retrieval_root, "cls_lookup.npz"),
        key_sorted=key_sorted,
        row_ids_sorted=row_ids_sorted,
    )
    print(f"Saved CLS retrieval data to: {retrieval_root}")

# Explicitly release memory at the end of the run to avoid buildup across repeated experiments.
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
