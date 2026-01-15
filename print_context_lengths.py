import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
from relbench.datasets import get_dataset
from relbench.tasks import get_task
from dataset import EntityTimeSeriesBuilder, create_ar_dataloaders, create_strict_ar_dataloaders, create_random_ar_dataloaders

def plot_context_lengths(dataset_name, task_name, split_lengths, save_dir, window_size):
    """
    Create a bar graph showing the distribution of context lengths for a task.
    Shows subplots for train, val, and test splits.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    fig.suptitle(f"Context Length Distribution: {dataset_name} / {task_name}", fontsize=16)
    
    splits = ['train', 'val', 'test']
    
    for i, split in enumerate(splits):
        lengths = split_lengths.get(split, [])
        if not lengths:
            axes[i].text(0.5, 0.5, "No Data", ha='center', va='center')
            axes[i].set_title(f"{split.upper()} (No data)")
            continue
            
        # Count frequencies of each length
        counts = Counter(lengths)
        x = list(range(window_size + 1))
        y = [counts.get(val, 0) for val in x]
        
        axes[i].bar(x, y, color='skyblue', edgecolor='navy', alpha=0.7)
        axes[i].set_title(f"{split.upper()} Split (N={len(lengths)})")
        axes[i].set_xlabel("Context Length")
        axes[i].set_ylabel("Frequency")
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot
    save_path = save_dir / f"{dataset_name}_{task_name}_context_distribution.png"
    plt.savefig(save_path)
    plt.close()
    print(f"  ✓ Plot saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot context length distributions for each task")
    parser.add_argument("--index_path", type=str, default="/data/relts/snapshots", help="Path to snapshot indices")
    parser.add_argument("--window_size", type=int, default=32, help="Maximum window size")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for processing")
    parser.add_argument("--mode", type=str, default="recent", choices=["recent", "random", "strict"], help="Sampling mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--analysis_dir", type=str, default="analysis", help="Directory to save plots")
    args = parser.parse_args()

    # Ensure analysis directory exists
    analysis_dir = Path("RelTS") / args.analysis_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # List of all datasets and tasks
    tasks = [
        ("rel-amazon", "item-churn"),
        ("rel-amazon", "user-churn"),
        ("rel-avito", "user-clicks"),
        ("rel-avito", "user-visits"),
        ("rel-f1", "driver-dnf"),
        ("rel-f1", "driver-top3"),
        ("rel-stack", "user-badge"),
        ("rel-stack", "user-engagement"),
    ]

    for dataset_name, task_name in tasks:
        print("\n" + "="*80)
        print(f"PROCESSING: {dataset_name} | {task_name}")
        print("="*80)

        try:
            # Load dataset and task
            dataset = get_dataset(dataset_name, download=True)
            task = get_task(dataset_name, task_name, download=True)

            # Build entity sequences
            builder = EntityTimeSeriesBuilder(
                index_path=args.index_path,
                dataset_name=dataset_name,
                task_name=task_name,
                task=task,
            )

            # Create dataloaders
            if args.mode == "recent":
                ar_loader_dict = create_ar_dataloaders(
                    entity_sequences=builder.entity_sequences,
                    split_indices=builder.split_indices,
                    window_size=args.window_size,
                    batch_size=args.batch_size,
                    num_workers=0,
                    min_input_length=0,
                )
            elif args.mode == "random":
                ar_loader_dict = create_random_ar_dataloaders(
                    entity_sequences=builder.entity_sequences,
                    split_indices=builder.split_indices,
                    window_size=args.window_size,
                    batch_size=args.batch_size,
                    num_workers=0,
                    min_input_length=0,
                    samples_per_epoch=None,
                    use_strict=False,
                    seed=args.seed,
                )
            elif args.mode == "strict":
                ar_loader_dict = create_strict_ar_dataloaders(
                    entity_sequences=builder.entity_sequences,
                    split_indices=builder.split_indices,
                    window_size=args.window_size,
                    batch_size=args.batch_size,
                    num_workers=0,
                    min_input_length=0,
                )

            # Collect lengths for all splits
            split_lengths = {}
            for split, loader in ar_loader_dict.items():
                lengths = []
                for batch in tqdm(loader, desc=f"  Collecting {split}", leave=False):
                    l = batch['input_mask'].sum(dim=1).cpu().numpy().tolist()
                    lengths.extend(l)
                split_lengths[split] = lengths

            # Generate and save plot
            plot_context_lengths(dataset_name, task_name, split_lengths, analysis_dir, args.window_size)

        except Exception as e:
            print(f"Error processing {dataset_name}/{task_name}: {e}")

if __name__ == "__main__":
    main()
