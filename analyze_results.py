"""Analyze hyperparameter tuning results and export to CSV."""
import json
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd


def analyze_results(json_path):
    """Analyze tuning results and create CSV summary."""
    # Load results
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    # Group by setting and seed
    setting_results = defaultdict(lambda: defaultdict(list))
    
    for setting_str, timestamps in results.items():
        for timestamp, result in timestamps.items():
            seed = result['seed']
            roc_auc = result['test_metrics']['roc_auc']
            setting_results[setting_str][seed].append(roc_auc)
    
    # Calculate statistics
    rows = []
    for setting_str, seed_dict in setting_results.items():
        setting = json.loads(setting_str)
        
        # Average per seed (in case of duplicates)
        seed_means = [np.mean(values) for values in seed_dict.values()]
        
        row = {
            'model': setting['model'],
            'mode': setting.get('mode', 'recent'),
            'lr': setting['lr'],
            'weight_decay': setting['weight_decay'],
            'num_heads': setting['num_heads'],
            'num_layers': setting['num_layers'],
            'dropout': setting['dropout'],
            'window_size': setting['window_size'],
            'mean_roc_auc': np.mean(seed_means),
            'std_roc_auc': np.std(seed_means, ddof=1) if len(seed_means) > 1 else 0.0,
            'num_seeds': len(seed_means)
        }
        rows.append(row)
    
    # Create DataFrame and sort by mean ROC-AUC
    df = pd.DataFrame(rows)
    df = df.sort_values('mean_roc_auc', ascending=False).reset_index(drop=True)
    
    # Save to CSV (same directory as JSON)
    output_csv = Path(json_path).parent / f"{Path(json_path).stem}_summary.csv"
    df.to_csv(output_csv, index=False, float_format='%.6f')
    
    print(f"Results saved to: {output_csv}")
    print(f"\nTop 10 Settings:")
    print(df.head(10).to_string(index=False))
    print(f"\n{'='*80}")
    print("BEST SETTING:")
    print(f"{'='*80}")
    best = df.iloc[0]
    print(f"Model: {best['model']}")
    print(f"Mode: {best['mode']}")
    print(f"Learning Rate: {best['lr']}")
    print(f"Weight Decay: {best['weight_decay']}")
    print(f"Number of Heads: {best['num_heads']}")
    print(f"Number of Layers: {best['num_layers']}")
    print(f"Dropout: {best['dropout']}")
    print(f"Window Size: {best['window_size']}")
    print(f"\nMean ROC-AUC: {best['mean_roc_auc']:.6f} ± {best['std_roc_auc']:.6f}")
    print(f"Number of Seeds: {int(best['num_seeds'])}")
    print(f"{'='*80}")
    
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze hyperparameter tuning results')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., rel-f1)')
    parser.add_argument('--task', type=str, required=True, help='Task name (e.g., driver-top3)')
    
    args = parser.parse_args()
    
    # Construct JSON path
    json_path = f"results/{args.dataset}_{args.task}.json"
    
    if not Path(json_path).exists():
        print(f"Error: {json_path} not found!")
        print(f"Make sure you have run experiments for {args.dataset}/{args.task}")
        exit(1)
    
    df = analyze_results(json_path)
