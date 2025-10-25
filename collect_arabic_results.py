#!/usr/bin/env python3
"""
Collect and analyze Arabic results from both Tower and Hermes models
"""
import json
import pandas as pd
from pathlib import Path

def collect_results(base_dir):
    """Collect scores from all k directories"""
    data = []
    base_path = Path(base_dir)
    
    for k in range(11):
        score_file = base_path / f"k_{k}" / f"scores_k{k}.json"
        if score_file.exists():
            with open(score_file) as f:
                scores = json.load(f)
                data.append({
                    'k': k,
                    'BLEU': scores.get('BLEU Score', 0),
                    'chrF': scores.get('chrF Score', 0),
                    'chrF++': scores.get('CHRF++ Score', 0),
                })
        else:
            print(f"Warning: {score_file} not found")
    
    return pd.DataFrame(data).sort_values('k')

# Collect Tower results
print("="*80)
print("ARABIC RESULTS - TOWER MODEL")
print("="*80)
tower_df = collect_results("ablation_results/arabic_600tokens")
print(tower_df.to_string(index=False))

# Collect Hermes results
print("\n" + "="*80)
print("ARABIC RESULTS - HERMES MODEL")
print("="*80)
hermes_df = collect_results("ablation_results/arabic_hermes_600tokens")
print(hermes_df.to_string(index=False))

# Save summaries
tower_df.to_csv("ablation_results/arabic_600tokens/summary.csv", index=False)
hermes_df.to_csv("ablation_results/arabic_hermes_600tokens/summary.csv", index=False)
print("\n✅ Saved summary files")

# Analysis
print("\n" + "="*80)
print("COMPARATIVE ANALYSIS")
print("="*80)

print("\nTower Model:")
print(f"  k=0 (baseline): BLEU={tower_df[tower_df['k']==0]['BLEU'].values[0]:.4f}")
best_tower = tower_df.loc[tower_df['BLEU'].idxmax()]
print(f"  Best: k={int(best_tower['k'])}, BLEU={best_tower['BLEU']:.4f}")
improvement_tower = (best_tower['BLEU'] - tower_df[tower_df['k']==0]['BLEU'].values[0]) / tower_df[tower_df['k']==0]['BLEU'].values[0] * 100
print(f"  Improvement: +{improvement_tower:.1f}%")

print("\nHermes Model:")
print(f"  k=0 (baseline): BLEU={hermes_df[hermes_df['k']==0]['BLEU'].values[0]:.4f}")
best_hermes = hermes_df.loc[hermes_df['BLEU'].idxmax()]
print(f"  Best: k={int(best_hermes['k'])}, BLEU={best_hermes['BLEU']:.4f}")
improvement_hermes = (best_hermes['BLEU'] - hermes_df[hermes_df['k']==0]['BLEU'].values[0]) / hermes_df[hermes_df['k']==0]['BLEU'].values[0] * 100
print(f"  Improvement: +{improvement_hermes:.1f}%")

print("\nModel Comparison:")
print(f"  Tower vs Hermes at k=0: {tower_df[tower_df['k']==0]['BLEU'].values[0]:.4f} vs {hermes_df[hermes_df['k']==0]['BLEU'].values[0]:.4f}")
print(f"  Tower vs Hermes (best): {best_tower['BLEU']:.4f} vs {best_hermes['BLEU']:.4f}")
winner = "Tower" if best_tower['BLEU'] > best_hermes['BLEU'] else "Hermes"
print(f"  Winner: {winner}")

