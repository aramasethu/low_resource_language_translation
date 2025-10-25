#!/usr/bin/env python3
"""
Create comparison plots for Arabic results
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Load data
tower_df = pd.read_csv("ablation_results/arabic_600tokens/summary.csv")
hermes_df = pd.read_csv("ablation_results/arabic_hermes_600tokens/summary.csv")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: BLEU comparison
axes[0, 0].plot(tower_df['k'], tower_df['BLEU'], marker='o', linewidth=2, markersize=8, label='Tower', color='#2E86AB')
axes[0, 0].plot(hermes_df['k'], hermes_df['BLEU'], marker='s', linewidth=2, markersize=8, label='Hermes', color='#A23B72')
axes[0, 0].set_xlabel('k (Number of Few-Shot Examples)', fontsize=12)
axes[0, 0].set_ylabel('BLEU Score', fontsize=12)
axes[0, 0].set_title('Arabic Translation: BLEU Score vs k', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xticks(range(11))

# Plot 2: chrF comparison
axes[0, 1].plot(tower_df['k'], tower_df['chrF'], marker='o', linewidth=2, markersize=8, label='Tower', color='#2E86AB')
axes[0, 1].plot(hermes_df['k'], hermes_df['chrF'], marker='s', linewidth=2, markersize=8, label='Hermes', color='#A23B72')
axes[0, 1].set_xlabel('k (Number of Few-Shot Examples)', fontsize=12)
axes[0, 1].set_ylabel('chrF Score', fontsize=12)
axes[0, 1].set_title('Arabic Translation: chrF Score vs k', fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=11)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xticks(range(11))

# Plot 3: Bar chart comparison at key k values
k_values = [0, 3, 5, 9]
tower_bleu = [tower_df[tower_df['k']==k]['BLEU'].values[0] for k in k_values]
hermes_bleu = [hermes_df[hermes_df['k']==k]['BLEU'].values[0] for k in k_values]

x = range(len(k_values))
width = 0.35
axes[1, 0].bar([i - width/2 for i in x], tower_bleu, width, label='Tower', alpha=0.8, color='#2E86AB')
axes[1, 0].bar([i + width/2 for i in x], hermes_bleu, width, label='Hermes', alpha=0.8, color='#A23B72')
axes[1, 0].set_xlabel('k Value', fontsize=12)
axes[1, 0].set_ylabel('BLEU Score', fontsize=12)
axes[1, 0].set_title('Arabic: Tower vs Hermes at Key k Values', fontsize=14, fontweight='bold')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels([f'k={k}' for k in k_values])
axes[1, 0].legend(fontsize=11)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 4: Improvement over baseline
tower_improvement = [(row['BLEU'] - tower_df[tower_df['k']==0]['BLEU'].values[0]) / tower_df[tower_df['k']==0]['BLEU'].values[0] * 100 
                     for _, row in tower_df.iterrows()]
hermes_improvement = [(row['BLEU'] - hermes_df[hermes_df['k']==0]['BLEU'].values[0]) / hermes_df[hermes_df['k']==0]['BLEU'].values[0] * 100 
                      for _, row in hermes_df.iterrows()]

axes[1, 1].plot(tower_df['k'], tower_improvement, marker='o', linewidth=2, markersize=8, label='Tower', color='#2E86AB')
axes[1, 1].plot(hermes_df['k'], hermes_improvement, marker='s', linewidth=2, markersize=8, label='Hermes', color='#A23B72')
axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[1, 1].set_xlabel('k (Number of Few-Shot Examples)', fontsize=12)
axes[1, 1].set_ylabel('Improvement over Baseline (%)', fontsize=12)
axes[1, 1].set_title('Arabic: Few-Shot Learning Benefit', fontsize=14, fontweight='bold')
axes[1, 1].legend(fontsize=11)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xticks(range(11))

plt.tight_layout()
plt.savefig('arabic_model_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: arabic_model_comparison.png")

# Create individual plots for ablation study doc
fig2, ax = plt.subplots(figsize=(10, 6))
ax.plot(tower_df['k'], tower_df['BLEU'], marker='o', linewidth=2.5, markersize=10, label='Tower', color='#2E86AB')
ax.plot(hermes_df['k'], hermes_df['BLEU'], marker='s', linewidth=2.5, markersize=10, label='Hermes', color='#A23B72')
ax.set_xlabel('k (Number of Few-Shot Examples)', fontsize=14)
ax.set_ylabel('BLEU Score', fontsize=14)
ax.set_title('Arabic Translation: Model Comparison', fontsize=16, fontweight='bold')
ax.legend(fontsize=13, loc='best')
ax.grid(True, alpha=0.3)
ax.set_xticks(range(11))
plt.tight_layout()
plt.savefig('arabic_bleu_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: arabic_bleu_comparison.png")

plt.close('all')
print("\n✅ All plots created successfully!")

