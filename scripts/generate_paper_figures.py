#!/usr/bin/env python3
"""
Generate publication-ready figures for the ablation study paper.
Creates three key figures:
1. Konkani BLEU vs k (both models)
2. Tunisian Arabic BLEU vs k (both models)  
3. Pivot vs No-Pivot comparison
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path

# Set publication-quality defaults
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.titlesize'] = 12
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 8

# Color scheme
COLORS = {
    'tower': '#2E86AB',  # Blue
    'hermes': '#A23B72',  # Purple/Magenta
    'pivot': '#06A77D',  # Teal
    'no_pivot': '#D45D79'  # Rose
}

# Paths
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / 'ablation_results'
OUTPUT_DIR = BASE_DIR / 'paper_updates' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_ablation_data():
    """Load all ablation study data."""
    # Konkani - Tower
    konkani_tower = pd.read_csv(RESULTS_DIR / 'konkani_600tokens' / 'ablation_summary.csv')
    konkani_tower['Model'] = 'TowerInstruct'
    konkani_tower['Language'] = 'Konkani'
    
    # Konkani - Hermes
    konkani_hermes = pd.read_csv(RESULTS_DIR / 'konkani_hermes_600tokens' / 'ablation_summary.csv')
    konkani_hermes['Model'] = 'Hermes'
    konkani_hermes['Language'] = 'Konkani'
    
    # Arabic - Tower
    arabic_tower = pd.read_csv(RESULTS_DIR / 'arabic_600tokens' / 'summary.csv')
    arabic_tower['Model'] = 'TowerInstruct'
    arabic_tower['Language'] = 'Tunisian Arabic'
    
    # Arabic - Hermes (use corrected 20251126 data if available, else fallback)
    arabic_hermes_path = RESULTS_DIR / 'arabic_hermes_20251126' / 'ablation_summary.csv'
    if not arabic_hermes_path.exists():
        arabic_hermes_path = RESULTS_DIR / 'arabic_hermes_600tokens' / 'summary.csv'
    arabic_hermes = pd.read_csv(arabic_hermes_path)
    arabic_hermes['Model'] = 'Hermes'
    arabic_hermes['Language'] = 'Tunisian Arabic'
    
    return konkani_tower, konkani_hermes, arabic_tower, arabic_hermes


def create_figure_1_konkani(konkani_tower, konkani_hermes):
    """Figure 1: Konkani BLEU vs k for both models."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    # Plot lines
    ax.plot(konkani_tower['k'], konkani_tower['BLEU'], 
            marker='o', color=COLORS['tower'], label='TowerInstruct-7B',
            linewidth=2.5, markersize=7)
    ax.plot(konkani_hermes['k'], konkani_hermes['BLEU'], 
            marker='s', color=COLORS['hermes'], label='Hermes-Llama-3-8B',
            linewidth=2.5, markersize=7)
    
    # Highlight optimal k ranges
    ax.axvspan(3, 5, alpha=0.1, color=COLORS['tower'], 
               label='TowerInstruct optimal range')
    ax.axvspan(5, 7, alpha=0.1, color=COLORS['hermes'], 
               label='Hermes optimal range')
    
    # Labels and formatting
    ax.set_xlabel('Number of Few-Shot Examples ($k$)', fontsize=11, fontweight='bold')
    ax.set_ylabel('BLEU Score', fontsize=11, fontweight='bold')
    ax.set_title('Konkani Translation: Impact of Few-Shot Examples\n(with Marathi Pivot Language)', 
                 fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(range(0, 11))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.legend(loc='upper right', framealpha=0.95, edgecolor='gray')
    
    # Add annotations for key insights
    ax.annotate('TowerInstruct\nCatastrophic\nFailure', 
                xy=(8, 0), xytext=(8.5, 2.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=8, color='red', ha='left')
    
    ax.annotate('+451%', 
                xy=(7, 8.22), xytext=(7.5, 7),
                arrowprops=dict(arrowstyle='->', color=COLORS['hermes'], lw=1.5),
                fontsize=9, fontweight='bold', color=COLORS['hermes'], ha='left')
    
    plt.tight_layout()
    
    # Save as both PDF and PNG
    fig.savefig(OUTPUT_DIR / 'konkani_ablation.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'konkani_ablation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created Figure 1: {OUTPUT_DIR / 'konkani_ablation.pdf'}")


def create_figure_2_arabic(arabic_tower, arabic_hermes):
    """Figure 2: Tunisian Arabic BLEU vs k for both models."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    # Plot lines
    ax.plot(arabic_tower['k'], arabic_tower['BLEU'], 
            marker='o', color=COLORS['tower'], label='TowerInstruct-7B',
            linewidth=2.5, markersize=7)
    ax.plot(arabic_hermes['k'], arabic_hermes['BLEU'], 
            marker='s', color=COLORS['hermes'], label='Hermes-Llama-3-8B',
            linewidth=2.5, markersize=7)
    
    # Labels and formatting
    ax.set_xlabel('Number of Few-Shot Examples ($k$)', fontsize=11, fontweight='bold')
    ax.set_ylabel('BLEU Score', fontsize=11, fontweight='bold')
    ax.set_title('Tunisian Arabic Translation: Impact of Few-Shot Examples\n(with MSA Pivot Language)', 
                 fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(range(0, 11))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 8)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.legend(loc='upper left', framealpha=0.95, edgecolor='gray')
    
    # Highlight that improvements are modest
    ax.text(8, 6.5, 'Modest improvements\n(11-46%)', 
            fontsize=9, color='gray', style='italic',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout()
    
    # Save as both PDF and PNG
    fig.savefig(OUTPUT_DIR / 'arabic_ablation.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'arabic_ablation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created Figure 2: {OUTPUT_DIR / 'arabic_ablation.pdf'}")


def create_figure_3_pivot_comparison():
    """Figure 3: Pivot vs No-Pivot comparison (base models, k=5 ablation results)."""
    # "With Pivot" data from ablation study k=5 results (base models)
    # "Without Pivot" data from original paper experiments
    data = {
        'Konkani\nHermes': {'With Pivot': 8.06, 'Without Pivot': 1.39},
        'Konkani\nTower': {'With Pivot': 7.41, 'Without Pivot': 1.39},
        'Arabic\nHermes': {'With Pivot': 5.52, 'Without Pivot': 0.00},
        'Arabic\nTower': {'With Pivot': 4.02, 'Without Pivot': 0.00},
    }
    
    configs = list(data.keys())
    with_pivot = [data[c]['With Pivot'] for c in configs]
    without_pivot = [data[c]['Without Pivot'] for c in configs]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(configs))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, without_pivot, width, label='Without Pivot',
                   color=COLORS['no_pivot'], edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x + width/2, with_pivot, width, label='With Pivot Language',
                   color=COLORS['pivot'], edgecolor='black', linewidth=0.8)
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    autolabel(bars1)
    autolabel(bars2)
    
    # Calculate and display improvement percentages
    for i, config in enumerate(configs):
        wp = with_pivot[i]
        wop = without_pivot[i]
        if wop > 0:
            improvement = ((wp - wop) / wop) * 100
            ax.text(i, max(wp, wop) + 0.5, f'+{improvement:.0f}%',
                   ha='center', fontsize=8, color='green', fontweight='bold')
    
    # Labels and formatting
    ax.set_xlabel('Model Configuration', fontsize=11, fontweight='bold')
    ax.set_ylabel('BLEU Score', fontsize=11, fontweight='bold')
    ax.set_title('Impact of Pivot Language on Translation Performance\n(Base Models, k=5 few-shot examples)', 
                 fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=0, ha='center', fontsize=9)
    ax.set_ylim(0, 14)
    ax.legend(loc='upper left', framealpha=0.95, edgecolor='gray', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
    
    # Add a horizontal line at y=0 for clarity
    ax.axhline(y=0, color='black', linewidth=0.8)
    
    plt.tight_layout()
    
    # Save as both PDF and PNG
    fig.savefig(OUTPUT_DIR / 'pivot_comparison.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'pivot_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created Figure 3: {OUTPUT_DIR / 'pivot_comparison.pdf'}")


def main():
    """Generate all figures."""
    print("=" * 60)
    print("Generating Publication-Ready Figures for Ablation Study")
    print("=" * 60)
    
    # Load data
    print("\n📊 Loading ablation study data...")
    konkani_tower, konkani_hermes, arabic_tower, arabic_hermes = load_ablation_data()
    print(f"   ✓ Loaded Konkani data: {len(konkani_tower)} Tower, {len(konkani_hermes)} Hermes")
    print(f"   ✓ Loaded Arabic data: {len(arabic_tower)} Tower, {len(arabic_hermes)} Hermes")
    
    # Create figures
    print("\n🎨 Generating figures...")
    create_figure_1_konkani(konkani_tower, konkani_hermes)
    create_figure_2_arabic(arabic_tower, arabic_hermes)
    create_figure_3_pivot_comparison()
    
    print("\n" + "=" * 60)
    print("✅ All figures generated successfully!")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print("=" * 60)
    print("\nGenerated files:")
    print("  • konkani_ablation.pdf/.png - Konkani k ablation")
    print("  • arabic_ablation.pdf/.png - Arabic k ablation")
    print("  • pivot_comparison.pdf/.png - Pivot vs no-pivot")
    print("\nUse the .pdf versions in your LaTeX paper for best quality.")


if __name__ == '__main__':
    main()

