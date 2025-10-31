#!/usr/bin/env python3
"""
Analyze sampling ablation study results.

This script:
1. Loads all experiment results
2. Creates comparison graphs (random vs semantic for each k)
3. Identifies problematic predictions
4. Generates a comprehensive analysis report
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import defaultdict
import re

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Paths
OUTPUT_DIR = Path("outputs/sampling_ablation")
ANALYSIS_DIR = Path("outputs/sampling_ablation_analysis")
ANALYSIS_DIR.mkdir(exist_ok=True)

def load_experiment_summary():
    """Load the experiment summary."""
    with open(OUTPUT_DIR / "experiment_summary.json", 'r') as f:
        return json.load(f)

def load_experiment_details(model, strategy, k):
    """Load detailed results for a specific experiment."""
    filename = f"{model}_{strategy}_k{k}.json"
    filepath = OUTPUT_DIR / filename
    
    if filepath.exists():
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def extract_metrics_by_model_strategy():
    """Extract metrics organized by model and strategy."""
    summary = load_experiment_summary()
    
    data = defaultdict(lambda: defaultdict(list))
    
    for result in summary['results']:
        model = result['model']
        strategy = result['strategy']
        k = result['k']
        
        if result['result']['status'] == 'success':
            metrics = result['result']['metrics']
            data[model][strategy].append({
                'k': k,
                'bleu': metrics['bleu'],
                'chrf': metrics['chrf'],
                'chrf_pp': metrics['chrf_pp'],
                'time_minutes': result['result']['time_minutes']
            })
    
    return data

def create_comparison_plots(data):
    """Create comparison plots for each model."""
    
    for model in ['tower', 'hermes']:
        model_name = "Tower" if model == "tower" else "Hermes"
        
        # Convert to DataFrames
        semantic_df = pd.DataFrame(data[model]['semantic']).sort_values('k')
        random_df = pd.DataFrame(data[model]['random']).sort_values('k')
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{model_name} Model: Semantic vs Random Retrieval', fontsize=16, fontweight='bold')
        
        # BLEU scores
        ax = axes[0, 0]
        ax.plot(semantic_df['k'], semantic_df['bleu'], 'o-', label='Semantic', linewidth=2, markersize=8)
        ax.plot(random_df['k'], random_df['bleu'], 's-', label='Random', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Few-Shot Examples (k)', fontsize=12)
        ax.set_ylabel('BLEU Score', fontsize=12)
        ax.set_title('BLEU Score Comparison', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, 11))
        
        # chrF scores
        ax = axes[0, 1]
        ax.plot(semantic_df['k'], semantic_df['chrf'], 'o-', label='Semantic', linewidth=2, markersize=8)
        ax.plot(random_df['k'], random_df['chrf'], 's-', label='Random', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Few-Shot Examples (k)', fontsize=12)
        ax.set_ylabel('chrF Score', fontsize=12)
        ax.set_title('chrF Score Comparison', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, 11))
        
        # chrF++ scores
        ax = axes[1, 0]
        ax.plot(semantic_df['k'], semantic_df['chrf_pp'], 'o-', label='Semantic', linewidth=2, markersize=8)
        ax.plot(random_df['k'], random_df['chrf_pp'], 's-', label='Random', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Few-Shot Examples (k)', fontsize=12)
        ax.set_ylabel('chrF++ Score', fontsize=12)
        ax.set_title('chrF++ Score Comparison', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, 11))
        
        # Inference time
        ax = axes[1, 1]
        ax.plot(semantic_df['k'], semantic_df['time_minutes'], 'o-', label='Semantic', linewidth=2, markersize=8)
        ax.plot(random_df['k'], random_df['time_minutes'], 's-', label='Random', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Few-Shot Examples (k)', fontsize=12)
        ax.set_ylabel('Inference Time (minutes)', fontsize=12)
        ax.set_title('Inference Time Comparison', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, 11))
        
        plt.tight_layout()
        plt.savefig(ANALYSIS_DIR / f'{model}_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved {model}_comparison.png")
        plt.close()
        
        # Create improvement/degradation plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'{model_name} Model: Semantic vs Random Performance Delta', fontsize=16, fontweight='bold')
        
        # Calculate differences (semantic - random)
        k_values = semantic_df['k'].values
        bleu_delta = semantic_df['bleu'].values - random_df['bleu'].values
        chrf_delta = semantic_df['chrf'].values - random_df['chrf'].values
        chrf_pp_delta = semantic_df['chrf_pp'].values - random_df['chrf_pp'].values
        
        # BLEU delta
        ax = axes[0]
        colors = ['green' if x > 0 else 'red' for x in bleu_delta]
        ax.bar(k_values, bleu_delta, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Number of Few-Shot Examples (k)', fontsize=12)
        ax.set_ylabel('BLEU Delta (Semantic - Random)', fontsize=12)
        ax.set_title('BLEU Score Delta', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(range(0, 11))
        
        # chrF delta
        ax = axes[1]
        colors = ['green' if x > 0 else 'red' for x in chrf_delta]
        ax.bar(k_values, chrf_delta, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Number of Few-Shot Examples (k)', fontsize=12)
        ax.set_ylabel('chrF Delta (Semantic - Random)', fontsize=12)
        ax.set_title('chrF Score Delta', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(range(0, 11))
        
        # chrF++ delta
        ax = axes[2]
        colors = ['green' if x > 0 else 'red' for x in chrf_pp_delta]
        ax.bar(k_values, chrf_pp_delta, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Number of Few-Shot Examples (k)', fontsize=12)
        ax.set_ylabel('chrF++ Delta (Semantic - Random)', fontsize=12)
        ax.set_title('chrF++ Score Delta', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(range(0, 11))
        
        plt.tight_layout()
        plt.savefig(ANALYSIS_DIR / f'{model}_delta.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved {model}_delta.png")
        plt.close()

def is_problematic_prediction(prediction, target):
    """
    Identify if a prediction is problematic.
    
    Criteria:
    - Empty or very short
    - Contains excessive repeated characters (spaces, symbols, etc)
    - Contains garbled Unicode replacement characters (�)
    - Contains nonsense repeated ASCII patterns
    - Only whitespace/newlines
    
    NOTE: Devanagari script (Marathi/Konkani) is VALID, not "special characters"
    """
    if not prediction or len(prediction.strip()) < 5:
        return True, "Empty or too short"
    
    # Check for excessive repetition of single character (especially spaces, newlines, symbols)
    # Allow Devanagari characters to repeat (they're valid text)
    if re.search(r'([\s\(\)\[\]\{\}\<\>])\1{10,}', prediction):
        return True, "Excessive repetition of whitespace/punctuation"
    
    # Check for Unicode replacement characters (�) - indicates encoding issues
    if '�' in prediction and prediction.count('�') > 3:
        return True, "Contains Unicode replacement characters (�)"
    
    # Check for repeated gibberish patterns like "(A (A (A" or "क क क"
    # But be careful not to flag valid repeated words
    if re.search(r'(\([A-Z]\s*){5,}', prediction):
        return True, "Repeated ASCII pattern gibberish"
    
    # Check if prediction is ONLY whitespace and newlines
    if re.match(r'^[\s\n\r\t]+$', prediction):
        return True, "Only whitespace/newlines"
    
    # Check for suspiciously short output with mostly English characters when it should be Devanagari
    # (Model might have failed to generate proper Konkani)
    stripped = prediction.strip()
    if len(stripped) < 20:
        # Count Devanagari characters (range: U+0900 to U+097F)
        devanagari_chars = sum(1 for c in stripped if '\u0900' <= c <= '\u097F')
        ascii_chars = sum(1 for c in stripped if ord(c) < 128)
        # If very short and mostly ASCII, it's suspicious
        if ascii_chars > devanagari_chars and ascii_chars > 10:
            return True, "Too short with mostly ASCII (expected Devanagari)"
    
    # Check for the same short phrase repeated many times
    words = prediction.split()
    if len(words) > 10:
        # Check if more than 50% of words are duplicates
        unique_words = set(words)
        if len(unique_words) / len(words) < 0.5:
            return True, "High word repetition rate"
    
    return False, None

def analyze_predictions():
    """Analyze predictions to find problematic cases."""
    
    problematic_cases = defaultdict(lambda: defaultdict(list))
    
    for model in ['tower', 'hermes']:
        for strategy in ['semantic', 'random']:
            for k in range(0, 11):
                details = load_experiment_details(model, strategy, k)
                
                if details and 'outputs' in details:
                    outputs = details['outputs']
                    
                    for idx, item in enumerate(outputs):
                        prediction = item.get('prediction', '')
                        target = item.get('target', '')
                        source = item.get('source', '')
                        pivot = item.get('pivot', '')
                        
                        is_bad, reason = is_problematic_prediction(prediction, target)
                        
                        if is_bad:
                            problematic_cases[model][f"{strategy}_k{k}"].append({
                                'index': idx,
                                'reason': reason,
                                'source': source[:100] + "..." if len(source) > 100 else source,
                                'pivot': pivot[:100] + "..." if len(pivot) > 100 else pivot,
                                'target': target[:100] + "..." if len(target) > 100 else target,
                                'prediction': prediction[:100] + "..." if len(prediction) > 100 else prediction,
                            })
    
    return problematic_cases

def generate_report(data, problematic_cases):
    """Generate a comprehensive analysis report."""
    
    report_lines = []
    
    report_lines.append("# Sampling Ablation Study: Analysis Report")
    report_lines.append("")
    report_lines.append(f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Summary statistics
    report_lines.append("## Summary Statistics")
    report_lines.append("")
    
    for model in ['tower', 'hermes']:
        model_name = "Tower" if model == "tower" else "Hermes"
        report_lines.append(f"### {model_name} Model")
        report_lines.append("")
        
        semantic_df = pd.DataFrame(data[model]['semantic']).sort_values('k')
        random_df = pd.DataFrame(data[model]['random']).sort_values('k')
        
        # Average scores
        report_lines.append("#### Average Scores Across All k Values")
        report_lines.append("")
        report_lines.append("| Strategy | Avg BLEU | Avg chrF | Avg chrF++ | Avg Time (min) |")
        report_lines.append("|----------|----------|----------|------------|----------------|")
        report_lines.append(f"| Semantic | {semantic_df['bleu'].mean():.2f} | {semantic_df['chrf'].mean():.2f} | {semantic_df['chrf_pp'].mean():.2f} | {semantic_df['time_minutes'].mean():.2f} |")
        report_lines.append(f"| Random   | {random_df['bleu'].mean():.2f} | {random_df['chrf'].mean():.2f} | {random_df['chrf_pp'].mean():.2f} | {random_df['time_minutes'].mean():.2f} |")
        report_lines.append("")
        
        # Best performing k
        best_semantic_bleu_k = semantic_df.loc[semantic_df['bleu'].idxmax(), 'k']
        best_random_bleu_k = random_df.loc[random_df['bleu'].idxmax(), 'k']
        
        report_lines.append("#### Best Performing k Values (by BLEU)")
        report_lines.append("")
        report_lines.append(f"- **Semantic**: k={best_semantic_bleu_k:.0f} (BLEU: {semantic_df.loc[semantic_df['k'] == best_semantic_bleu_k, 'bleu'].values[0]:.2f})")
        report_lines.append(f"- **Random**: k={best_random_bleu_k:.0f} (BLEU: {random_df.loc[random_df['k'] == best_random_bleu_k, 'bleu'].values[0]:.2f})")
        report_lines.append("")
        
        # Detailed table
        report_lines.append("#### Detailed Results by k Value")
        report_lines.append("")
        report_lines.append("| k | Semantic BLEU | Random BLEU | Semantic chrF | Random chrF | Semantic chrF++ | Random chrF++ |")
        report_lines.append("|---|---------------|-------------|---------------|-------------|-----------------|---------------|")
        
        for k in range(0, 11):
            sem_row = semantic_df[semantic_df['k'] == k].iloc[0]
            ran_row = random_df[random_df['k'] == k].iloc[0]
            report_lines.append(f"| {k} | {sem_row['bleu']:.2f} | {ran_row['bleu']:.2f} | {sem_row['chrf']:.2f} | {ran_row['chrf']:.2f} | {sem_row['chrf_pp']:.2f} | {ran_row['chrf_pp']:.2f} |")
        
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
    
    # Key findings
    report_lines.append("## Key Findings")
    report_lines.append("")
    
    for model in ['tower', 'hermes']:
        model_name = "Tower" if model == "tower" else "Hermes"
        report_lines.append(f"### {model_name} Model")
        report_lines.append("")
        
        semantic_df = pd.DataFrame(data[model]['semantic']).sort_values('k')
        random_df = pd.DataFrame(data[model]['random']).sort_values('k')
        
        # Calculate when semantic wins
        semantic_wins = sum((semantic_df['bleu'].values - random_df['bleu'].values) > 0)
        random_wins = sum((random_df['bleu'].values - semantic_df['bleu'].values) > 0)
        ties = 11 - semantic_wins - random_wins
        
        report_lines.append(f"1. **Overall Strategy Comparison (by BLEU):**")
        report_lines.append(f"   - Semantic retrieval wins in {semantic_wins}/11 cases")
        report_lines.append(f"   - Random retrieval wins in {random_wins}/11 cases")
        if ties > 0:
            report_lines.append(f"   - Ties: {ties}")
        report_lines.append("")
        
        # Average improvement
        avg_bleu_delta = (semantic_df['bleu'].values - random_df['bleu'].values).mean()
        avg_chrf_delta = (semantic_df['chrf'].values - random_df['chrf'].values).mean()
        
        if avg_bleu_delta > 0:
            report_lines.append(f"2. **Average Performance Delta:** Semantic retrieval performs {avg_bleu_delta:.2f} BLEU points better on average")
        else:
            report_lines.append(f"2. **Average Performance Delta:** Random retrieval performs {abs(avg_bleu_delta):.2f} BLEU points better on average")
        report_lines.append("")
        
        # Zero-shot baseline
        zero_shot_bleu = semantic_df[semantic_df['k'] == 0]['bleu'].values[0]
        report_lines.append(f"3. **Zero-shot Baseline:** BLEU = {zero_shot_bleu:.2f}")
        report_lines.append("")
        
        # Best improvement over zero-shot
        semantic_improvements = semantic_df['bleu'].values - zero_shot_bleu
        random_improvements = random_df['bleu'].values - zero_shot_bleu
        
        best_semantic_improvement_k = np.argmax(semantic_improvements)
        best_random_improvement_k = np.argmax(random_improvements)
        
        report_lines.append(f"4. **Best Improvement over Zero-shot:**")
        report_lines.append(f"   - Semantic: k={best_semantic_improvement_k} (+{semantic_improvements[best_semantic_improvement_k]:.2f} BLEU)")
        report_lines.append(f"   - Random: k={best_random_improvement_k} (+{random_improvements[best_random_improvement_k]:.2f} BLEU)")
        report_lines.append("")
        
        # Inference time analysis
        avg_semantic_time = semantic_df['time_minutes'].mean()
        avg_random_time = random_df['time_minutes'].mean()
        report_lines.append(f"5. **Inference Time:**")
        report_lines.append(f"   - Semantic avg: {avg_semantic_time:.2f} minutes")
        report_lines.append(f"   - Random avg: {avg_random_time:.2f} minutes")
        report_lines.append(f"   - Difference: {abs(avg_semantic_time - avg_random_time):.2f} minutes")
        report_lines.append("")
        
        report_lines.append("---")
        report_lines.append("")
    
    # Problematic predictions
    report_lines.append("## Problematic Predictions Analysis")
    report_lines.append("")
    report_lines.append("This section identifies cases where the model generated problematic or nonsensical predictions.")
    report_lines.append("")
    
    for model in ['tower', 'hermes']:
        model_name = "Tower" if model == "tower" else "Hermes"
        report_lines.append(f"### {model_name} Model")
        report_lines.append("")
        
        model_problems = problematic_cases[model]
        
        # Count problems by experiment
        problem_counts = {}
        for exp_key, problems in model_problems.items():
            if problems:
                problem_counts[exp_key] = len(problems)
        
        if problem_counts:
            report_lines.append("#### Problem Count by Experiment")
            report_lines.append("")
            report_lines.append("| Experiment | Number of Problematic Predictions | Problem Rate |")
            report_lines.append("|------------|-----------------------------------|--------------|")
            
            for exp_key in sorted(problem_counts.keys()):
                count = problem_counts[exp_key]
                rate = (count / 205) * 100  # 205 samples total
                report_lines.append(f"| {exp_key} | {count} | {rate:.1f}% |")
            
            report_lines.append("")
            
            # Show examples
            report_lines.append("#### Example Problematic Cases")
            report_lines.append("")
            
            # Get worst experiments (most problems)
            worst_exps = sorted(problem_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for exp_key, count in worst_exps:
                report_lines.append(f"##### {exp_key} ({count} problems)")
                report_lines.append("")
                
                problems = model_problems[exp_key][:3]  # Show first 3 examples
                
                for i, prob in enumerate(problems, 1):
                    report_lines.append(f"**Example {i}:** {prob['reason']}")
                    report_lines.append("")
                    report_lines.append(f"- **Source (English):** {prob['source']}")
                    report_lines.append(f"- **Pivot (Marathi):** {prob['pivot']}")
                    report_lines.append(f"- **Target (Konkani):** {prob['target']}")
                    report_lines.append(f"- **Prediction:** {prob['prediction']}")
                    report_lines.append("")
        else:
            report_lines.append("✅ No problematic predictions found for this model.")
            report_lines.append("")
        
        report_lines.append("---")
        report_lines.append("")
    
    # Recommendations
    report_lines.append("## Recommendations")
    report_lines.append("")
    
    for model in ['tower', 'hermes']:
        model_name = "Tower" if model == "tower" else "Hermes"
        semantic_df = pd.DataFrame(data[model]['semantic']).sort_values('k')
        random_df = pd.DataFrame(data[model]['random']).sort_values('k')
        
        best_overall_idx = pd.concat([
            semantic_df.assign(strategy='semantic'),
            random_df.assign(strategy='random')
        ])['bleu'].idxmax()
        
        best_overall = pd.concat([
            semantic_df.assign(strategy='semantic'),
            random_df.assign(strategy='random')
        ]).iloc[best_overall_idx]
        
        report_lines.append(f"### {model_name} Model")
        report_lines.append("")
        report_lines.append(f"**Best Configuration:** {best_overall['strategy']} retrieval with k={int(best_overall['k'])} examples")
        report_lines.append(f"- BLEU: {best_overall['bleu']:.2f}")
        report_lines.append(f"- chrF: {best_overall['chrf']:.2f}")
        report_lines.append(f"- chrF++: {best_overall['chrf_pp']:.2f}")
        report_lines.append("")
    
    # Write report
    report_path = ANALYSIS_DIR / "analysis_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✅ Saved analysis report to {report_path}")

def main():
    """Main analysis pipeline."""
    
    print("\n" + "="*80)
    print("SAMPLING ABLATION STUDY ANALYSIS")
    print("="*80 + "\n")
    
    print("📊 Loading experiment data...")
    data = extract_metrics_by_model_strategy()
    
    print("📈 Creating comparison plots...")
    create_comparison_plots(data)
    
    print("🔍 Analyzing predictions for problematic cases...")
    problematic_cases = analyze_predictions()
    
    print("📝 Generating analysis report...")
    generate_report(data, problematic_cases)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {ANALYSIS_DIR}")
    print(f"  - Tower comparison: {ANALYSIS_DIR / 'tower_comparison.png'}")
    print(f"  - Tower delta: {ANALYSIS_DIR / 'tower_delta.png'}")
    print(f"  - Hermes comparison: {ANALYSIS_DIR / 'hermes_comparison.png'}")
    print(f"  - Hermes delta: {ANALYSIS_DIR / 'hermes_delta.png'}")
    print(f"  - Analysis report: {ANALYSIS_DIR / 'analysis_report.md'}")
    print()

if __name__ == "__main__":
    main()

