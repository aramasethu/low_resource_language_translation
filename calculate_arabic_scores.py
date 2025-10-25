#!/usr/bin/env python3
"""
Calculate scores from Arabic results CSV files
"""
import pandas as pd
import json
from pathlib import Path
from sacrebleu.metrics import BLEU, CHRF

def calculate_scores_from_csv(csv_file):
    """Calculate BLEU and chrF scores from results CSV"""
    df = pd.read_csv(csv_file)
    
    # Get references and hypotheses
    references = df['tn'].tolist()
    hypotheses = df['response'].tolist()
    
    # Filter out NaN/empty
    valid_pairs = [(h, r) for h, r in zip(hypotheses, references) 
                   if pd.notna(h) and pd.notna(r) and str(h).strip() and str(r).strip()]
    
    if not valid_pairs:
        return None
    
    hypotheses = [str(h) for h, r in valid_pairs]
    references = [[str(r)] for h, r in valid_pairs]
    
    # Calculate scores
    bleu = BLEU()
    chrf = CHRF()
    chrf_pp = CHRF(word_order=2)
    
    bleu_score = bleu.corpus_score(hypotheses, references).score
    chrf_score = chrf.corpus_score(hypotheses, references).score
    chrfpp_score = chrf_pp.corpus_score(hypotheses, references).score
    
    return {
        'BLEU Score': bleu_score,
        'chrF Score': chrf_score,
        'CHRF++ Score': chrfpp_score,
        'samples_processed': len(valid_pairs)
    }

def process_results(base_dir, model_name):
    """Process all k directories in a results folder"""
    print(f"\n{'='*80}")
    print(f"PROCESSING {model_name}")
    print(f"{'='*80}\n")
    
    base_path = Path(base_dir)
    data = []
    
    for k in range(11):
        csv_file = base_path / f"k_{k}" / f"results_k{k}.csv"
        print(f"Processing k={k}...", end=' ')
        
        if csv_file.exists():
            scores = calculate_scores_from_csv(csv_file)
            if scores:
                # Save scores JSON
                score_file = base_path / f"k_{k}" / f"scores_k{k}.json"
                with open(score_file, 'w') as f:
                    json.dump({**scores, 'k': k}, f, indent=2)
                
                data.append({
                    'k': k,
                    'BLEU': scores['BLEU Score'],
                    'chrF': scores['chrF Score'],
                    'chrF++': scores['CHRF++ Score']
                })
                print(f"✅ BLEU={scores['BLEU Score']:.4f}")
            else:
                print("❌ No valid data")
        else:
            print(f"❌ File not found")
    
    # Create summary DataFrame
    df = pd.DataFrame(data).sort_values('k')
    
    # Save summary
    summary_file = base_path / "summary.csv"
    df.to_csv(summary_file, index=False)
    print(f"\n✅ Summary saved: {summary_file}")
    
    return df

# Process Tower model
tower_df = process_results("ablation_results/arabic_600tokens", "TOWER MODEL")

# Process Hermes model
hermes_df = process_results("ablation_results/arabic_hermes_600tokens", "HERMES MODEL")

# Analysis
print("\n" + "="*80)
print("COMPARATIVE ANALYSIS - ARABIC")
print("="*80)

print("\n" + "-"*80)
print("TOWER MODEL (TowerInstruct-7B-v0.1)")
print("-"*80)
print(tower_df.to_string(index=False))
print(f"\nBaseline (k=0): BLEU={tower_df[tower_df['k']==0]['BLEU'].values[0]:.4f}")
best_tower = tower_df.loc[tower_df['BLEU'].idxmax()]
print(f"Best: k={int(best_tower['k'])}, BLEU={best_tower['BLEU']:.4f}, chrF={best_tower['chrF']:.2f}")
improvement_tower = (best_tower['BLEU'] - tower_df[tower_df['k']==0]['BLEU'].values[0]) / tower_df[tower_df['k']==0]['BLEU'].values[0] * 100
print(f"Improvement: +{improvement_tower:.1f}%")

print("\n" + "-"*80)
print("HERMES MODEL (Hermes-2-Pro-Llama-3-8B)")
print("-"*80)
print(hermes_df.to_string(index=False))
print(f"\nBaseline (k=0): BLEU={hermes_df[hermes_df['k']==0]['BLEU'].values[0]:.4f}")
best_hermes = hermes_df.loc[hermes_df['BLEU'].idxmax()]
print(f"Best: k={int(best_hermes['k'])}, BLEU={best_hermes['BLEU']:.4f}, chrF={best_hermes['chrF']:.2f}")
improvement_hermes = (best_hermes['BLEU'] - hermes_df[hermes_df['k']==0]['BLEU'].values[0]) / hermes_df[hermes_df['k']==0]['BLEU'].values[0] * 100
print(f"Improvement: +{improvement_hermes:.1f}%")

print("\n" + "-"*80)
print("MODEL COMPARISON")
print("-"*80)
print(f"Zero-shot (k=0):")
print(f"  Tower:  {tower_df[tower_df['k']==0]['BLEU'].values[0]:.4f}")
print(f"  Hermes: {hermes_df[hermes_df['k']==0]['BLEU'].values[0]:.4f}")
print(f"  Winner: {'Tower' if tower_df[tower_df['k']==0]['BLEU'].values[0] > hermes_df[hermes_df['k']==0]['BLEU'].values[0] else 'Hermes'}")

print(f"\nBest performance:")
print(f"  Tower:  k={int(best_tower['k'])}, BLEU={best_tower['BLEU']:.4f}")
print(f"  Hermes: k={int(best_hermes['k'])}, BLEU={best_hermes['BLEU']:.4f}")
print(f"  Winner: {'Tower' if best_tower['BLEU'] > best_hermes['BLEU'] else 'Hermes'}")

print(f"\nFew-shot learning benefit:")
print(f"  Tower:  +{improvement_tower:.1f}%")
print(f"  Hermes: +{improvement_hermes:.1f}%")
print(f"  Better learner: {'Tower' if improvement_tower > improvement_hermes else 'Hermes'}")

print("\n" + "="*80)

