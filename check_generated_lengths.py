#!/usr/bin/env python3
"""
Check token lengths in generated results files.
"""
from transformers import AutoTokenizer
import pandas as pd
import os

print("="*80)
print("CHECKING GENERATED TRANSLATION LENGTHS")
print("="*80)

# Load tokenizer
model = "Unbabel/TowerInstruct-7B-v0.1"
print(f"\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model)
print("✅ Tokenizer loaded")

# Check Arabic results
print("\n" + "="*80)
print("ARABIC GENERATED TRANSLATIONS")
print("="*80)

arabic_files = [
    ("k=0", "ablation_results/arabic_full/k_0/results_k0.csv"),
    ("k=5", "ablation_results/arabic_full/k_5/results_k5.csv"),
    ("k=10", "ablation_results/arabic_full/k_10/results_k10.csv"),
]

for label, filepath in arabic_files:
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        
        # Get token lengths for generated responses
        response_lengths = []
        for resp in df['response']:
            if pd.notna(resp):
                tokens = tokenizer.encode(str(resp), add_special_tokens=False)
                response_lengths.append(len(tokens))
            else:
                response_lengths.append(0)
        
        # Get token lengths for references
        ref_lengths = []
        for ref in df['tn']:
            if pd.notna(ref):
                tokens = tokenizer.encode(str(ref), add_special_tokens=False)
                ref_lengths.append(len(tokens))
            else:
                ref_lengths.append(0)
        
        print(f"\n{label}:")
        print(f"  Generated: mean={sum(response_lengths)/len(response_lengths):.1f}, max={max(response_lengths)}, min={min(response_lengths)}")
        print(f"  Reference: mean={sum(ref_lengths)/len(ref_lengths):.1f}, max={max(ref_lengths)}, min={min(ref_lengths)}")
        
        # Check if any are close to 200
        near_200 = [x for x in response_lengths if x >= 190]
        if near_200:
            print(f"  ⚠️  {len(near_200)} generations are >=190 tokens (possibly truncated)")

# Check Konkani results if available
print("\n" + "="*80)
print("KONKANI GENERATED TRANSLATIONS")
print("="*80)

konkani_files = [
    ("k=0", "ablation_results/konkani_full/k_0/results_k0.csv"),
    ("k=3", "ablation_results/konkani_full/k_3/results_k3.csv"),
    ("k=5", "ablation_results/konkani_full/k_5/results_k5.csv"),
    ("k=10", "ablation_results/konkani_full/k_10/results_k10.csv"),
]

for label, filepath in konkani_files:
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        
        # Get token lengths for generated responses
        response_lengths = []
        for resp in df['response']:
            if pd.notna(resp):
                tokens = tokenizer.encode(str(resp), add_special_tokens=False)
                response_lengths.append(len(tokens))
            else:
                response_lengths.append(0)
        
        # Get token lengths for references
        ref_lengths = []
        target_col = 'gom' if 'gom' in df.columns else 'kok'
        for ref in df[target_col]:
            if pd.notna(ref):
                tokens = tokenizer.encode(str(ref), add_special_tokens=False)
                ref_lengths.append(len(tokens))
            else:
                ref_lengths.append(0)
        
        print(f"\n{label}:")
        print(f"  Generated: mean={sum(response_lengths)/len(response_lengths):.1f}, max={max(response_lengths)}, min={min(response_lengths)}")
        print(f"  Reference: mean={sum(ref_lengths)/len(ref_lengths):.1f}, max={max(ref_lengths)}, min={min(ref_lengths)}")
        
        # Check if any are close to 200
        near_200 = [x for x in response_lengths if x >= 190]
        if near_200:
            print(f"  ⚠️  {len(near_200)} generations are >=190 tokens (possibly truncated)")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("""
If generated translations are consistently shorter than references,
or if many are near 200 tokens, then max_new_tokens=200 may be limiting quality.

If generated and reference lengths are similar, then max_new_tokens=200 is fine.
""")

