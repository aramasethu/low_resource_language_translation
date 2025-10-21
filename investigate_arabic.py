#!/usr/bin/env python3
"""
Investigation script for Arabic ablation anomalies.
Run in conda env: lrlt_exp
"""
import lancedb
import os
import pandas as pd
import json
from collections import Counter

print("="*80)
print("STEP 1: Verify Vector Database")
print("="*80)

# Check if DB exists
db_path = "arabic_translations"
if not os.path.exists(db_path):
    print(f"❌ ERROR: Database directory '{db_path}' does NOT exist!")
    exit(1)

print(f"✅ Database directory exists: {db_path}")

# Connect and inspect
try:
    db = lancedb.connect(db_path)
    tables = db.table_names()
    print(f"✅ Connected to database")
    print(f"   Tables found: {tables}")
    
    # Check each table
    for table_name in tables:
        table = db.open_table(table_name)
        count = table.count_rows()
        print(f"\n📊 Table: {table_name}")
        print(f"   Total rows: {count}")
        
        if count > 0:
            # Get sample entries
            sample = table.head(5).to_pandas()
            print(f"   Columns: {list(sample.columns)}")
            print(f"\n   Sample entries:")
            for idx, row in sample.iterrows():
                print(f"   Row {idx}:")
                if 'text' in row:
                    text_val = str(row['text'])
                    print(f"     text: {text_val[:80]}..." if len(text_val) > 80 else f"     text: {text_val}")
                if 'msa' in row:
                    msa_val = str(row['msa'])
                    print(f"     msa: {msa_val[:60]}..." if len(msa_val) > 60 else f"     msa: {msa_val}")
                if 'en' in row:
                    en_val = str(row['en'])
                    print(f"     en: {en_val[:60]}..." if len(en_val) > 60 else f"     en: {en_val}")
                if 'tn' in row:
                    tn_val = str(row['tn'])
                    print(f"     tn: {tn_val[:60]}..." if len(tn_val) > 60 else f"     tn: {tn_val}")
                print()
        else:
            print("   ⚠️  WARNING: Table is EMPTY!")
            
except Exception as e:
    print(f"❌ ERROR connecting to database: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*80)
print("STEP 2: Check Test Set Quality")
print("="*80)

try:
    from datasets import load_dataset
    
    dataset = load_dataset("predictionguard/arabic_acl_corpus")
    test_data = dataset['test']
    
    print(f"✅ Dataset loaded")
    print(f"   Test set size: {len(test_data)}")
    print(f"   Features: {test_data.features}")
    
    # Flatten and check
    flattened = []
    for item in test_data:
        translation_dict = item['translation']
        flattened.append({
            'en': translation_dict.get('en', ''),
            'msa': translation_dict.get('msa', ''),
            'tn': translation_dict.get('tn', ''),
        })
    
    test_df = pd.DataFrame(flattened)
    print(f"\n   Flattened test set shape: {test_df.shape}")
    print(f"   Columns: {list(test_df.columns)}")
    
    # Check for empty fields
    empty_counts = {
        'en': (test_df['en'] == '').sum(),
        'msa': (test_df['msa'] == '').sum(),
        'tn': (test_df['tn'] == '').sum()
    }
    print(f"\n   Empty field counts:")
    for field, count in empty_counts.items():
        print(f"     {field}: {count} empty ({count/len(test_df)*100:.1f}%)")
    
    # Check for duplicates
    dup_count = test_df.duplicated().sum()
    print(f"\n   Duplicate rows: {dup_count}")
    
    # Show samples
    print(f"\n   Sample test entries:")
    for idx in range(min(3, len(test_df))):
        row = test_df.iloc[idx]
        print(f"\n   Example {idx}:")
        print(f"     MSA: {row['msa'][:70]}..." if len(row['msa']) > 70 else f"     MSA: {row['msa']}")
        print(f"     EN:  {row['en'][:70]}..." if len(row['en']) > 70 else f"     EN:  {row['en']}")
        print(f"     TN:  {row['tn'][:70]}..." if len(row['tn']) > 70 else f"     TN:  {row['tn']}")
        
except Exception as e:
    print(f"❌ ERROR loading dataset: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("STEP 3: Inspect Generated Outputs")
print("="*80)

# Check k=0 vs k=5 outputs
k_values_to_check = [0, 5, 3]
for k in k_values_to_check:
    result_file = f"ablation_results/arabic_full/k_{k}/results_k{k}.csv"
    if os.path.exists(result_file):
        print(f"\n📄 k={k} results:")
        try:
            df = pd.read_csv(result_file)
            print(f"   File: {result_file}")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            
            if 'response' in df.columns:
                # Check for empty responses
                empty_resp = (df['response'] == '').sum()
                print(f"   Empty responses: {empty_resp} ({empty_resp/len(df)*100:.1f}%)")
                
                # Check for unique responses
                unique_resp = df['response'].nunique()
                print(f"   Unique responses: {unique_resp} out of {len(df)} ({unique_resp/len(df)*100:.1f}%)")
                
                # Show most common responses
                response_counts = Counter(df['response'].tolist())
                print(f"\n   Most common responses:")
                for resp, count in response_counts.most_common(3):
                    resp_preview = resp[:60] + "..." if len(resp) > 60 else resp
                    print(f"     [{count}x] {resp_preview}")
                
                # Show sample
                print(f"\n   Sample translations (first 2):")
                for idx in range(min(2, len(df))):
                    row = df.iloc[idx]
                    print(f"\n   Example {idx}:")
                    if 'en' in df.columns:
                        print(f"     Source (EN): {str(row['en'])[:60]}...")
                    if 'tn' in df.columns:
                        print(f"     Reference (TN): {str(row['tn'])[:60]}...")
                    if 'response' in df.columns:
                        print(f"     Generated: {str(row['response'])[:60]}...")
            else:
                print("   ⚠️  No 'response' column found!")
                
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")
    else:
        print(f"\n⚠️  k={k} results not found: {result_file}")

# Compare k=0 vs k=5 directly
print("\n" + "="*80)
print("STEP 4: Compare k=0 vs k=5 Outputs")
print("="*80)

try:
    df_k0 = pd.read_csv("ablation_results/arabic_full/k_0/results_k0.csv")
    df_k5 = pd.read_csv("ablation_results/arabic_full/k_5/results_k5.csv")
    
    if 'response' in df_k0.columns and 'response' in df_k5.columns:
        # Check if responses are identical
        identical_count = (df_k0['response'] == df_k5['response']).sum()
        print(f"Identical responses between k=0 and k=5: {identical_count}/{len(df_k0)} ({identical_count/len(df_k0)*100:.1f}%)")
        
        if identical_count == len(df_k0):
            print("⚠️  WARNING: ALL responses are IDENTICAL between k=0 and k=5!")
            print("   This explains why scores are the same!")
        
        # Show differences
        diff_indices = (df_k0['response'] != df_k5['response'])
        if diff_indices.sum() > 0:
            print(f"\nShowing first 2 differences:")
            diff_df = df_k0[diff_indices].head(2)
            for idx in diff_df.index[:2]:
                print(f"\n  Example at index {idx}:")
                print(f"    k=0 output: {df_k0.loc[idx, 'response'][:80]}...")
                print(f"    k=5 output: {df_k5.loc[idx, 'response'][:80]}...")
        
except Exception as e:
    print(f"❌ Error comparing k=0 and k=5: {e}")

print("\n" + "="*80)
print("STEP 5: Check Scores Directly")
print("="*80)

# Load all scores
scores_data = []
for k in range(11):
    score_file = f"ablation_results/arabic_full/k_{k}/scores_k{k}.json"
    if os.path.exists(score_file):
        with open(score_file) as f:
            scores = json.load(f)
            scores_data.append({
                'k': k,
                'BLEU': scores.get('BLEU Score', 0),
                'chrF': scores.get('chrF Score', 0),
            })

if scores_data:
    scores_df = pd.DataFrame(scores_data)
    print("\nAll scores:")
    print(scores_df.to_string(index=False))
    
    # Group by score
    print("\n\nGrouped by BLEU score:")
    grouped = scores_df.groupby('BLEU')['k'].apply(list).to_dict()
    for bleu, k_list in sorted(grouped.items()):
        print(f"  BLEU={bleu:.6f}: k={k_list}")

print("\n" + "="*80)
print("INVESTIGATION COMPLETE")
print("="*80)

