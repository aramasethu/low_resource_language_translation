from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("predictionguard/english-hindi-marathi-konkani-corpus")
# Basic preview of the dataset
try:
    # Handle both DatasetDict (multiple splits) and single Dataset
    if hasattr(ds, "keys"):  # DatasetDict
        print("Splits:", list(ds.keys()))
        first_split = next(iter(ds.keys()))
        print("Features:", ds[first_split].features)
        print("First example from", first_split, ":", ds[first_split][0])
    else:  # Single Dataset
        print("Features:", ds.features)
        print("First example:", ds[0])
except Exception as e:
    print("Error previewing dataset:", e)

# -------------------------
# Simple EDA for the dataset
# -------------------------
try:
    import statistics
    import pandas as pd  # pandas is commonly available with datasets/pyarrow

    def get_splits(dataset):
        if hasattr(dataset, "keys"):
            return dataset  # DatasetDict
        return {"train": dataset}

    splits = get_splits(ds)
    print("\n=== EDA Summary ===")
    for split_name, split_ds in splits.items():
        print(f"\n-- Split: {split_name} --")
        try:
            num_rows = len(split_ds)
        except Exception:
            num_rows = split_ds.num_rows if hasattr(split_ds, "num_rows") else "unknown"
        print("Rows:", num_rows)
        print("Features:", split_ds.features)

        # Convert to DataFrame for quick stats
        try:
            df = split_ds.to_pandas()
        except Exception as e_to_pd:
            print("to_pandas failed:", e_to_pd)
            df = None

        if df is not None:
            # Null counts
            try:
                null_counts = df.isna().sum()
                print("\nNull counts (top 20):")
                print(null_counts.head(20))
            except Exception as e_null:
                print("Null count calc failed:", e_null)

            # Basic describe
            try:
                desc = df.describe(include="all").transpose()
                print("\nDescribe (top 10 rows):")
                print(desc.head(10))
            except Exception as e_desc:
                print("Describe failed:", e_desc)

            # Text length stats for object columns
            try:
                object_cols = [c for c in df.columns if df[c].dtype == "object"]
                for col in object_cols:
                    series = df[col].dropna()
                    # Only compute lengths for strings
                    series_str = series[series.apply(lambda v: isinstance(v, str))]
                    lengths = series_str.apply(len)
                    if len(lengths) == 0:
                        continue
                    print(f"\nText length stats for '{col}':")
                    print("count:", len(lengths))
                    print("min:", int(lengths.min()))
                    print("max:", int(lengths.max()))
                    print("mean:", round(float(lengths.mean()), 2))
                    try:
                        print("median:", statistics.median(lengths))
                    except Exception:
                        pass
            except Exception as e_txt:
                print("Text length stats failed:", e_txt)

            # Samples
            try:
                n = min(3, len(df))
                if n > 0:
                    samples = df.sample(n=n, random_state=42).to_dict(orient="records")
                    print("\nSamples:")
                    for i, ex in enumerate(samples, 1):
                        print(f"Sample {i}:", ex)
            except Exception as e_samp:
                print("Sampling failed:", e_samp)
except Exception as e:
    print("EDA failed:", e)

# -------------------------
# Advanced EDA: Alignment, Unicode, Duplicates, Leakage
# -------------------------
try:
    import re
    from collections import Counter, defaultdict
    
    print("\n=== Advanced EDA ===")
    
    def analyze_unicode_coverage(texts, lang_name):
        """Analyze Unicode character coverage"""
        all_chars = set()
        for text in texts:
            all_chars.update(text)
        
        # Categorize characters
        devanagari = sum(1 for c in all_chars if '\u0900' <= c <= '\u097F')
        latin = sum(1 for c in all_chars if c.isascii() and c.isalpha())
        digits = sum(1 for c in all_chars if c.isdigit())
        punctuation = sum(1 for c in all_chars if not c.isalnum() and not c.isspace())
        spaces = sum(1 for c in all_chars if c.isspace())
        other = len(all_chars) - devanagari - latin - digits - punctuation - spaces
        
        print(f"\n{lang_name} Unicode Coverage:")
        print(f"  Total unique chars: {len(all_chars)}")
        print(f"  Devanagari: {devanagari}")
        print(f"  Latin: {latin}")
        print(f"  Digits: {digits}")
        print(f"  Punctuation: {punctuation}")
        print(f"  Spaces: {spaces}")
        print(f"  Other: {other}")
        
        return all_chars
    
    def check_alignment_quality(split_ds, lang_cols):
        """Check if translations are properly aligned"""
        print(f"\nAlignment Quality Check:")
        
        # Check for empty or very short texts
        for col in lang_cols:
            texts = [row[col] for row in split_ds]
            empty_count = sum(1 for t in texts if not t or len(t.strip()) < 5)
            print(f"  {col}: {empty_count} empty/very short texts")
        
        # Check length correlations
        if len(lang_cols) >= 2:
            import statistics
            for i, col1 in enumerate(lang_cols):
                for col2 in lang_cols[i+1:]:
                    lengths1 = [len(row[col1]) for row in split_ds]
                    lengths2 = [len(row[col2]) for row in split_ds]
                    
                    # Simple correlation coefficient
                    n = len(lengths1)
                    if n > 1:
                        mean1, mean2 = statistics.mean(lengths1), statistics.mean(lengths2)
                        numerator = sum((l1-mean1)*(l2-mean2) for l1, l2 in zip(lengths1, lengths2))
                        denom1 = sum((l-mean1)**2 for l in lengths1)
                        denom2 = sum((l-mean2)**2 for l in lengths2)
                        
                        if denom1 > 0 and denom2 > 0:
                            corr = numerator / (denom1 * denom2) ** 0.5
                            print(f"  Length correlation {col1}-{col2}: {corr:.3f}")
    
    def find_duplicates(split_ds, lang_cols):
        """Find potential duplicate entries"""
        print(f"\nDuplicate Detection:")
        
        for col in lang_cols:
            texts = [row[col] for row in split_ds]
            text_counts = Counter(texts)
            duplicates = {text: count for text, count in text_counts.items() if count > 1}
            
            if duplicates:
                print(f"  {col}: {len(duplicates)} duplicate texts found")
                # Show top 3 duplicates
                top_dups = sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[:3]
                for text, count in top_dups:
                    preview = text[:50] + "..." if len(text) > 50 else text
                    print(f"    '{preview}' appears {count} times")
            else:
                print(f"  {col}: No exact duplicates found")
    
    def check_train_test_leakage(train_ds, test_ds, lang_cols):
        """Check for data leakage between train and test"""
        print(f"\nTrain-Test Leakage Check:")
        
        # Convert to sets for faster lookup
        train_texts = {}
        for col in lang_cols:
            train_texts[col] = set(row[col] for row in train_ds)
        
        for col in lang_cols:
            test_texts = [row[col] for row in test_ds]
            leaked = [text for text in test_texts if text in train_texts[col]]
            
            if leaked:
                print(f"  {col}: {len(leaked)} texts leaked from train to test")
                # Show examples
                for i, text in enumerate(leaked[:2]):
                    preview = text[:50] + "..." if len(text) > 50 else text
                    print(f"    Leaked example {i+1}: '{preview}'")
            else:
                print(f"  {col}: No leakage detected")
    
    def analyze_language_patterns(split_ds, lang_cols):
        """Analyze language-specific patterns"""
        print(f"\nLanguage Pattern Analysis:")
        
        for col in lang_cols:
            texts = [row[col] for row in split_ds]
            
            # Word count (rough estimate)
            word_counts = []
            for text in texts:
                # Simple word splitting
                words = re.findall(r'\S+', text)
                word_counts.append(len(words))
            
            if word_counts:
                print(f"  {col} word counts - min: {min(word_counts)}, max: {max(word_counts)}, mean: {statistics.mean(word_counts):.1f}")
            
            # Check for common patterns
            if col == 'eng':
                # English-specific patterns
                has_quotes = sum(1 for text in texts if '"' in text or "'" in text)
                has_numbers = sum(1 for text in texts if re.search(r'\d', text))
                print(f"  {col}: {has_quotes} texts with quotes, {has_numbers} texts with numbers")
            
            elif col in ['hin', 'mar', 'gom']:
                # Devanagari-specific patterns
                has_devanagari = sum(1 for text in texts if re.search(r'[\u0900-\u097F]', text))
                has_english = sum(1 for text in texts if re.search(r'[a-zA-Z]', text))
                print(f"  {col}: {has_devanagari} texts with Devanagari, {has_english} texts with English chars")
    
    # Run advanced analysis for each split
    splits = get_splits(ds)
    lang_cols = ['hin', 'gom', 'mar', 'eng']
    
    for split_name, split_ds in splits.items():
        print(f"\n--- Advanced Analysis: {split_name} ---")
        
        # Unicode coverage
        for col in lang_cols:
            texts = [row[col] for row in split_ds]
            analyze_unicode_coverage(texts, f"{split_name}_{col}")
        
        # Alignment quality
        check_alignment_quality(split_ds, lang_cols)
        
        # Duplicates
        find_duplicates(split_ds, lang_cols)
        
        # Language patterns
        analyze_language_patterns(split_ds, lang_cols)
    
    # Cross-split analysis
    if len(splits) > 1:
        print(f"\n--- Cross-Split Analysis ---")
        train_ds = splits.get('train')
        test_ds = splits.get('test')
        
        if train_ds and test_ds:
            check_train_test_leakage(train_ds, test_ds, lang_cols)
    
    print(f"\n=== Advanced EDA Complete ===")

except Exception as e:
    print("Advanced EDA failed:", e)
    import traceback
    traceback.print_exc()