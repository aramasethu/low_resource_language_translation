from datasets import load_dataset

# Load dataset (login via `huggingface-cli login` if access required)
ds = load_dataset("predictionguard/arabic_acl_corpus")

print("=== Basic Info ===")
try:
    if hasattr(ds, "keys"):
        print("Splits:", list(ds.keys()))
        first_split = next(iter(ds.keys()))
        print("Features:", ds[first_split].features)
        try:
            print("First example:", ds[first_split][0])
        except Exception as e:
            print("First example fetch failed:", e)
    else:
        print("Features:", ds.features)
        print("First example:", ds[0])
except Exception as e:
    print("Preview failed:", e)

print("\n=== EDA Summary ===")

import statistics
import re

# Helpers

def get_splits(dataset):
    if hasattr(dataset, "keys"):
        return dataset
    return {"train": dataset}

ARABIC_RANGES = [
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
]

def is_arabic_char(ch: str) -> bool:
    code = ord(ch)
    for start, end in ARABIC_RANGES:
        if start <= code <= end:
            return True
    return False


def analyze_unicode_coverage(texts, label: str):
    unique_chars = set()
    for t in texts:
        if isinstance(t, str):
            unique_chars.update(t)
    arabic = sum(1 for c in unique_chars if is_arabic_char(c))
    latin = sum(1 for c in unique_chars if c.isascii() and c.isalpha())
    digits = sum(1 for c in unique_chars if c.isdigit())
    punctuation = sum(1 for c in unique_chars if not c.isalnum() and not c.isspace())
    spaces = sum(1 for c in unique_chars if c.isspace())
    other = len(unique_chars) - arabic - latin - digits - punctuation - spaces

    print(f"\nUnicode Coverage [{label}]:")
    print("  Total unique chars:", len(unique_chars))
    print("  Arabic:", arabic)
    print("  Latin:", latin)
    print("  Digits:", digits)
    print("  Punctuation:", punctuation)
    print("  Spaces:", spaces)
    print("  Other:", other)


def try_to_pandas(hf_ds):
    try:
        return hf_ds.to_pandas()
    except Exception as e:
        print("to_pandas failed:", e)
        return None


def infer_text_columns(df):
    if df is None:
        return []
    return [c for c in df.columns if df[c].dtype == "object"]


def length_stats(series):
    s = series.dropna()
    s = s[s.apply(lambda v: isinstance(v, str))]
    lengths = s.apply(len)
    if len(lengths) == 0:
        return None
    return {
        "count": int(len(lengths)),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "median": float(statistics.median(lengths)),
    }


def duplicates_report(texts):
    from collections import Counter
    counts = Counter(texts)
    dups = {t: c for t, c in counts.items() if c > 1}
    return dups


splits = get_splits(ds)
for split_name, split_ds in splits.items():
    print(f"\n-- Split: {split_name} --")
    try:
        print("Rows:", len(split_ds))
    except Exception:
        print("Rows:", getattr(split_ds, "num_rows", "unknown"))
    print("Features:", split_ds.features)

    df = try_to_pandas(split_ds)
    if df is not None:
        # Flatten translation dict into separate language columns if present
        try:
            if "translation" in df.columns:
                feat = split_ds.features
                trans_keys = []
                try:
                    trans_keys = list(feat["translation"].keys())  # expected: ['eg','en',...]
                except Exception:
                    # Fallback: sample one row to infer keys
                    sample_val = df["translation"].dropna().iloc[0]
                    if isinstance(sample_val, dict):
                        trans_keys = list(sample_val.keys())
                for k in trans_keys:
                    col_name = f"translation_{k}"
                    if col_name not in df.columns:
                        df[col_name] = df["translation"].apply(lambda d: d.get(k) if isinstance(d, dict) else None)
        except Exception as e:
            print("Flatten translation failed:", e)

        # Nulls
        try:
            nulls = df.isna().sum()
            print("\nNull counts (top 20):")
            print(nulls.head(20))
        except Exception as e:
            print("Null calc failed:", e)

        # Describe
        try:
            desc = df.describe(include="all").transpose()
            print("\nDescribe (top 10 rows):")
            print(desc.head(10))
        except Exception as e:
            print("Describe failed:", e)

        # Text columns and stats (prefer flattened translation_* columns)
        flattened_cols = [c for c in df.columns if c.startswith("translation_")]
        text_cols = flattened_cols if flattened_cols else infer_text_columns(df)
        # Exclude non-text like raw 'translation' dict and list 'id'
        text_cols = [c for c in text_cols if c not in ["translation", "id"]]
        for col in text_cols:
            stats = length_stats(df[col])
            if stats:
                print(f"\nText length stats for '{col}':")
                for k, v in stats.items():
                    print(f"  {k}: {v}")

        # Unicode coverage for each text column (use DataFrame values)
        for col in text_cols:
            try:
                texts = [v for v in df[col].dropna().tolist() if isinstance(v, str)]
            except Exception:
                texts = []
            analyze_unicode_coverage(texts, f"{split_name}.{col}")

        # Duplicates per text column
        print("\nDuplicate Detection:")
        for col in text_cols:
            try:
                texts = [v for v in df[col].dropna().tolist() if isinstance(v, str)]
            except Exception:
                texts = []
            dups = duplicates_report(texts) if texts else {}
            if dups:
                print(f"  {col}: {len(dups)} duplicate texts (showing up to 3):")
                for i, (t, c) in enumerate(sorted(dups.items(), key=lambda x: x[1], reverse=True)[:3], 1):
                    preview = t[:80] + "..." if isinstance(t, str) and len(t) > 80 else t
                    print(f"    #{i}: occurs {c} times -> {preview}")
            else:
                print(f"  {col}: No exact duplicates found")

# Cross-split leakage check (if train/test exist)
if isinstance(splits, dict) and "train" in splits and "test" in splits:
    print("\n-- Cross-Split Leakage Check --")
    train = splits["train"]
    test = splits["test"]

    # Try to infer a primary text column (first object column)
    train_df = try_to_pandas(train)
    test_df = try_to_pandas(test)
    if train_df is not None and test_df is not None:
        text_cols = infer_text_columns(train_df)
        for col in text_cols:
            train_set = set(x for x in train_df[col].dropna().tolist() if isinstance(x, str))
            test_vals = [x for x in test_df[col].dropna().tolist() if isinstance(x, str)]
            leaked = [x for x in test_vals if x in train_set]
            if leaked:
                print(f"  Leakage on column '{col}': {len(leaked)} overlapped texts (showing up to 2):")
                for i, x in enumerate(leaked[:2], 1):
                    preview = x[:80] + "..." if len(x) > 80 else x
                    print(f"    Example {i}: {preview}")
            else:
                print(f"  No leakage on column '{col}'")

print("\n=== EDA Complete ===")
