#!/usr/bin/env python3
from datasets import load_dataset

# Load and describe datasets
datasets = [
    "predictionguard/english-hindi-marathi-konkani-corpus",
    "predictionguard/arabic_acl_corpus"
]

for dataset_name in datasets:
    print(f"\nDataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    print(f"Splits: {list(dataset.keys())}")
    for split, data in dataset.items():
        print(f"  {split}: {len(data)} rows, columns: {list(data.features.keys())}")
        if len(data) > 0:
            print(f"    Sample: {data[0]}")