# dataset preparation
import pandas as pd
from datasets import Dataset, DatasetDict

# Define file paths
file_gom = 'test.gom_Deva'
file_hin = 'test.hin_Deva'
file_mar = 'test.mar_Deva'
file_eng = 'test.eng_Latn'

# Read the content from each file
with open(file_gom, 'r', encoding='utf-8') as f:
    gom_lines = f.readlines()

with open(file_hin, 'r', encoding='utf-8') as f:
    hin_lines = f.readlines()

with open(file_mar, 'r', encoding='utf-8') as f:
    mar_lines = f.readlines()
with open(file_eng, 'r', encoding='utf-8') as f:
    eng_lines = f.readlines()

# Ensure all files have the same number of lines
min_length = min(len(gom_lines), len(hin_lines), len(mar_lines),len(eng_lines))
gom_lines = gom_lines[:min_length]
hin_lines = hin_lines[:min_length]
mar_lines = mar_lines[:min_length]
eng_lines = eng_lines[:min_length]


# Combine into a DataFrame
df = pd.DataFrame({
    'hin': [line.strip() for line in hin_lines],
    'gom': [line.strip() for line in gom_lines],
    'mar': [line.strip() for line in mar_lines],
    'eng': [line.strip() for line in eng_lines],

})

# Save the DataFrame as a CSV file
csv_file_path = 'merged_eng_hin_mar_gom.csv'
df.to_csv(csv_file_path, index=False)

print(f"Merged CSV file saved to: {csv_file_path}")

# Split the DataFrame into 80% train and 20% test
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

# Convert the splits into Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Combine into a DatasetDict
dataset_dict = DatasetDict({'train': train_dataset, 'test': test_dataset})

# Push the dataset to Hugging Face
dataset_dict.push_to_hub("predictionguard/english-hindi-marathi-konkani-corpus")