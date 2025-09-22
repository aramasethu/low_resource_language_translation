import gc
import torch
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing import List, Tuple
from langchain import PromptTemplate, FewShotPromptTemplate
import predictionguard as pg
from huggingface_hub import notebook_login
from sacrebleu.metrics import BLEU, CHRF, TER
import lancedb
from lancedb.embeddings import with_embeddings
from sentence_transformers import SentenceTransformer
from getpass import getpass
import pdb
import os

from datasets import load_dataset, concatenate_datasets


from transformers import pipeline

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0,
    torch_dtype=torch.float16  # Use FP16 precision
)

# Load data
dataset = load_dataset("predictionguard/english-hindi-marathi-konkani-corpus")
test_df = pd.DataFrame(dataset['test'])
print (test_df.shape)
print(test_df.head())

def translate_prompt_with_ft(english):

  # Pull the most similar examples:
  table = db.open_table("translations_gom")
  results = table.search(embed(english)).limit(10).to_pandas()
  results = results[results['text'] != english]
  results = results[results['text'] != ""]
  # results = results[results['msa'] != ""]
  results = results[results['gom'] != ""]
  results.dropna(inplace=True)
  results.sort_values(by="_distance", ascending=True, inplace=True)

  # Get a random integer between 1 and 3
  messages = []
  num_examples = 5
  for i in range(0, num_examples+1):
    if i != 0 and i <= len(results):
      try:
        messages.append({
            "role": "user",
            "content": USER_PREFIX +  USER_MIDDLE + results['text'].values[i-1]+ USER_SUFFIX
        })
        messages.append({
            "role": "assistant",
            "content": results['gom'].values[i-1]
        })
      except:
        print(results)

  # Add the current context
  messages.append({
      "role": "user",
      "content": USER_PREFIX + USER_MIDDLE + english + USER_SUFFIX
  })

  return tokenizer_mistral.apply_chat_template(messages, tokenize=False)

# Try it out for a sample
prompt = translate_prompt_with_ft(test_df['eng'].values[0])
print(prompt)

# Iterate through each row in the test_df and generate the text
for i, row in test_df.iterrows():
    # Create the prompt for the current row
    prompt = translate_prompt_with_ft(row['eng'])

    # print(f"Prompt for row {i}: {prompt}")

    # Generate text using the pipeline
    try:
        mt = pipeline(
                      prompt,
                      do_sample=True,
                      temperature=0.1,           # Low temperature for focused translation
                      num_return_sequences=1,    # Only one translation per prompt
                      max_new_tokens=200,        # Limit the translation length
                      return_full_text=False,    # Return only the translation output    # Penalize repetition of tokens
   # Prevent repetition of 3-word sequences
                      top_k=50,                  # Limits token choices to the top 50
                      top_p=1,                 # Limits diversity of the output
        # Encourages longer outputs, useful for more detailed translations
                  )
        # Store the generated text in the DataFrame
        test_df.at[i, 'response'] = mt[0]['generated_text']
        print(f"Generated text for row {i}: {mt[0]['generated_text']}")
    except Exception as e:
        print(f"Error in generating text for row {i}: {e}")

# Print the updated DataFrame with the new column
print(test_df.head())

# Save the DataFrame as a CSV file
csv_file_path = 'translated_few-shot-ft-eng-gom-tower.csv'
test_df.to_csv(csv_file_path, index=False)

print(f"Merged CSV file saved to: {csv_file_path}")

import sacrebleu
import json

# Function to compute BLEU score
def calculate_bleu(references, hypotheses):
    # Each reference should be wrapped in a list for sacrebleu
    formatted_references = [[ref] for ref in references]
    bleu = sacrebleu.corpus_bleu(hypotheses, formatted_references)
    return bleu.score

# Function to compute chrF score
def calculate_chrf(references, hypotheses):
    # Each reference should be wrapped in a list for sacrebleu
    formatted_references = [[ref] for ref in references]
    chrf = sacrebleu.corpus_chrf(hypotheses, formatted_references)
    return chrf.score

def calculate_chrf_plus_plus(references, hypotheses):
    # Each reference should be wrapped in a list for sacrebleu
    formatted_references = [[ref] for ref in references]
    chrf_pp = sacrebleu.corpus_chrf(hypotheses, formatted_references, beta=2, word_order=2)
    return chrf_pp.score

# Prepare references and hypotheses
references = test_df['gom'].tolist()
hypotheses = test_df['response'].tolist()

# Calculate BLEU score
bleu_score = calculate_bleu(references, hypotheses)
print(f"Normalized BLEU Score: {bleu_score / 100:.2f}")  # Normalize to 0-1 for consistency with Hugging Face examples
print(f"BLEU Score: {bleu_score}")

# Calculate chrF score
chrf_score = calculate_chrf(references, hypotheses)
print(f"chrF Score: {chrf_score}")

# Calculate chrF++ score
chrf_pp_score = calculate_chrf_plus_plus(references, hypotheses)
print("CHRF++ score:", chrf_pp_score)

# Save the dictionary to a JSON file
with open("scores.json", "w") as f:
    json.dump({
        "BLEU Score": bleu_score,
        "Normalized BLEU Score": bleu_score / 100,
        "chrF Score": chrf_score,
        "CHRF++ Score": chrf_pp_score
    }, f)

