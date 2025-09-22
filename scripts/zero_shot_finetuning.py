"""# Import data"""

import os
import gc
import torch
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing import List, Tuple

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset, Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import bitsandbytes as bnb
from huggingface_hub import notebook_login
import lancedb
from lancedb.embeddings import with_embeddings
from sentence_transformers import SentenceTransformer


# model_name = "NousResearch/Hermes-2-Pro-Llama-3-8B"
# new_model = "hermesllama-ft-hin-mar-gom"

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


# Load the dataset
dataset = load_dataset("predictionguard/english-hindi-marathi-konkani-corpus")

# Convert the train split to a DataFrame
train_dataset = pd.DataFrame(dataset['train'])

# Print column names to check structure
print("Columns in dataset:", train_dataset.columns)

# Ensure the DataFrame contains the necessary columns
required_columns = ['hin', 'gom', 'mar','eng']
missing_columns = [col for col in required_columns if col not in train_dataset.columns]

if missing_columns:
    print(f"Missing columns in dataset: {missing_columns}")
else:
    print("All required columns are present.")

# Reset index if needed
train_dataset.reset_index(drop=True, inplace=True)

# Define the columns for your dialects
dialect1_ = "mar"
dialect2_ = "gom"
# dialect3_ = "eng"

# Create a new DataFrame with the selected columns
new_df = train_dataset[['eng', dialect2_, dialect1_]]

# Filter out rows with empty strings or NaN values
mask = (new_df[['eng', dialect2_, dialect1_]].fillna("").astype(str) == "").any(axis=1)
new_df = new_df[~mask]

# Check the number of rows in the original DataFrame and the new DataFrame
original_rows = len(train_dataset)
new_rows = len(new_df)
print("Original rows:", original_rows)
print("New rows after removing empty rows:", new_rows)

# Embedding part
name = "all-MiniLM-L12-v2"
embed_model = SentenceTransformer(name)

def embed(sentence):
    return embed_model.encode(sentence)

def embed_func(batch):
    filtered_batch = [sentence if sentence is not None else "" for sentence in batch]
    return [embed_model.encode(sentence) for sentence in filtered_batch]

# Connect to LanceDB
db = lancedb.connect('translations_db')  # Use a descriptive DB name
table_name = "translations_" + dialect2_

# Prepare data with embeddings
data = with_embeddings(embed_func, new_df.rename(columns={"eng": "text"}))

# Create or replace table in LanceDB
tbl = db.create_table(table_name, data, mode='overwrite')

# Display the table data
print(tbl.to_pandas())

tbl.to_pandas()

"""# Load vectorized demonstrations

# Few shot setup
"""

tokenizer_mistral = AutoTokenizer.from_pretrained("Unbabel/TowerInstruct-v0.1")

chat = [
   {"role": "user", "content": "Hello, how are you?"},
   {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
   {"role": "user", "content": "I'd like to show off how chat templating works!"},
]

tokenizer_mistral.apply_chat_template(chat, tokenize=False)
#tokenizer_mistral.pad_token = tokenizer_mistral.eos_token
#tokenizer_mistral.padding_side = "left"

# user prefix for each message (w/out pivot)
language_name = "Konkani"
USER_PREFIX = 'Translate the following source text from English to Konkani. Only return the Konkani translation and nothing else.'
USER_MIDDLE = '\nSource: '
USER_SUFFIX = '\nTranslation: '

# # user prefix for each message ( PIVOT )
# language_name = "Modern Standard Arabic"
# USER_PREFIX = 'APE is a task designed to enhance the quality of the translation by performing only minor adjustments to fix any existing translation mistakes. If the translation is already correct, you should retain it as is.\nOriginal (English): '
# USER_MIDDLE = '\nTranslation (' + language_name + '): '
# USER_SUFFIX = '\nPost-edited (Tunisian): '

print(db.table_names())

"""# Training specific code"""

def convert_ex_to_prompt(english,target):

  # Pull the most similar examples:
  table = db.open_table('translations_gom')
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
            "content": USER_PREFIX +  USER_MIDDLE + results['text'].values[i-1] + USER_SUFFIX
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
  messages.append({
      "role": "assistant",
      "content": target
  })

  return tokenizer_mistral.apply_chat_template(messages, tokenize=False)

print(convert_ex_to_prompt(train_dataset['eng'].values[0], train_dataset['gom'].values[0]))

train_texts = []

for _, row in train_dataset.iterrows():
  if row['eng'] != "" and row['gom'] != "":
    train_texts.append(convert_ex_to_prompt(row['eng'], row['gom']))

len(train_texts)

train_texts[0]

train_dataset = Dataset.from_dict({"text": train_texts})

"""## Train model with SFT"""

model_name = "Unbabel/TowerInstruct-v0.1"
new_model = "few-shot-ft-eng-gom-tower"

# LoRA configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
)

# Model to fine-tune
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    load_in_4bit=True
)
model.config.use_cache = False

# Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    max_steps=400,
    save_strategy="no",
    logging_steps=10,
    output_dir=new_model,
    optim="paged_adamw_32bit",
    warmup_steps=100,
    bf16=True
)

instruction_template = "<|im_start|>user\n"
response_template = "<|im_start|>assistant\n"
collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template,
                                           response_template=response_template,
                                           tokenizer=tokenizer_mistral,
                                           mlm=False)

trainer = SFTTrainer(
    model,
    train_dataset=train_dataset,
    dataset_text_field="text",
    args=training_args,
    tokenizer=tokenizer_mistral,
    peft_config=peft_config,
    data_collator=collator,
    max_seq_length=4096
)

trainer.train()

"""## Upload model"""

# Save artifacts
trainer.model.save_pretrained("final_checkpoint")
tokenizer_mistral.save_pretrained("final_checkpoint")


# Flush memory
del trainer, model
gc.collect()
torch.cuda.empty_cache()

# Reload model in FP16 (instead of NF4)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    return_dict=True,
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Merge base model with the adapter
model = PeftModel.from_pretrained(base_model, "final_checkpoint")
model = model.merge_and_unload()

# Save model and tokenizer
model.save_pretrained(new_model)
tokenizer.save_pretrained(new_model)

# # Push them to the HF Hub
# model.push_to_hub("predictionguard/zero-shot-ft-en-msa-tn-llama", use_temp_dir=False)
# tokenizer.push_to_hub("predictionguard/zero-shot-ft-en-msa-tn-llama", use_temp_dir=False)

