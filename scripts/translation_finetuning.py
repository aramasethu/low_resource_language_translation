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
from trl import SFTTrainer
from transformers import DataCollatorForLanguageModeling
import bitsandbytes as bnb
from huggingface_hub import notebook_login
import lancedb
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
from huggingface_hub import notebook_login
from sacrebleu.metrics import BLEU, CHRF, TER
import lancedb
from sentence_transformers import SentenceTransformer
from getpass import getpass
import pdb
import os

from datasets import load_dataset, concatenate_datasets

import argparse

def main():
    parser = argparse.ArgumentParser(description="Zero-shot fine-tuning for translation")
    parser.add_argument("--dataset", default="predictionguard/english-hindi-marathi-konkani-corpus", help="Dataset name")
    parser.add_argument("--model", default="Unbabel/TowerInstruct-v0.1", help="Base model name")
    parser.add_argument("--new-model", default="few-shot-ft-eng-gom-tower", help="New model name to save")
    parser.add_argument("--num-examples", type=int, default=5, help="Number of few-shot examples")
    parser.add_argument("--pivot", default="eng", help="Pivot language column")
    parser.add_argument("--source", default="mar", help="Source language column")
    parser.add_argument("--target", default="gom", help="Target language column")
    parser.add_argument("--db", default="translations_db", help="Database name")
    parser.add_argument("--hf-token", help="HuggingFace token")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Load the dataset
    dataset = load_dataset(args.dataset)
    train_dataset = pd.DataFrame(dataset['train'])
    
    print("Columns in dataset:", train_dataset.columns)
    
    # Ensure the DataFrame contains the necessary columns
    required_columns = [args.pivot, args.target, args.source]
    missing_columns = [col for col in required_columns if col not in train_dataset.columns]
    
    if missing_columns:
        print(f"Missing columns in dataset: {missing_columns}")
        return
    
    print("All required columns are present.")
    train_dataset.reset_index(drop=True, inplace=True)
    
    # Create a new DataFrame with the selected columns
    new_df = train_dataset[[args.pivot, args.target, args.source]]
    
    # Filter out rows with empty strings or NaN values
    mask = (new_df.fillna("").astype(str) == "").any(axis=1)
    new_df = new_df[~mask]
    
    print(f"Original rows: {len(train_dataset)}")
    print(f"New rows after removing empty rows: {len(new_df)}")
    
    # Setup embeddings and database (only for few-shot mode)
    if args.num_examples > 0:
        # Embedding part
        embed_model = SentenceTransformer("all-MiniLM-L12-v2")
        
        def embed(sentence):
            return embed_model.encode(sentence)
        
        def embed_func(batch):
            filtered_batch = [sentence if sentence is not None else "" for sentence in batch]
            return [embed_model.encode(sentence) for sentence in filtered_batch]
        
        # Connect to LanceDB
        db = lancedb.connect(args.db)
        table_name = f"translations_{args.target}"
        
        # Create embeddings manually
        embeddings = embed_func(new_df[args.pivot].tolist())

        # Prepare data for LanceDB
        data = []
        for i, row in new_df.iterrows():
            data.append({
                "text": row[args.pivot],
                "vector": embeddings[i].tolist(),
                args.source: row[args.source],
                args.target: row[args.target]
            })

        # Create or replace table in LanceDB
        tbl = db.create_table(table_name, data, mode='overwrite')
        print(f"Created table '{table_name}' with {len(tbl)} entries")
    else:
        print("Skipping database setup (num_examples = 0)")
    
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Setup prompts
    language_name = args.target.title()
    USER_PREFIX = f'Translate the following source text from {args.pivot.title()} to {language_name}. Only return the {language_name} translation and nothing else.'
    USER_MIDDLE = '\nSource: '
    USER_SUFFIX = '\nTranslation: '
    
    def convert_ex_to_prompt(pivot_text, target_text):
        messages = []
        
        # Add few-shot examples only if num_examples > 0
        if args.num_examples > 0:
            # Pull the most similar examples
            table = db.open_table(f"translations_{args.target}")
            results = table.search(embed(pivot_text)).limit(10).to_pandas()
            results = results[results['text'] != pivot_text]
            results = results[results['text'] != ""]
            results = results[results[args.target] != ""]
            results.dropna(inplace=True)
            results.sort_values(by="_distance", ascending=True, inplace=True)
            
            for i in range(0, args.num_examples + 1):
                if i != 0 and i <= len(results):
                    try:
                        messages.append({
                            "role": "user",
                            "content": USER_PREFIX + USER_MIDDLE + results['text'].values[i-1] + USER_SUFFIX
                        })
                        messages.append({
                            "role": "assistant",
                            "content": results[args.target].values[i-1]
                        })
                    except:
                        print("Error adding example")
        
        # Always add current context
        messages.append({
            "role": "user",
            "content": USER_PREFIX + USER_MIDDLE + pivot_text + USER_SUFFIX
        })
        messages.append({
            "role": "assistant",
            "content": target_text
        })
        
        return tokenizer.apply_chat_template(messages, tokenize=False)
    
    # Prepare training data
    train_texts = []
    for _, row in new_df.iterrows():
        if row[args.pivot] != "" and row[args.target] != "":
            train_texts.append(convert_ex_to_prompt(row[args.pivot], row[args.target]))
    
    print(f"Prepared {len(train_texts)} training examples")
    train_dataset = Dataset.from_dict({"text": train_texts})
    
    # Rest of the training code...
    # (LoRA config, model setup, training, etc.)

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
        args.model,
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
        output_dir=args.new_model,
        optim="paged_adamw_32bit",
        warmup_steps=100,
        bf16=True
    )

    instruction_template = "<|im_start|>user\n"
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = SFTTrainer(
        model,
        train_dataset=train_dataset,
        dataset_text_field="text",
        args=training_args,
        tokenizer=tokenizer,
        peft_config=peft_config,
        data_collator=collator,
        max_seq_length=4096
    )

    trainer.train()

    # Save artifacts
    trainer.model.save_pretrained("final_checkpoint")
    tokenizer.save_pretrained("final_checkpoint")


    # Flush memory
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    # Reload model in FP16 (instead of NF4)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        return_dict=True,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Merge base model with the adapter
    model = PeftModel.from_pretrained(base_model, "final_checkpoint")
    model = model.merge_and_unload()

    # Save model and tokenizer
    model.save_pretrained(args.new_model)
    tokenizer.save_pretrained(args.new_model)

    # # Push them to the HF Hub
    if args.hf_token:
        model.push_to_hub(args.new_model, use_temp_dir=False)
        tokenizer.push_to_hub(args.new_model, use_temp_dir=False)


if __name__ == "__main__":
    main()

