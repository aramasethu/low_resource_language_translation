


# low_resource_language_translation
Experiments for low resource language translation using few shot semantically similar examples. This is for the scenario where we have very little data for translation and the models available have been pretrained on the language. In such constrained scenarios we are trying to assess if this prompting technique can lead to better quality translation during inferece and while finetuning. 

# Models
1. Unbabel/TowerInstruct-Mistral-7B-v0.2
2. NousResearch/Hermes-2-Pro-Llama-3-8B

# Languages 
1. Konkani
2. Tunisian Arabic

# Parallel Corpus 
You can find the datasets for running the experiments on Hugging face. It has been made public and can be loaded using the transformers library 

1. Konkani dataset: predictionguard/english-hindi-marathi-konkani-corpus
2. Tunisian Arabic Dataset: predictionguard/arabic_acl_corpus

# Pivot language
For running the experiment we make use of pivot language: this is a language that is similar to the language the low resource target language, that the model is familiar with. We use this language to guide the model to translate to a low resource language it is not exposed to. 

1. For konkani: english (source) -> marathi (pivot) -> konkani (target)
2. For tunisian arabic: english (source) -> modern standard arabic (pivot) -> tunisian arabic (target)

# Prompt template:
Here is the prompt template being used in the code. Refer to the appendix for examples for the prompt template:
```
<|im_start|>user
APE is a task designed to enhance
the quality of the translation
by performing minor adjustments
Original (English): [Original text]
Translation: [Pivot language]
Post-edited:
<|im_end|>
<|im_start|>assistant
[LLM translation]
<|im_end|>
```

# Instructions to run the code:
## Scripts:
1. `konkani_corpus_preparation.py`
This code need not be run again. It is there to illustrate how the corpus was constructed. If you see any obvious red flags please raise an issue. The Konkani parallel corpus was constructed using a dataset open-sourced by AI4Bharat, which also contributed to the training set for the IndicTrans2 model (https://openreview.net/forum?id=vfT4YuzAYA). This corpus includes English, Marathi, and Konkani.
Similarly, for Tunisian Arabic, the corpus was derived from the work described here: https://aclanthology.org/L14-1435/ , with Modern Standard Arabic chosen as the pivot language. The parallel corpus for Tunisian Arabic contained 1,000 records, with 900 used in the training set and 100 used in the test set. 

2. `describe_datasets.py`
This code is a simple implementation of using the datasets library to do very preliminary exploration of the dataset. Run this code to get the basic information of the two parallel corpus. Like the train and test composnents and columns and what the `translation` column in the dataset contains.

3. `create_vector_db.py`
For construcitng the few shot prompts with the semantically similar examples, we will need to create a vector DB. This script helps in creating the vector DB, this can then be used later in during the inference. 

example on how to run: 
```
python scripts/create_vector_db.py \
    --dataset "predictionguard/english-hindi-marathi-konkani-corpus" \
    --pivot "mar" \
    --source "eng" \
    --target "gom"
```
Replace the dataset name and the source pivot and target. 

4. `run_inference.py`
This script is to perform inference with either the finetuned model that is uploaded to huggingface or a new model to calculate baselines. Before running this script make sure that you have created the vector DB that is necessary to fetch the semantically similar examples to construct the prompt. You can run this script using this command and changing the model and dataset:

```
# python scripts/run_inference.py \
    --dataset "predictionguard/english-hindi-marathi-konkani-corpus" \
    --model "Unbabel/TowerInstruct-7B-v0.1" \
    --pivot "hin" \
    --source "mar" \
    --target "gom" \
    --db "konkani_translations" \
    --output "konkani_results.csv" \
    --scores "konkani_scores.json"
```
As you run the inference please do commit the scores to the repository so it can be tracked for paper writing. 

5. `translation_finetune.py`
This script is used for fineutning. You can run it with the following command:

```
python scripts/translation_finetuning.py \
    --dataset "predictionguard/english-hindi-marathi-konkani-corpus" \
    --model "Unbabel/TowerInstruct-v0.1" \
    --new-model "few-shot-eng-mar-gom-tower" \
    --num-examples 5 \
    --pivot "eng" \
    --source "mar" \
    --target "gom" \
    --db "konkani_translations" \
    --hf-token "hf_your_token_here" \
    --seed 42
```

6. `generate_ablation_table.py`
This script is used to generate ablation tables from the results of the inferences. It automatically gets the scores from the json files available in the directory. You can run it with the following command:

```
python scripts/generate_ablation_table.py \
    --format markdown \
    --output ablation_results.md
```