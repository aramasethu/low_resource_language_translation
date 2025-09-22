# low_resource_language_translation
Experiments for low resource language translation using few shot semantically similar examples. This is for the scenario where we have very little data for translation and the models available have been pretrained on the language. In such constrained scenarios we are trying to assess if this prompting technique can lead to better quality translation during inferece and while finetuning. 

# Models
1. Unbabel/TowerInstruct-Mistral-7B-v0.2
2. NousResearch/Hermes-2-Pro-Llama-3-8B

# Languages 
1. Konkani
2. Tunisian Arabic

# Parallel Corpus 
You can find the datasets for running th experiments on Hugging face: 

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