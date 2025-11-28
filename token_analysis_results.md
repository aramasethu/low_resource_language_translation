# Token Analysis Report

## Overview
Analysis of tokenization behavior for Tower (Mistral-based) and Hermes (Llama-based) models
across languages in the parallel corpora.

## Token Fertility (Tokens per Word)

Lower values indicate more efficient tokenization for that language.

| Dataset | Language | Tower | Hermes |
|---------|----------|-------|--------|
| Konkani | English | 1.59 | 1.34 |
| Konkani | Hindi | 5.92 | 2.85 |
| Konkani | Marathi | 7.73 | 4.08 |
| Konkani | Konkani | 7.65 | 4.09 |
| Arabic | English | 1.27 | 1.21 |
| Arabic | Modern Standard Arabic | 4.74 | 2.12 |
| Arabic | Tunisian Arabic | 4.96 | 2.16 |
| Arabic | Egyptian Arabic | 4.88 | 2.15 |

## Token Efficiency (Tokens per Character)

Lower values indicate better character-level efficiency.

| Dataset | Language | Tower | Hermes |
|---------|----------|-------|--------|
| Konkani | English | 0.306 | 0.261 |
| Konkani | Hindi | 1.329 | 0.641 |
| Konkani | Marathi | 1.267 | 0.669 |
| Konkani | Konkani | 1.234 | 0.660 |
| Arabic | English | 0.301 | 0.284 |
| Arabic | Modern Standard Arabic | 1.176 | 0.527 |
| Arabic | Tunisian Arabic | 1.176 | 0.509 |
| Arabic | Egyptian Arabic | 1.175 | 0.519 |

## Interpretation

### Token Fertility (Tokens per Word)
- **English**: Typically ~1.3-1.5 tokens/word (well-represented in training)
- **Low-resource languages**: Higher values (2-4+) indicate the tokenizer breaks words into more subword units
- **High fertility = potential translation challenges**: More tokens needed to represent the same content

### Token Premium
The 'token premium' for a language is the ratio of its fertility compared to English:
- Premium = (tokens/word for language) / (tokens/word for English)
- Higher premium means the model 'pays more' to represent that language

## Token Premium (vs English)

| Dataset | Language | Tower Premium | Hermes Premium |
|---------|----------|---------------|----------------|
| Konkani | English | 1.00x (baseline) | 1.00x (baseline) |
| Konkani | Hindi | 3.73x | 2.12x |
| Konkani | Marathi | 4.87x | 3.04x |
| Konkani | Konkani | 4.83x | 3.04x |
| Arabic | English | 1.00x (baseline) | 1.00x (baseline) |
| Arabic | Modern Standard Arabic | 3.73x | 1.76x |
| Arabic | Tunisian Arabic | 3.91x | 1.79x |
| Arabic | Egyptian Arabic | 3.84x | 1.78x |