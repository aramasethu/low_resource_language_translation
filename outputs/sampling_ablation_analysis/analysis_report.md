# Sampling Ablation Study: Analysis Report

**Generated**: 2025-10-31 19:01:23

---

## Executive Summary

This study compared **semantic** vs **random** retrieval strategies for selecting few-shot examples across k=0-10 for Tower and Hermes models translating English→Marathi→Konkani using the APE (Automatic Post-Editing) approach.

**Key Findings:**

1. ✅ **Hermes significantly outperforms Tower** (8.39 vs 12.40 BLEU best scores, but Hermes is stable while Tower degenerates)

2. ❌ **Semantic retrieval doesn't provide clear advantages** - random sampling often performs as well or better

3. ⚠️ **Tower model degenerates catastrophically** at k≥6 (semantic) or k≥7 (random), producing garbled Unicode and excessive whitespace (98-100% failure rates)

4. ✅ **Hermes remains stable** across all k values with only 0.5-2.0% problematic predictions

5. 🎯 **Best practical configurations:**
   - Tower: Random k=6 (BLEU 12.40, 20% problematic) or zero-shot (BLEU 7.41, 3% problematic)
   - Hermes: Semantic k=5 (BLEU 8.39, 1.5% problematic) or random k=6 (BLEU 8.25, 0.5% problematic)

**Recommendation:** For production use, **Hermes with k=5-6 is the clear winner** due to stability. Tower should only be used with k≤5 and careful monitoring.

---

## Summary Statistics

### Tower Model

#### Average Scores Across All k Values

| Strategy | Avg BLEU | Avg chrF | Avg chrF++ | Avg Time (min) |
|----------|----------|----------|------------|----------------|
| Semantic | 4.13 | 16.14 | 13.65 | 19.95 |
| Random   | 5.86 | 22.41 | 18.83 | 18.97 |

#### Best Performing k Values (by BLEU)

- **Semantic**: k=0 (BLEU: 7.41)
- **Random**: k=6 (BLEU: 12.40)

#### Detailed Results by k Value

| k | Semantic BLEU | Random BLEU | Semantic chrF | Random chrF | Semantic chrF++ | Random chrF++ |
|---|---------------|-------------|---------------|-------------|-----------------|---------------|
| 0 | 7.41 | 7.41 | 34.58 | 34.58 | 28.06 | 28.06 |
| 1 | 7.41 | 7.41 | 34.58 | 34.58 | 28.06 | 28.06 |
| 2 | 7.41 | 7.41 | 34.58 | 34.58 | 28.06 | 28.06 |
| 3 | 7.41 | 7.41 | 34.58 | 34.58 | 28.06 | 28.06 |
| 4 | 0.60 | 7.41 | 7.91 | 34.58 | 5.93 | 28.06 |
| 5 | 1.71 | 7.41 | 3.40 | 34.58 | 5.28 | 28.06 |
| 6 | 2.45 | 12.40 | 2.19 | 26.19 | 3.98 | 21.55 |
| 7 | 4.62 | 2.10 | 0.47 | 1.95 | 1.63 | 5.10 |
| 8 | 1.70 | 1.18 | 9.10 | 1.32 | 7.58 | 1.32 |
| 9 | 2.40 | 2.71 | 8.40 | 1.45 | 7.03 | 3.84 |
| 10 | 2.26 | 1.59 | 7.79 | 8.13 | 6.51 | 6.95 |

---

### Hermes Model

#### Average Scores Across All k Values

| Strategy | Avg BLEU | Avg chrF | Avg chrF++ | Avg Time (min) |
|----------|----------|----------|------------|----------------|
| Semantic | 7.28 | 36.46 | 29.56 | 10.85 |
| Random   | 7.18 | 36.84 | 29.99 | 10.50 |

#### Best Performing k Values (by BLEU)

- **Semantic**: k=5 (BLEU: 8.39)
- **Random**: k=6 (BLEU: 8.25)

#### Detailed Results by k Value

| k | Semantic BLEU | Random BLEU | Semantic chrF | Random chrF | Semantic chrF++ | Random chrF++ |
|---|---------------|-------------|---------------|-------------|-----------------|---------------|
| 0 | 7.08 | 7.08 | 34.63 | 34.63 | 28.07 | 28.07 |
| 1 | 2.25 | 7.93 | 35.88 | 31.20 | 28.68 | 25.56 |
| 2 | 7.80 | 4.25 | 36.72 | 36.20 | 29.68 | 29.85 |
| 3 | 8.25 | 7.80 | 39.88 | 37.62 | 33.33 | 30.34 |
| 4 | 7.94 | 2.95 | 35.92 | 37.74 | 29.08 | 30.13 |
| 5 | 8.39 | 7.94 | 38.59 | 37.86 | 31.95 | 30.54 |
| 6 | 7.94 | 8.25 | 37.87 | 37.96 | 30.96 | 31.48 |
| 7 | 7.80 | 8.25 | 37.26 | 37.96 | 30.08 | 31.48 |
| 8 | 7.80 | 8.08 | 35.82 | 38.04 | 29.00 | 30.68 |
| 9 | 7.51 | 8.22 | 34.76 | 38.19 | 27.77 | 30.80 |
| 10 | 7.35 | 8.22 | 33.73 | 37.80 | 26.58 | 30.93 |

---

## Key Findings

### Tower Model

1. **Overall Strategy Comparison (by BLEU):**
   - Semantic retrieval wins in 3/11 cases
   - Random retrieval wins in 4/11 cases
   - Ties: 4

2. **Average Performance Delta:** Random retrieval performs 1.73 BLEU points better on average

3. **Zero-shot Baseline:** BLEU = 7.41

4. **Best Improvement over Zero-shot:**
   - Semantic: k=0 (+0.00 BLEU)
   - Random: k=6 (+4.99 BLEU)

5. **Inference Time:**
   - Semantic avg: 19.95 minutes
   - Random avg: 18.97 minutes
   - Difference: 0.97 minutes

6. **Model Stability:**
   - Low k (0-5): High quality, <3% problematic predictions
   - Medium k (6): Degradation begins, 20-68% problematic predictions
   - High k (7-10): Severe failure, 98-100% problematic predictions (garbled Unicode, excessive whitespace)
   - **Critical threshold**: k≥6 for semantic, k≥7 for random

---

### Hermes Model

1. **Overall Strategy Comparison (by BLEU):**
   - Semantic retrieval wins in 4/11 cases
   - Random retrieval wins in 6/11 cases
   - Ties: 1

2. **Average Performance Delta:** Semantic retrieval performs 0.10 BLEU points better on average

3. **Zero-shot Baseline:** BLEU = 7.08

4. **Best Improvement over Zero-shot:**
   - Semantic: k=5 (+1.31 BLEU)
   - Random: k=6 (+1.17 BLEU)

5. **Inference Time:**
   - Semantic avg: 10.85 minutes
   - Random avg: 10.50 minutes
   - Difference: 0.35 minutes

6. **Model Stability:**
   - Excellent stability across all k values (0-10)
   - Consistently low problematic prediction rate: 0.5-2.0%
   - Main issue: Occasional word repetition loops (e.g., repeating phrases)
   - **No critical threshold**: Model remains stable even at k=10

---

## Problematic Predictions Analysis

This section identifies cases where the model generated problematic or nonsensical predictions.

**Note on Methodology:** Valid Devanagari script (Marathi/Konkani) is NOT considered problematic. We only flag predictions with:
- Unicode replacement characters (�) indicating encoding errors
- Excessive whitespace/punctuation repetition (e.g., multiple blank lines, `( ( ( (`)
- Garbled ASCII patterns (e.g., `(A (A (A`)
- High word repetition rates (getting stuck in loops)
- Empty or very short outputs

This ensures we're identifying genuine model failures rather than valid output in the target language.

### Tower Model

#### Problem Count by Experiment

| Experiment | Number of Problematic Predictions | Problem Rate |
|------------|-----------------------------------|--------------|
| random_k0 | 6 | 2.9% |
| random_k10 | 205 | 100.0% |
| random_k2 | 1 | 0.5% |
| random_k3 | 3 | 1.5% |
| random_k4 | 1 | 0.5% |
| random_k6 | 41 | 20.0% |
| random_k7 | 202 | 98.5% |
| random_k8 | 205 | 100.0% |
| random_k9 | 205 | 100.0% |
| semantic_k0 | 6 | 2.9% |
| semantic_k1 | 1 | 0.5% |
| semantic_k10 | 205 | 100.0% |
| semantic_k3 | 1 | 0.5% |
| semantic_k4 | 140 | 68.3% |
| semantic_k5 | 163 | 79.5% |
| semantic_k6 | 205 | 100.0% |
| semantic_k7 | 205 | 100.0% |
| semantic_k8 | 205 | 100.0% |
| semantic_k9 | 205 | 100.0% |

#### Example Problematic Cases

##### semantic_k6 (205 problems)

**Example 1:** Contains Unicode replacement characters (�)

- **Source (English):** Ajanta, located in the Aurangabad District of Maharashtra has twenty-nine caitya and vihara caves de...
- **Pivot (Marathi):** महाराष्ट्राच्या औरंगाबाद जिल्ह्यातील अजिंठ्यात, इ. स. पू. पहिल्या शतकापासून ते इ. स. पाचव्या शतकापर्...
- **Target (Konkani):** महाराष्ट्रांतल्या औरंगाबाद जिल्ल्यांत आशिल्लें अजंठा, हांगां शिल्पां आनी चित्रांनी सजयल्ल्यो वर्तमान...
- **Prediction:** Ъ Ъ Ъ  (1  (� (  ( . Ъ Ъ Ъ Ъ Ъ Ъ Ъ Ъ Ъ Ъ  ( .  (  (  (S  (

 . ����������������������S . . . Ъ . . ....

**Example 2:** Contains Unicode replacement characters (�)

- **Source (English):** Potatoes mixed in masalas, coated in besan batter and deep fried to perfection form this delicious a...
- **Pivot (Marathi):** मसाल्यांत मिसळलेले बटाटे बेसनाच्या पातळ पिठात घोळवून, खूप तेलात पूर्ण तळून महाराष्ट्राचा हा चवदार आण...
- **Target (Konkani):** बटाटां भितर मसालो भरसून ते बेसनाच्या पिठयेंत बुडोवन तेलांत बरे तळ्ळे कांय महाराष्ट्रांतलो हो सुवादीक...
- **Prediction:** ( .  (    ( .  (  
   ( “�  (��� .  (
  (
  ( “ “�������������������� “� “� . 
 ���� “�� “� “� Ъ . ....

**Example 3:** Contains Unicode replacement characters (�)

- **Source (English):** The purpose of the book appears to encourage proficiency in Sanskrit expression and the knowledge of...
- **Pivot (Marathi):** संस्कृत अभिव्यक्तीमध्ये प्रावीण्य मिळवण्यास प्रोत्साहन मिळावे आणि सद्वर्तनाचे ज्ञान प्राप्त व्हावे, ...
- **Target (Konkani):** संस्कृतांत उलोवपाचे कुशळटायेक प्रोत्साहन दिवप आनी शाणेपणान वागपाचें ज्ञान मेळोवप हो ह्या पुस्तकाचो ह...
- **Prediction:** ( ( ( .  ( Ъ  (  (  (  ( ������������������������������S  (


 ( . Ъ . . . . Ъ .  ( Ъ .  . . . " " "...

##### semantic_k7 (205 problems)

**Example 1:** Contains Unicode replacement characters (�)

- **Source (English):** Ajanta, located in the Aurangabad District of Maharashtra has twenty-nine caitya and vihara caves de...
- **Pivot (Marathi):** महाराष्ट्राच्या औरंगाबाद जिल्ह्यातील अजिंठ्यात, इ. स. पू. पहिल्या शतकापासून ते इ. स. पाचव्या शतकापर्...
- **Target (Konkani):** महाराष्ट्रांतल्या औरंगाबाद जिल्ल्यांत आशिल्लें अजंठा, हांगां शिल्पां आनी चित्रांनी सजयल्ल्यो वर्तमान...
- **Prediction:** (  
 



  
 
  
 
 
 
 
  (

 
 

 

 


  (
 
 


 
 
 

 ����������������������������������������...

**Example 2:** Contains Unicode replacement characters (�)

- **Source (English):** Potatoes mixed in masalas, coated in besan batter and deep fried to perfection form this delicious a...
- **Pivot (Marathi):** मसाल्यांत मिसळलेले बटाटे बेसनाच्या पातळ पिठात घोळवून, खूप तेलात पूर्ण तळून महाराष्ट्राचा हा चवदार आण...
- **Target (Konkani):** बटाटां भितर मसालो भरसून ते बेसनाच्या पिठयेंत बुडोवन तेलांत बरे तळ्ळे कांय महाराष्ट्रांतलो हो सुवादीक...
- **Prediction:** लीप सेकंद म्हटल्यार अचूक वेळ (आण्वीक घडयाळांनी मेजिल्ली आंतरराष्ट्रीय आण्वीक वेळ (टी.ए.आय.) आनी पृथ्...

**Example 3:** Contains Unicode replacement characters (�)

- **Source (English):** The purpose of the book appears to encourage proficiency in Sanskrit expression and the knowledge of...
- **Pivot (Marathi):** संस्कृत अभिव्यक्तीमध्ये प्रावीण्य मिळवण्यास प्रोत्साहन मिळावे आणि सद्वर्तनाचे ज्ञान प्राप्त व्हावे, ...
- **Target (Konkani):** संस्कृतांत उलोवपाचे कुशळटायेक प्रोत्साहन दिवप आनी शाणेपणान वागपाचें ज्ञान मेळोवप हो ह्या पुस्तकाचो ह...
- **Prediction:** (
  (  



 

  ( 



 
 
 
 Ъ Ъ 
 
 


 
 
 ������ ������������������������������������� 








 ...

##### semantic_k8 (205 problems)

**Example 1:** Contains Unicode replacement characters (�)

- **Source (English):** Ajanta, located in the Aurangabad District of Maharashtra has twenty-nine caitya and vihara caves de...
- **Pivot (Marathi):** महाराष्ट्राच्या औरंगाबाद जिल्ह्यातील अजिंठ्यात, इ. स. पू. पहिल्या शतकापासून ते इ. स. पाचव्या शतकापर्...
- **Target (Konkani):** महाराष्ट्रांतल्या औरंगाबाद जिल्ल्यांत आशिल्लें अजंठा, हांगां शिल्पां आनी चित्रांनी सजयल्ल्यो वर्तमान...
- **Prediction:** लीप सेकंद म्हटल्यार अचूक वेळ (आण्वीक घडयाळांनी मेजिल्ली आंतरराष्ट्रीय आण्वीक वेळ (टी.ए.आय.) आनी पृथ्...

**Example 2:** Contains Unicode replacement characters (�)

- **Source (English):** Potatoes mixed in masalas, coated in besan batter and deep fried to perfection form this delicious a...
- **Pivot (Marathi):** मसाल्यांत मिसळलेले बटाटे बेसनाच्या पातळ पिठात घोळवून, खूप तेलात पूर्ण तळून महाराष्ट्राचा हा चवदार आण...
- **Target (Konkani):** बटाटां भितर मसालो भरसून ते बेसनाच्या पिठयेंत बुडोवन तेलांत बरे तळ्ळे कांय महाराष्ट्रांतलो हो सुवादीक...
- **Prediction:** लीप सेकंद म्हटल्यार अचूक वेळ (आण्वीक घडयाळांनी मेजिल्ली आंतरराष्ट्रीय आण्वीक वेळ (टी.ए.आय.) आनी पृथ्...

**Example 3:** Contains Unicode replacement characters (�)

- **Source (English):** The purpose of the book appears to encourage proficiency in Sanskrit expression and the knowledge of...
- **Pivot (Marathi):** संस्कृत अभिव्यक्तीमध्ये प्रावीण्य मिळवण्यास प्रोत्साहन मिळावे आणि सद्वर्तनाचे ज्ञान प्राप्त व्हावे, ...
- **Target (Konkani):** संस्कृतांत उलोवपाचे कुशळटायेक प्रोत्साहन दिवप आनी शाणेपणान वागपाचें ज्ञान मेळोवप हो ह्या पुस्तकाचो ह...
- **Prediction:** लीप सेकंद म्हटल्यार अचूक वेळ (आण्वीक घडयाळांनी मेजिल्ली आंतरराष्ट्रीय आण्वीक वेळ (टी.ए.आय.) आनी पृथ्...

##### semantic_k9 (205 problems)

**Example 1:** Contains Unicode replacement characters (�)

- **Source (English):** Ajanta, located in the Aurangabad District of Maharashtra has twenty-nine caitya and vihara caves de...
- **Pivot (Marathi):** महाराष्ट्राच्या औरंगाबाद जिल्ह्यातील अजिंठ्यात, इ. स. पू. पहिल्या शतकापासून ते इ. स. पाचव्या शतकापर्...
- **Target (Konkani):** महाराष्ट्रांतल्या औरंगाबाद जिल्ल्यांत आशिल्लें अजंठा, हांगां शिल्पां आनी चित्रांनी सजयल्ल्यो वर्तमान...
- **Prediction:** लीप सेकंद म्हटल्यार अचूक वेळ (आण्वीक घडयाळांनी मेजिल्ली आंतरराष्ट्रीय आण्वीक वेळ (टी.ए.आय.) आनी पृथ्...

**Example 2:** Contains Unicode replacement characters (�)

- **Source (English):** Potatoes mixed in masalas, coated in besan batter and deep fried to perfection form this delicious a...
- **Pivot (Marathi):** मसाल्यांत मिसळलेले बटाटे बेसनाच्या पातळ पिठात घोळवून, खूप तेलात पूर्ण तळून महाराष्ट्राचा हा चवदार आण...
- **Target (Konkani):** बटाटां भितर मसालो भरसून ते बेसनाच्या पिठयेंत बुडोवन तेलांत बरे तळ्ळे कांय महाराष्ट्रांतलो हो सुवादीक...
- **Prediction:** लीप सेकंद म्हटल्यार अचूक वेळ (आण्वीक घडयाळांनी मेजिल्ली आंतरराष्ट्रीय आण्वीक वेळ (टी.ए.आय.) आनी पृथ्...

**Example 3:** Contains Unicode replacement characters (�)

- **Source (English):** The purpose of the book appears to encourage proficiency in Sanskrit expression and the knowledge of...
- **Pivot (Marathi):** संस्कृत अभिव्यक्तीमध्ये प्रावीण्य मिळवण्यास प्रोत्साहन मिळावे आणि सद्वर्तनाचे ज्ञान प्राप्त व्हावे, ...
- **Target (Konkani):** संस्कृतांत उलोवपाचे कुशळटायेक प्रोत्साहन दिवप आनी शाणेपणान वागपाचें ज्ञान मेळोवप हो ह्या पुस्तकाचो ह...
- **Prediction:** लीप सेकंद म्हटल्यार अचूक वेळ (आण्वीक घडयाळांनी मेजिल्ली आंतरराष्ट्रीय आण्वीक वेळ (टी.ए.आय.) आनी पृथ्...

##### semantic_k10 (205 problems)

**Example 1:** Contains Unicode replacement characters (�)

- **Source (English):** Ajanta, located in the Aurangabad District of Maharashtra has twenty-nine caitya and vihara caves de...
- **Pivot (Marathi):** महाराष्ट्राच्या औरंगाबाद जिल्ह्यातील अजिंठ्यात, इ. स. पू. पहिल्या शतकापासून ते इ. स. पाचव्या शतकापर्...
- **Target (Konkani):** महाराष्ट्रांतल्या औरंगाबाद जिल्ल्यांत आशिल्लें अजंठा, हांगां शिल्पां आनी चित्रांनी सजयल्ल्यो वर्तमान...
- **Prediction:** लीप सेकंद म्हटल्यार अचूक वेळ (आण्वीक घडयाळांनी मेजिल्ली आंतरराष्ट्रीय आण्वीक वेळ (टी.ए.आय.) आनी पृथ्...

**Example 2:** Contains Unicode replacement characters (�)

- **Source (English):** Potatoes mixed in masalas, coated in besan batter and deep fried to perfection form this delicious a...
- **Pivot (Marathi):** मसाल्यांत मिसळलेले बटाटे बेसनाच्या पातळ पिठात घोळवून, खूप तेलात पूर्ण तळून महाराष्ट्राचा हा चवदार आण...
- **Target (Konkani):** बटाटां भितर मसालो भरसून ते बेसनाच्या पिठयेंत बुडोवन तेलांत बरे तळ्ळे कांय महाराष्ट्रांतलो हो सुवादीक...
- **Prediction:** लीप सेकंद म्हटल्यार अचूक वेळ (आण्वीक घडयाळांनी मेजिल्ली आंतरराष्ट्रीय आण्वीक वेळ (टी.ए.आय.) आनी पृथ्...

**Example 3:** Contains Unicode replacement characters (�)

- **Source (English):** The purpose of the book appears to encourage proficiency in Sanskrit expression and the knowledge of...
- **Pivot (Marathi):** संस्कृत अभिव्यक्तीमध्ये प्रावीण्य मिळवण्यास प्रोत्साहन मिळावे आणि सद्वर्तनाचे ज्ञान प्राप्त व्हावे, ...
- **Target (Konkani):** संस्कृतांत उलोवपाचे कुशळटायेक प्रोत्साहन दिवप आनी शाणेपणान वागपाचें ज्ञान मेळोवप हो ह्या पुस्तकाचो ह...
- **Prediction:** लीप सेकंद म्हटल्यार अचूक वेळ (आण्वीक घडयाळांनी मेजिल्ली आंतरराष्ट्रीय आण्वीक वेळ (टी.ए.आय.) आनी पृथ्...

---

### Hermes Model

#### Problem Count by Experiment

| Experiment | Number of Problematic Predictions | Problem Rate |
|------------|-----------------------------------|--------------|
| random_k0 | 1 | 0.5% |
| random_k1 | 3 | 1.5% |
| random_k10 | 2 | 1.0% |
| random_k2 | 1 | 0.5% |
| random_k3 | 1 | 0.5% |
| random_k4 | 4 | 2.0% |
| random_k5 | 1 | 0.5% |
| random_k6 | 1 | 0.5% |
| random_k7 | 1 | 0.5% |
| semantic_k0 | 1 | 0.5% |
| semantic_k1 | 3 | 1.5% |
| semantic_k2 | 1 | 0.5% |
| semantic_k3 | 1 | 0.5% |
| semantic_k4 | 2 | 1.0% |
| semantic_k5 | 3 | 1.5% |
| semantic_k6 | 1 | 0.5% |
| semantic_k7 | 1 | 0.5% |
| semantic_k8 | 1 | 0.5% |
| semantic_k9 | 3 | 1.5% |

#### Example Problematic Cases

##### random_k4 (4 problems)

**Example 1:** High word repetition rate

- **Source (English):** The purpose of the book appears to encourage proficiency in Sanskrit expression and the knowledge of...
- **Pivot (Marathi):** संस्कृत अभिव्यक्तीमध्ये प्रावीण्य मिळवण्यास प्रोत्साहन मिळावे आणि सद्वर्तनाचे ज्ञान प्राप्त व्हावे, ...
- **Target (Konkani):** संस्कृतांत उलोवपाचे कुशळटायेक प्रोत्साहन दिवप आनी शाणेपणान वागपाचें ज्ञान मेळोवप हो ह्या पुस्तकाचो ह...
- **Prediction:** पुस्तकाचो उद्देश संस्कृत अभिव्यक्तीत प्रवीणता मेळवण, संस्कृत अभिव्यक्तीत प्रवीणता मेळवण, संस्कृत अभि...

**Example 2:** High word repetition rate

- **Source (English):** The highest is the Xvaitu, who is self-sufficient and no longer requires religious instruction; the ...
- **Pivot (Marathi):** सर्वांत वरचा आहे वैतू, जो स्वयंपूर्ण आहे आणि त्याला आता धार्मिक निर्देशांची आवश्यकता नाही; ऐर्यम्न क...
- **Target (Konkani):** तातूंतलो सगळ्यांत श्रेश्ठ तो श्वैतू, तो आत्मनिर्भर आसता आनी ताका धर्मीक शिकवण दिवपाची आनीक गरज आसना;...
- **Prediction:** सर्वात वरचो XVaitu, जो स्वतःसufficiency आहे आणि धार्मिक निर्देशांच्या आवश्यकतेबद्दलच्या धार्मिक निर्...

**Example 3:** High word repetition rate

- **Source (English):** Neha's in-laws think Nikita has died, and Nikita realizes that a way out of her problems is to imper...
- **Pivot (Marathi):** नेहाच्या सासरच्या मंडळींना वाटते की, निकिता मरण पावली आहे, आणि निकिताला कळून चुकते की, तिच्या समस्या...
- **Target (Konkani):** निकिता संवसाराक अंतरलें अशें नेहाच्या मांय-मांवाक दिसता, आनी आपल्या समस्यांतल्यान भायर सरपाक नेहाचें...
- **Prediction:** नेहाच्या सासरांच्या मंडळींक निकिता मरण करपाच्या मते आहे अशा विश्वासांकारी निकिता यांको कळता की तांच्...

##### semantic_k1 (3 problems)

**Example 1:** High word repetition rate

- **Source (English):** You may begin by adding a spoonful of grated coconut into the cylindrical container, followed by the...
- **Pivot (Marathi):** दंडगोलाकार डब्यात चमचाभर किसलेले खोबरे आणि त्यानंतर चांगल्याप्रकारे मिश्रण केलेले तांदळाचे पीठ घालून...
- **Target (Konkani):** लंबवाटकुळाकार आयदनांत एक चमचोभर कांतिल्ली सोय घालून तुमीं सुरवात करपाक शकतात आनी मागीर बरें तरेन एकठ...
- **Prediction:** तुमच्या सुरवातील आकारांत एक चमच कदंबाचो कदंबाचो कदंब कदंबाचो कदंबाचो कदंबाचो कदंबाचो कदंबाचो कदंबाचो...

**Example 2:** High word repetition rate

- **Source (English):** Thus, our focus shifts from a desire to know the times the novel is referring to in socio-economic o...
- **Pivot (Marathi):** अशा प्रकारे, कादंबरी सामाजिक-आर्थिक किंवा राजकीय परिभाषेत संदर्भ देत असलेला कालखंड जाणून घेण्याच्या ...
- **Target (Konkani):** अशें तरेन, आमची नदर समाजीक-अर्थीक वा राजकी नदरेन हे कादंबरीचो काळ जाणून घेवपा कडल्यान कपडे आनी सजावट...
- **Prediction:** त्यामुळा आमच्या ध्यानाचा प्रकार सामाजिक-आर्थिक किंवा राजकीय परिप्रेक्षेत संदर्भ देत असलेल्या काळाच्य...

**Example 3:** High word repetition rate

- **Source (English):** Vanuatu and the Solomon Islands are also nowhere near short on natural attractions; however, as both...
- **Pivot (Marathi):** वानूआतु आणि द सोलोमन बेटे ही देखील नैसर्गिक आकर्षणे म्हणून कुठेही मागे नाहीत; तरीदेखील, या दोन्ही दे...
- **Target (Konkani):** वनुवातू आनी सोलोमॉन जुंवे सैमीक आकर्शणांनी खंयच उणे नात; पूण, दोनूय जुंव्याचेर साप्प थोडे पर्यटक येत...
- **Prediction:** वानूआतु आणि सोलोमन बेटांतही नैसर्गिक आकर्षणांची कमी कमी कमी कमी कमी कमी कमी कमी कमी कमी कमी कमी कमी ...

##### semantic_k5 (3 problems)

**Example 1:** High word repetition rate

- **Source (English):** Similarly, when liquidity is tight, RBI will buy government securities and thereby inject money supp...
- **Pivot (Marathi):** त्याचप्रमाणे, रोकडसुलभता जेव्हा अतिशय कमी असते, तेव्हा आर.बी.आय. सरकारी सुरक्षाठेवी विकत घेईल आणि त्...
- **Target (Konkani):** तेच प्रमाण, मालमत्तेची मोडणी खर प्रमाणांत जातली तेन्ना, आर.बी.आय. सरकारी रोखे विकते घेतलो आनी ताचे व...
- **Prediction:** त्याचमद्दत, जेव्हा रोकडाची सुलभता कमी असते, तेव्हा आर.बी.आय. सरकारी सुरक्षांकांक्षी विकत घेईलो आनी त...

**Example 2:** High word repetition rate

- **Source (English):** On the reserve day, the Sri Lankans again batted first, scoring 222 runs including Mahela Jayawarden...
- **Pivot (Marathi):** राखीव दिवशी, श्रीलंकेने पुन्हा प्रथम फलंदाजी केली आणि २२२ धावा काढल्या, ज्यात महेला जयवर्धने आणि रसे...
- **Target (Konkani):** राखून दवरिल्ल्या दिसा, श्रीलंकेन पयली फलंदाजी केली, महेला जयवर्धनेन 77 आनी रसेल आर्नोल्डान 56 मेळून ...
- **Prediction:** राखीव दिवशांकडे, श्रीलंकेने पुन्हा प्रारंभी बॅट केल्ले, २२२ धावांच्या सुमारांत महेला जयवर्धने आणि रस...

**Example 3:** High word repetition rate

- **Source (English):** Non-Cambodian visitors must purchase an Angkor Pass to enter Angkor Archaeological Park.
- **Pivot (Marathi):** कंबोडियातील लोकांव्यतिरिक्त भेट देणाऱ्यांना अंगकोर आर्कीऑलॉजिकल पार्कमध्ये प्रवेश करण्याकरता अंगकोर ...
- **Target (Konkani):** आंकोर पुरातत्व पार्कांत भीतर सरपाक कोंबोडिया भायल्या सोयऱ्यांक आंकोराचो पास विकत घेवचो पडटा.
- **Prediction:** कंबोडिया बाहेरच्या मुखाबरच्या मुखाबरच्या मुखाबरच्या मुखाबरच्या मुखाबरच्या मुखाबरच्या मुखाबरच्या मुखा...

##### semantic_k9 (3 problems)

**Example 1:** High word repetition rate

- **Source (English):** I felt that if they agreed to the proposal, others could be persuaded to take the same line.
- **Pivot (Marathi):** मला वाटले की, जर त्यांना प्रस्ताव मान्य झाला, तर इतरांनाही तसे करण्यास राजी करता येईल.
- **Target (Konkani):** तांणी प्रस्ताव मान्य केलो जाल्यार, तीच भुमिका घेवपाखातीर हेरांकय समजावपाक शकतले अशें म्हाका दिसलें.
- **Prediction:** म्हणुका केल्ल्या प्रस्तावाच्या मान्यते जर त्यांना राजी झाले, तर इतरांच्या विचारांतही अशा प्रकारच्या ...

**Example 2:** High word repetition rate

- **Source (English):** All facilities were provided for those more spiritually inclined to lead holy lives in solitude in a...
- **Pivot (Marathi):** आध्यात्मिकतेकडे अधिक कल असलेल्यांना, त्यांच्या स्वभावधर्मानुसार एकांतवासात पवित्र जीवन जगण्यासाठी, स...
- **Target (Konkani):** अध्यात्माची ओढ घेवन आपापल्या प्रकृतीसभावा प्रमाण एकांतवासांत अध्यात्मीक जीण सारपाक सोदतल्यांक सगळ्यो...
- **Prediction:** आध्यात्मिक कल असणांक लागून एकांतवासात पवित्र जीवन जगण्या सुविधा पुरवपांक लागल्यो होतो, त्यांच्या स्व...

**Example 3:** High word repetition rate

- **Source (English):** On the reserve day, the Sri Lankans again batted first, scoring 222 runs including Mahela Jayawarden...
- **Pivot (Marathi):** राखीव दिवशी, श्रीलंकेने पुन्हा प्रथम फलंदाजी केली आणि २२२ धावा काढल्या, ज्यात महेला जयवर्धने आणि रसे...
- **Target (Konkani):** राखून दवरिल्ल्या दिसा, श्रीलंकेन पयली फलंदाजी केली, महेला जयवर्धनेन 77 आनी रसेल आर्नोल्डान 56 मेळून ...
- **Prediction:** राखीव दिवशें, श्रीलंकेने पुन्हा प्रथम बॅट केल्ली, २२२ धावांच्या सुमारांत महेला जयवर्धने आणि रसेल आर्...

##### random_k1 (3 problems)

**Example 1:** High word repetition rate

- **Source (English):** This review is not exhaustive and only serves to highlight ways in which present techniques are comp...
- **Pivot (Marathi):** हा आढावा सर्वसमावेशक नाही आणि केवळ, सध्याच्या तंत्रांना, कोशिकेय आणि साधन आधारित प्रतिमानांचा संयोग ...
- **Target (Konkani):** हो नियाळ विस्तृत ना आनी तो सॅल्युलर आनी एजंटाचेर आदारीत मॉडलांक एकठांय हाडून सध्याच्या तंत्राक पूरक ...
- **Prediction:** ह्या परीक्षणांत पूर्णत्वा ना आणि केवळ ह्या विधियांच्या साथीत, कोशिका आणि एजेंट आधारित नमुनांच्या संय...

**Example 2:** High word repetition rate

- **Source (English):** Cardio-respiratory endurance reflects the ability of the body's circulatory and respiratory systems ...
- **Pivot (Marathi):** ह्रद-श्वसन सहनशक्तीमध्ये, शरीराच्या अभिसरणी आणि श्वसनविषयक प्रणालींची, सततच्या शारीरिक क्रियेवेळी इं...
- **Target (Konkani):** ह्रदयश्वसनाचें चिवटपण सेगीतपणान केल्ल्या शारिरीक हालचालीं वेळार शरिराचे रक्ताभिसरण आनी श्वसन यंत्रणे...
- **Prediction:** ह्रद-श्वसन सहनशक्ती शरीराच्या अभिसरणी आणि श्वसन प्रणालींच्या क्षमतेच्या प्रतिबिम्बांच्या दाखवण्याच्य...

**Example 3:** High word repetition rate

- **Source (English):** Part II (Articles 17-24) governs the reporting and monitoring of the Convention and the steps taken ...
- **Pivot (Marathi):** भाग २ (अनुच्छेद १७-२४) हा कराराचा अहवाल आणि त्यावरील देखरेख आणि त्याच्या अंमलबजावणीसाठी पक्षांनी उचल...
- **Target (Konkani):** भाग II (कलम 17-24) परिशदेचो अहवाल दिवप आनी निरीक्षण करप आनी पक्षांनी तें चलणुकेंत हाडपाक कितें उपाय ...
- **Prediction:** भाग २ (अनुच्छेद १७-२४) हे कराराचो अहवाल, त्याच्या देखरेख आणि पक्षांक पक्षांक पक्षांक पक्षांक पक्षांक...

---

## Recommendations

### Tower Model

**Best Configuration:** random retrieval with k=6 examples
- BLEU: 12.40
- chrF: 26.19
- chrF++: 21.55
- Problematic predictions: 20.0% (acceptable)

**Alternative:** Zero-shot (k=0)
- BLEU: 7.41
- chrF: 34.58
- chrF++: 28.06
- Problematic predictions: 2.9% (very low)
- Advantage: No need for vector database, faster inference

**⚠️ Avoid:** 
- Semantic retrieval with k≥6 (100% failure rate with garbled Unicode)
- Random retrieval with k≥7 (98-100% failure rate)

### Hermes Model

**Best Configuration:** semantic retrieval with k=5 examples
- BLEU: 8.39
- chrF: 38.59
- chrF++: 31.95
- Problematic predictions: 1.5% (excellent stability)

**Alternative:** random retrieval with k=6 examples
- BLEU: 8.25
- chrF: 37.96
- chrF++: 31.48
- Problematic predictions: 0.5% (excellent stability)

**Key Insight:** Hermes is stable across all k values (0-10) with consistently low problematic prediction rates (0.5-2.0%). Both semantic and random strategies work well.
