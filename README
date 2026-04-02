# Linguistic Analysis of Annotator Disagreement in Hate Speech Data

## Pipeline Overview

```
Raw MHS data
    ↓
relevant_dataset.py       →  filters and cleans raw data
    ↓
disagreement_score.py     →  computes annotator disagreement via normalized entropy
    ↓
feature_extraction.py     →  extracts 22 linguistic features per comment
    ↓
logistic_regression_model.py  →  trains classifier, outputs feature importance
```

---

## Folder Structure

```
hatespeech-disagreement-nlp/
│
├── data/
│   ├── raw/
│   │   ├── measuring_hate_speech_raw.csv   ← original MHS dataset
│   │   └── hatespeech_raw.csv              ← filtered subset used in pipeline
│   └── processed/
│       ├── hatespeech_disagreement.csv     ← disagreement scores per comment
│       ├── hatespeech_features.csv         ← 22 features per comment
│       └── hatespeech_model_results.csv    ← logistic regression coefficients
│
├── src/
│   ├── relevant_dataset.py                 ← filters raw MHS data
│   ├── disagreement_score.py               ← computes normalized entropy scores
│   ├── feature_extraction.py               ← extracts linguistic features
│   ├── logistic_regression_model.py        ← trains model, ranks features
│   ├── eda_test_dataset.py                 ← exploratory data analysis
│   ├── debug_labels.py                     ← debugging script for label checks
│   └── imbalance_diagreement_check.py      ← checks class imbalance
│
├── plots/
│   ├── annotators_per_comment.png
│   ├── disagreement_imbalance.png
│   ├── hate_speech_score_dist.png
│   └── text_length_dist.png
│
├── project_config.py                       ← all file paths in one place
├── dataset_download.py                     ← downloads raw MHS dataset
├── .env                                    ← local paths (not tracked by git)
└── .gitignore
```

---

## Dataset

**Measuring Hate Speech** — Kennedy et al. (2022)  
- 135,556 annotations across ~40,000 comments  
- Each comment rated by multiple annotators on hate speech, sentiment, and related dimensions  
- Annotator-level ratings used to compute disagreement scores via normalized entropy  

Download: [HuggingFace — ucberkeley-dsp/measuring-hate-speech](https://huggingface.co/datasets/ucberkeley-dsp/measuring-hate-speech)

---

## Features Extracted (22)
```
| Feature | Description |
|---|---|

| `sent_length` | Number of tokens in the comment |
| `negation_count` | Count of negation words (not, never, no...) |
| `hedge_count` | Count of hedging words (maybe, might, possibly...) |
| `pronoun_count` | Count of pronouns — group targeting signal |
| `verb_count` | Count of verbs — action intensity |
| `uncertainty_count` | Count of uncertainty words (unclear, ambiguous...) |
| `contrast_count` | Count of contrast words (but, however, although...) |
| `sarcasm_markers` | Punctuation/typographic sarcasm signals |
| `intensity_count` | Count of intensifiers (absolutely, extremely...) |
| `exclamation_count` | Number of exclamation marks |
| `dep_depth` | Syntactic depth — grammatical complexity |
| `clause_count` | Number of subordinate clauses |
| `modal_count` | Count of modal verbs (should, would, must...) |
| `type_token_ratio` | Lexical diversity (0=repetitive, 1=all unique) |
| `is_question` | Whether the comment ends with a question mark |
| `polarity` | Overall sentiment (-1=negative, +1=positive) |
| `emotion_diversity` | Whether both positive and negative tones are present |
| `emotion_intensity` | Strength of overall emotion (0=weak, 1=strong) |
| `emotional_conflict` | Whether comment pulls in both emotional directions |
| `profanity_count` | Count of profane/offensive words |
| `allcaps_ratio` | Proportion of words written in ALL CAPS |
| `avg_token_length` | Average word length — proxy for language complexity |

```

---

## Tools Used

 Tool                                           Purpose 

 spaCy (`en_core_web_sm`)                  POS tagging, dependency parsing, syntactic features 
 TextBlob                                  Sentiment polarity 
 VADER (NLTK)                              Emotion intensity and emotional conflict detection 
 better_profanity                          Profanity detection 
 Custom word lists                         Hedging, negation, contrast, uncertainty, intensity 
 Scikit-learn                              Logistic Regression classifier 

---

## Setup

```bash
git clone <https link>
```

Now change your directory to the project folder
```
cd hatespeech-disagreement-nlp
```

Install all the pre-requisties for the project
```
pip install -r requirements.txt
```

load the english model for spacy, used for feature extraction
```
python -m spacy download en_core_web_sm
```

Create a `.env` file in the root with your local base path:

```
BASE_PATH=C:/your/path/to/project
```

---

## Results (Logistic Regression)

- Test accuracy: **0.60**
- Cross-val accuracy: **0.60 ± 0.003**

**Top features triggering HIGH disagreement:** sarcasm markers, profanity count, emotional conflict, emotion diversity

**Top features triggering LOW disagreement:** sentence length, polarity, pronoun count, syntactic depth

---

## Requirements

```
pandas
numpy
spacy
textblob
nltk
better_profanity
scikit-learn
python-dotenv
```