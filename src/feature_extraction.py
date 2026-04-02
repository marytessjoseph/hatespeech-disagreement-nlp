import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
import re
import nltk
import spacy
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from better_profanity import profanity
from project_config import PROCESSED_DATA, FEATURES_FILE

nltk.download('vader_lexicon', quiet=True)

df = pd.read_csv(PROCESSED_DATA)

# Load spacy
nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

# Initialise VADER once
vader = SentimentIntensityAnalyzer()

# --- Word lists ---
negation_words = {
    "not", "no", "never", "neither", "nor",
    "nobody", "nothing", "nowhere", "none", "n't",
    "hardly", "barely", "scarcely", "rarely", "seldom",
    "lack", "lacks", "lacking", "fail", "fails", "failed",
    "refuse", "refuses", "refused", "deny", "denies", "denied",
    "without", "absence",
}

hedge_words = {
    "maybe", "perhaps", "might", "could", "possibly",
    "probably", "seems", "appears", "may", "should", "would",
    "believe", "suppose", "guess", "assume", "suspect", "reckon",
    "somewhat", "fairly", "rather", "quite", "pretty",
    "roughly", "approximately",
    "apparently", "allegedly", "supposedly", "seemingly",
    "reportedly", "presumably", "sometimes", "occasionally",
}

uncertainty_words = {
    "unclear", "unsure", "uncertain", "confused", "confusing",
    "ambiguous", "vague", "mixed", "complicated", "complex",
    "depends", "depending", "whatever", "whichever",
    "somehow", "something", "someone", "somewhere",
    "unknown", "unresolved", "undecided", "unexplained",
    "puzzling", "puzzled", "baffling", "baffled",
    "perplexed", "bewildered", "conflicted", "torn",
    "weird", "strange", "odd", "peculiar",
    "debatable", "questionable", "doubtful", "dubious",
    "controversial", "contested",
}

contrast_words = {
    "but", "however", "yet", "although", "though",
    "despite", "while", "whereas", "nevertheless",
    "nonetheless", "still", "except", "unless", "rather",
    "admittedly", "granted", "instead", "alternatively",
    "conversely", "surprisingly", "unexpectedly",
    "ironically", "paradoxically", "oddly", "strangely",
    "partially", "partly", "mostly", "anymore",
    "formerly", "previously", "originally",
}

intensity_words = {
    "absolutely", "literally", "totally", "completely", "utterly",
    "extremely", "incredibly", "insanely", "ridiculously", "so",
    "very", "really", "truly", "deeply", "seriously", "honestly",
    "genuinely", "actually", "just",
}

sarcasm_pattern = re.compile(
    r'[*"\']{1,2}\w+[*"\']{1,2}'
    r'|\.{3,}'
    r'|!{2,}'
    r'|\?{2,}'
    r'|\?!'
    r'|!\?'
    r'|_{1,2}\w+_{1,2}'
    r'|\b[A-Z]{3,}\b'
    r'|(?:[A-Z][a-z]){2,}'
    r'|/s\b'
    r'|/sarc\b',
    re.IGNORECASE
)

# --- Feature extraction ---
def extract_features(doc, text):
    lower_tokens     = [t.text.lower() for t in doc if not t.is_space]
    pos_tags         = [t.pos_ for t in doc]
    n                = len(lower_tokens) or 1
    vs               = vader.polarity_scores(text)
    cap_words        = [t for t in doc if not t.is_space and len(t.text) >= 3 and t.text.isupper()]
    allcaps_ratio    = len(cap_words) / n
    avg_token_length = np.mean([len(t) for t in lower_tokens]) if lower_tokens else 0

    return {
        # Surface
        "sent_length"       : len(doc),
        "negation_count"    : sum(1 for t in lower_tokens if t in negation_words),
        "hedge_count"       : sum(1 for t in lower_tokens if t in hedge_words),
        "pronoun_count"     : pos_tags.count("PRON"),
        "verb_count"        : pos_tags.count("VERB"),
        "uncertainty_count" : sum(1 for t in lower_tokens if t in uncertainty_words),
        "contrast_count"    : sum(1 for t in lower_tokens if t in contrast_words),
        "sarcasm_markers"   : len(sarcasm_pattern.findall(text)),
        "intensity_count"   : sum(1 for t in lower_tokens if t in intensity_words),
        "exclamation_count" : len(re.findall(r'(?<!!)!(?!!)', text)),
        # Syntactic
        "dep_depth"         : max((len(list(t.ancestors)) for t in doc), default=0),
        "clause_count"      : sum(1 for t in doc if t.dep_ in {"ccomp", "advcl", "relcl"}),
        "modal_count"       : sum(1 for t in doc if t.tag_ == "MD"),
        "type_token_ratio"  : len(set(lower_tokens)) / n,
        "is_question"       : int(doc[-1].text == "?") if len(doc) > 0 else 0,
        # Sentiment
        "polarity"          : TextBlob(text).sentiment.polarity,
        # Emotion (VADER)
        "emotion_diversity" : int(vs['pos'] > 0) + int(vs['neg'] > 0),
        "emotion_intensity" : abs(vs['compound']),
        "emotional_conflict": int(vs['pos'] > 0.1 and vs['neg'] > 0.1),
        # New features
        "profanity_count"   : sum(1 for t in lower_tokens if profanity.contains_profanity(t)),
        "allcaps_ratio"     : allcaps_ratio,
        "avg_token_length"  : avg_token_length,
    }

# --- Batch process with progress ---
texts   = df["text"].astype(str).tolist()
records = []
total   = len(texts)
start   = time.time()

print("=" * 60)
print("FEATURE EXTRACTION — MHS DATASET")
print(f"Total comments to process: {total}")
print("=" * 60)

for i, (doc, text) in enumerate(zip(nlp.pipe(texts, batch_size=512, n_process=1), texts), 1):
    records.append(extract_features(doc, text))
    if i % 5000 == 0 or i == total:
        elapsed = time.time() - start
        print(f"  Processed {i}/{total} comments... ({elapsed:.1f}s elapsed)")

elapsed_total = time.time() - start
print(f"\nFeature extraction complete in {elapsed_total:.1f}s")

# --- Build final dataframe ---
features_df = pd.DataFrame(records)
df_final    = pd.concat([df, features_df], axis=1)

os.makedirs(os.path.dirname(FEATURES_FILE), exist_ok=True)
df_final.to_csv(FEATURES_FILE, index=False)

# --- Summary output ---
print("=" * 60)
print(f"\nInput  : hatespeech_disagreement.csv")
print(f"Output : hatespeech_features.csv")

print(f"\n--- Dataset ---")
print(f"Comments : {df_final.shape[0]}")
print(f"Columns  : {df_final.shape[1]}")

print(f"\n--- Disagreement Labels ---")
counts = df_final['disagreement_label'].value_counts()
for label, count in counts.items():
    pct = count / len(df_final) * 100
    print(f"  {label:<6} : {count} ({pct:.1f}%)")

feature_descriptions = {
    "sent_length"       : "Number of tokens in the comment",
    "negation_count"    : "Count of negation words (not, never, no...)",
    "hedge_count"       : "Count of hedging words (maybe, might, possibly...)",
    "pronoun_count"     : "Count of pronouns — group targeting signal",
    "verb_count"        : "Count of verbs — action intensity",
    "uncertainty_count" : "Count of uncertainty words (unclear, ambiguous...)",
    "contrast_count"    : "Count of contrast words (but, however, although...)",
    "sarcasm_markers"   : "Punctuation/typographic sarcasm signals",
    "intensity_count"   : "Count of intensifiers (absolutely, extremely...)",
    "exclamation_count" : "Number of exclamation marks",
    "dep_depth"         : "Syntactic depth — grammatical complexity",
    "clause_count"      : "Number of subordinate clauses",
    "modal_count"       : "Count of modal verbs (should, would, must...)",
    "type_token_ratio"  : "Lexical diversity (0=repetitive, 1=all unique)",
    "is_question"       : "Whether the comment ends with a question mark",
    "polarity"          : "Overall sentiment (-1=negative, +1=positive)",
    "emotion_diversity" : "Whether both positive and negative tones are present",
    "emotion_intensity" : "Strength of overall emotion (0=weak, 1=strong)",
    "emotional_conflict": "Whether comment pulls in both emotional directions",
    "profanity_count"   : "Count of profane/offensive words",
    "allcaps_ratio"     : "Proportion of words written in ALL CAPS",
    "avg_token_length"  : "Average word length — proxy for language complexity",
}

print(f"\n--- Features Extracted ({len(feature_descriptions)}) ---")
for name, desc in feature_descriptions.items():
    print(f"  {name:<22} : {desc}")

print(f"\n--- Feature Stats by Disagreement Label ---")
feature_cols = list(feature_descriptions.keys())
print(df_final.groupby('disagreement_label')[feature_cols].mean().round(3).T.to_string())

print(f"\n--- All Columns in Output File ---")
for col in df_final.columns.tolist():
    print(f"  {col}")

print(f"\nDone! Shape: {df_final.shape}")
print("=" * 60)