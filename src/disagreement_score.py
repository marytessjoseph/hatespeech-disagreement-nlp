import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from project_config import FILTERED_DATA, PROCESSED_DATA

df = pd.read_csv(FILTERED_DATA)
print(f"Loaded: {df.shape}")

# Step 1 - Remove duplicates
df = df.drop_duplicates(subset=['comment_id', 'annotator_id'], keep='first')
print(f"After dedup: {df.shape}")

# Step 2 - Count annotators per comment
annotator_counts = df.groupby('comment_id')['annotator_id'].count()

# Step 3 - Data driven thresholds
lower_threshold = int(annotator_counts.quantile(0.50))
Q1 = annotator_counts.quantile(0.25)
Q3 = annotator_counts.quantile(0.75)
IQR = Q3 - Q1
upper_cap = Q3 + 1.5 * IQR
print(f"\nData driven thresholds:")
print(f"Lower threshold : {lower_threshold} annotators")
print(f"Upper cap       : {upper_cap} annotators")

# Step 4 - Filter comments with enough annotators
valid_comments = annotator_counts[
    (annotator_counts >= lower_threshold) &
    (annotator_counts <= upper_cap)
].index
df_filtered = df[df['comment_id'].isin(valid_comments)]
print(f"\nComments after filter : {df_filtered['comment_id'].nunique()}")
print(f"Rows after filter     : {len(df_filtered)}")

# Step 5 - Normalized entropy function
def normalized_entropy(labels):
    labels = [l for l in labels if pd.notna(l)]
    n = len(labels)
    if n <= 1:
        return 0.0
    values, counts = np.unique(labels, return_counts=True)
    probabilities = counts / n
    raw_entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    n_categories = 3  # hatespeech labels: 0, 1, 2
    max_entropy = np.log2(n_categories)
    if max_entropy == 0:
        return 0.0
    return raw_entropy / max_entropy

# Step 6 - Compute disagreement score per comment
disagreement = df_filtered.groupby('comment_id').agg(
    text=('text', 'first'),
    n_annotators=('annotator_id', 'count'),
    mean_score=('hate_speech_score', 'mean'),
).reset_index()

entropy_scores = df_filtered.groupby('comment_id')['hatespeech'].apply(
    lambda x: normalized_entropy(x.tolist())
)
disagreement = disagreement.merge(
    entropy_scores.rename('disagreement_score'),
    on='comment_id'
)

# Clip floating point artifacts to 0
disagreement['disagreement_score'] = disagreement['disagreement_score'].clip(lower=0.0)

print(f"\nDisagreement score stats:")
print(disagreement['disagreement_score'].describe())
print(f"\nZero entropy comments: {(disagreement['disagreement_score'] == 0).sum()}")


# Step 7 - Categorize: any disagreement = High, full agreement = Low
disagreement['disagreement_label'] = disagreement['disagreement_score'].apply(
    lambda x: 'High' if x > 0.0 else 'Low'
)
print("\nLabel counts:")
print(disagreement['disagreement_label'].value_counts())


'''
# Step 7 - Categorize using median split
median = disagreement['disagreement_score'].median()
print(f"\nMedian disagreement score: {median:.4f}")
disagreement['disagreement_label'] = disagreement['disagreement_score'].apply(
    lambda x: 'High' if x > median else 'Low'
)
print("\nLabel counts:")
print(disagreement['disagreement_label'].value_counts())

'''



# Step 8 - Save
os.makedirs(os.path.dirname(PROCESSED_DATA), exist_ok=True)
disagreement.to_csv(PROCESSED_DATA, index=False)
print(f"\nSaved to: {PROCESSED_DATA}")