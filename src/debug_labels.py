import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from project_config import FILTERED_DATA, PROCESSED_DATA

# Check filtered data
df = pd.read_csv(FILTERED_DATA)
print("=== FILTERED DATA ===")
print("Shape:", df.shape)
print("Unique comments:", df['comment_id'].nunique())

# Check annotator counts
annotator_counts = df.groupby('comment_id')['annotator_id'].count()
print("\nAnnotator counts:")
print(annotator_counts.describe())
print("\nValue counts:")
print(annotator_counts.value_counts().sort_index().head(10))

# Check processed data
df2 = pd.read_csv(PROCESSED_DATA)
print("\n=== PROCESSED DATA ===")
print("Shape:", df2.shape)
print("\nColumns:", df2.columns.tolist())
print("\nFirst 5 rows:")
print(df2.head())
print("\nDisagreement score stats:")
print(df2['disagreement_score'].describe())
print("\nHow many zeros in disagreement_score:")
print((df2['disagreement_score'] == 0).sum())
print("\nUnique disagreement labels:")
print(df2['disagreement_label'].value_counts())