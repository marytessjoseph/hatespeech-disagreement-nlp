import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from project_config import RAW_DATA, FILTERED_DATA

df = pd.read_csv(RAW_DATA)
print(f"Raw data loaded: {df.shape}")

relevant_columns = [
    'comment_id',
    'annotator_id',
    'text',
    'hate_speech_score',
    'hatespeech',
    'platform'
]

df_filtered = df[relevant_columns]

os.makedirs(os.path.dirname(FILTERED_DATA), exist_ok=True)
df_filtered.to_csv(FILTERED_DATA, index=False)

print(f"Relevant columns kept: {relevant_columns}")
print(f"Filtered data saved: {df_filtered.shape}")
