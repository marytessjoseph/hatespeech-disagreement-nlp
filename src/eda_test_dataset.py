import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from project_config import FILTERED_DATA, PLOTS_PATH

os.makedirs(PLOTS_PATH, exist_ok=True)
df = pd.read_csv(FILTERED_DATA)

print("Shape:", df.shape)
print("\nColumn types:")
print(df.dtypes)
print("\nNull values:")
print(df.isnull().sum())
print("\nTotal rows:", len(df))
print("Unique comments:", df['comment_id'].nunique())
print("Unique annotators:", df['annotator_id'].nunique())

annotators_per_comment = df.groupby('comment_id')['annotator_id'].count()
print("\nAnnotators per comment:")
print(annotators_per_comment.describe())

plt.figure(figsize=(8, 4))
annotators_per_comment.value_counts().sort_index().plot(kind='bar')
plt.title("How many annotators rated each comment?")
plt.xlabel("Number of annotators")
plt.ylabel("Number of comments")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, "annotators_per_comment.png"))
plt.show()

print("\nhate_speech_score stats:")
print(df['hate_speech_score'].describe())

plt.figure(figsize=(8, 4))
sns.histplot(df['hate_speech_score'], bins=50, kde=True)
plt.title("Distribution of hate_speech_score")
plt.xlabel("hate_speech_score")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, "hate_speech_score_dist.png"))
plt.show()

print("\nPlatform distribution:")
print(df['platform'].value_counts())

df['text_length'] = df['text'].apply(lambda x: len(str(x).split()))
print("\nText length stats:")
print(df['text_length'].describe())

plt.figure(figsize=(8, 4))
sns.histplot(df['text_length'], bins=50, kde=True)
plt.title("Distribution of comment length (word count)")
plt.xlabel("Word count")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, "text_length_dist.png"))
plt.show()