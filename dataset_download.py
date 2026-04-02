import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset
from project_config import RAW_DATA

load_dotenv()

print("Loading from HuggingFace...")
dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech")

print("Converting to dataframe...")
df = pd.DataFrame(dataset['train'])

os.makedirs(os.path.dirname(RAW_DATA), exist_ok=True)
df.to_csv(RAW_DATA, index=False)

print(f"Raw data saved")
print("Done!")
#print(f"Shape: {df.shape}")
#print(f"Columns: {df.columns.tolist()}")