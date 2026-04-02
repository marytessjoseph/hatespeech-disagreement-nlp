import os
from dotenv import load_dotenv

load_dotenv()

BASE_PATH = os.getenv("BASE_PATH", "")

# Raw data
RAW_DATA        = os.path.join(BASE_PATH, "data", "raw", "measuring_hate_speech_raw.csv")
FILTERED_DATA   = os.path.join(BASE_PATH, "data", "raw", "hatespeech_raw.csv")

# Processed data
PROCESSED_DATA  = os.path.join(BASE_PATH, "data", "processed", "hatespeech_disagreement.csv")

#features extracted data
FEATURES_FILE   = os.path.join(BASE_PATH, "data", "processed", "hatespeech_features.csv")

#modelled result
MODEL_RESULTS  = os.path.join(BASE_PATH, "data", "processed", "hatespeech_model_results.csv")

# Plots
PLOTS_PATH      = os.path.join(BASE_PATH, "plots")