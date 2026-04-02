import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from project_config import FEATURES_FILE, MODEL_RESULTS

# --- Load data ---
df = pd.read_csv(FEATURES_FILE)

feature_cols = [
    'sent_length', 'negation_count', 'hedge_count', 'pronoun_count',
    'verb_count', 'uncertainty_count', 'contrast_count', 'sarcasm_markers',
    'intensity_count', 'exclamation_count', 'dep_depth', 'clause_count',
    'modal_count', 'type_token_ratio', 'is_question', 'polarity',
    'emotion_diversity', 'emotion_intensity', 'emotional_conflict',
    'profanity_count', 'allcaps_ratio', 'avg_token_length'
]

X = df[feature_cols]
y = df['disagreement_label'].map({'Low': 0, 'High': 1})

# --- Scale features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# --- Train Logistic Regression ---
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
cv     = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')

# --- Feature importance (coefficients) ---
coef_df = pd.DataFrame({
    'feature'    : feature_cols,
    'coefficient': model.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

# --- Save results ---
coef_df.to_csv(MODEL_RESULTS, index=False)

# --- Output ---
print("=" * 60)
print("LOGISTIC REGRESSION — MHS DATASET")
print("=" * 60)

print(f"\n--- Data ---")
print(f"Total comments : {len(df)}")
print(f"Train size     : {len(X_train)}")
print(f"Test size      : {len(X_test)}")
print(f"Features used  : {len(feature_cols)}")

print(f"\n--- Label Distribution ---")
counts = df['disagreement_label'].value_counts()
for label, count in counts.items():
    pct = count / len(df) * 100
    print(f"  {label:<6} : {count} ({pct:.1f}%)")

print(f"\n--- Model Performance ---")
print(f"  Test accuracy     : {acc:.4f}")
print(f"  Cross-val accuracy: {cv.mean():.4f} (+/- {cv.std():.4f})")

print(f"\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Low', 'High']))

print(f"\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)
print(f"               Predicted Low  Predicted High")
print(f"  Actual Low       {cm[0][0]:<10}     {cm[0][1]}")
print(f"  Actual High      {cm[1][0]:<10}     {cm[1][1]}")

# --- Split into High and Low disagreement triggers ---
high_triggers = coef_df[coef_df['coefficient'] > 0].sort_values('coefficient', ascending=False).head(10)
low_triggers  = coef_df[coef_df['coefficient'] < 0].sort_values('coefficient', ascending=True).head(10)

print(f"\n--- Top 10 Features That Trigger HIGH Disagreement ---")
print(f"  {'Rank':<6} {'Feature':<22} {'Coefficient'}")
print(f"  {'-'*45}")
for i, (_, row) in enumerate(high_triggers.iterrows(), 1):
    print(f"  {i:<6} {row['feature']:<22} {row['coefficient']:.4f}")

print(f"\n--- Top 10 Features That Trigger LOW Disagreement ---")
print(f"  {'Rank':<6} {'Feature':<22} {'Coefficient'}")
print(f"  {'-'*45}")
for i, (_, row) in enumerate(low_triggers.iterrows(), 1):
    print(f"  {i:<6} {row['feature']:<22} {row['coefficient']:.4f}")

print(f"\nResults saved to: hatespeech_model_results.csv")
print("=" * 60)