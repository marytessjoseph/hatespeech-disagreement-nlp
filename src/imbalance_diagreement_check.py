import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import matplotlib.pyplot as plt
from project_config import PROCESSED_DATA, PLOTS_PATH

os.makedirs(PLOTS_PATH, exist_ok=True)
df = pd.read_csv(PROCESSED_DATA)
total = len(df)

print("=" * 60)
print("DISAGREEMENT LABEL DISTRIBUTION")
print("=" * 60)

for label in ['Low', 'High']:
    count = (df['disagreement_label'] == label).sum()
    pct = count / total * 100
    bar = "█" * int(pct / 2)
    print(f"\nLabel    : {label} disagreement")
    print(f"Count    : {count}")
    print(f"Percent  : {pct:.1f}%  {bar}")

low_pct = (df['disagreement_label'] == 'Low').mean() * 100
high_pct = (df['disagreement_label'] == 'High').mean() * 100

print("\nVerdict:")
if abs(low_pct - high_pct) < 10:
    print("✅ Balanced — no action needed")
elif abs(low_pct - high_pct) < 20:
    print("⚠️  Slightly imbalanced — use class weights in model")
else:
    print("❌ Imbalanced — needs fixing before modelling")

plt.figure(figsize=(6, 4))
df['disagreement_label'].value_counts().plot(
    kind='bar',
    color=['steelblue', 'tomato']
)
plt.title("Disagreement Label Distribution")
plt.xlabel("Label")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_PATH, "disagreement_imbalance.png"))
plt.show()
