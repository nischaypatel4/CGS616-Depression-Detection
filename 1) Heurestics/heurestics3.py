import os
import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm
from sklearn.metrics import recall_score, confusion_matrix

# --- 1. Load labels ---
base_dir = "facial_features"
labels_df = pd.read_csv("detailed_lables.csv")
labels_df = labels_df[['Participant','Depression_label']].dropna()
labels_df['Participant'] = labels_df['Participant'].astype(int)
labels_dict = labels_df.set_index('Participant')['Depression_label'].to_dict()

# --- 2. Load & concatenate all AU data ---
feature_cols = ['AU15_r','AU01_r','AU04_r','AU12_r','AU45_r','gaze_angle_x','gaze_angle_y']
all_frames = []
valid_pids = []

print("Loading data...")
for pid in tqdm(labels_df['Participant']):
    path = os.path.join(base_dir, f"{pid}_facial", f"{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv")
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    if df.empty or not set(feature_cols).issubset(df.columns):
        continue
    df['Participant'] = pid
    all_frames.append(df[feature_cols+['Participant']])
    valid_pids.append(pid)

if not all_frames:
    raise RuntimeError("No valid data loaded!")
combined = pd.concat(all_frames, ignore_index=True)

# --- 3. Evaluation helper ---
def evaluate_by_recall(pred_df):
    y_true = pred_df['Actual_Label']
    y_pred = pred_df['Heuristic_Label']
    rec = recall_score(y_true, y_pred, zero_division=0)
    cm  = confusion_matrix(y_true, y_pred, labels=[0,1])
    return rec, cm

# --- 4. Grid of quantiles ---
q_grid = [0.35, 0.45, 0.55, 0.65]
keys   = ['AU15_r','AU01_r','AU04_r','AU12_r','AU45_r']
combos = list(product(q_grid, repeat=len(keys)))

# --- 5. Search ---
results = []
print("Searching over quantile combinations...")
for qs in tqdm(combos):
    # compute thresholds
    thr = {k: combined[k].quantile(q) for k,q in zip(keys, qs)}
    # simple “sad score” heuristic
    combined['Sad_Score'] = (
        (combined['AU15_r'] > thr['AU15_r']).astype(int) +
        (combined['AU01_r'] > thr['AU01_r']).astype(int) +
        (combined['AU04_r'] > thr['AU04_r']).astype(int) +
        (combined['AU12_r'] < thr['AU12_r']).astype(int) +
        (combined['AU45_r'] > thr['AU45_r']).astype(int)
    )
    # per-participant prediction: depressed if mean Sad_Score ≥ 3
    preds = (combined
             .groupby('Participant')['Sad_Score']
             .mean()
             .ge(3)
             .astype(int)
             .reset_index(name='Heuristic_Label'))
    preds['Actual_Label'] = preds['Participant'].map(labels_dict)
    preds = preds.dropna(subset=['Actual_Label'])
    
    rec, cm = evaluate_by_recall(preds)
    results.append({
        'quantiles': dict(zip(keys, qs)),
        'recall': rec,
        'conf_matrix': cm
    })

# --- 6. Print top 5 by recall ---
results.sort(key=lambda x: x['recall'], reverse=True)
print("\nTop 5 combinations by **Recall**:")
for rank, r in enumerate(results[:5], 1):
    qstr = ", ".join(f"{k}={v:.2f}" for k,v in r['quantiles'].items())
    print(f"\n{rank}. Quantiles: {qstr}")
    print(f"   → Recall = {r['recall']:.3f}")
    print(f"   → Confusion Matrix (TN, FP; FN, TP):\n{r['conf_matrix']}")
