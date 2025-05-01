import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Directory setup
base_dir = "facial_features"
detailed_labels_path = "detailed_lables.csv"

# Load depression labels
labels_df = pd.read_csv(detailed_labels_path)
labels_df = labels_df[['Participant', 'Depression_label']].dropna()
labels_df['Participant'] = labels_df['Participant'].astype(int)
labels_dict = labels_df.set_index('Participant')['Depression_label'].to_dict()

# Step 1: Collect data to estimate thresholds
feature_cols = ['AU15_r', 'AU01_r', 'AU04_r', 'AU12_r', 'AU45_r', 'gaze_angle_x', 'gaze_angle_y']
all_data = []
valid_participants = []

print("\n🔍 Loading facial AU data for threshold estimation...")
for pid in tqdm(labels_df['Participant']):
    file_path = os.path.join(base_dir, f"{pid}_facial", f"{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv")

    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}")
        continue

    try:
        df = pd.read_csv(file_path)
        if df.empty:
            print(f"⚠️ Skipped {pid}: Empty file.")
            continue
        if not set(feature_cols).issubset(df.columns):
            print(f"⚠️ Skipped {pid}: Required columns missing.")
            continue

        print(f"✅ Successfully loaded {pid} — Preview:")
        print(df[feature_cols].head())
        all_data.append(df[feature_cols])
        valid_participants.append(pid)

    except Exception as e:
        print(f"❌ Error loading {pid}: {e}")
        continue

if not all_data:
    raise ValueError("❌ No valid data files were loaded. Please check your file structure.")

combined_df = pd.concat(all_data, ignore_index=True)

# Step 2: Compute dynamic thresholds from data
# Thresholds for Approach 1
thresholds_approach1 = {
    'AU15_r': combined_df['AU15_r'].quantile(0.45),  # sad mouth
    'AU01_r': combined_df['AU01_r'].quantile(0.45),  # tight eyebrows
    'AU04_r': combined_df['AU04_r'].quantile(0.45),
    'AU12_r': combined_df['AU12_r'].quantile(0.35),  # less smile
    'AU45_r': combined_df['AU45_r'].quantile(0.35),  # reduced blink
    'gaze_sum': (combined_df['gaze_angle_x'].abs() + combined_df['gaze_angle_y'].abs()).quantile(0.25),  # dull eyes
    'tear_thresh': (combined_df['AU01_r'] + combined_df['AU04_r'] + combined_df['AU45_r']).quantile(0.75),  # tears
    'sad_face_thresh': (combined_df['AU15_r'] + combined_df['AU01_r'] + combined_df['AU04_r'] - combined_df['AU12_r']).quantile(0.75)
}

# Thresholds for Approach 2 (inverse logic needs different percentiles)
thresholds_approach2 = {
    'AU15_r': combined_df['AU15_r'].quantile(0.50),  # moderate mouth depression
    'AU01_r': combined_df['AU01_r'].quantile(0.50),  # moderate eyebrow movement
    'AU04_r': combined_df['AU04_r'].quantile(0.50),  # moderate brow lowering
    'AU12_r': combined_df['AU12_r'].quantile(0.50),  # moderate smile
    'AU45_r': combined_df['AU45_r'].quantile(0.50),  # moderate blink
    'gaze_sum': (combined_df['gaze_angle_x'].abs() + combined_df['gaze_angle_y'].abs()).quantile(0.50),  # moderate gaze
    'tear_thresh': (combined_df['AU01_r'] + combined_df['AU04_r'] + combined_df['AU45_r']).quantile(0.50),  # moderate tear indicators
    'sad_face_thresh': (combined_df['AU15_r'] + combined_df['AU01_r'] + combined_df['AU04_r'] - combined_df['AU12_r']).quantile(0.50)  # moderate sad face
}

# Use thresholds_approach1 as default for backward compatibility
thresholds = thresholds_approach1

print("Computed thresholds for Approach 1:")
for key, value in thresholds_approach1.items():
    print(f"{key}: {value:.4f}")
    
print("\nComputed thresholds for Approach 2:")
for key, value in thresholds_approach2.items():
    print(f"{key}: {value:.4f}")

# Function to evaluate model performance
def evaluate_model(results_df, method_name):
    if results_df.empty:
        return {
            'method': method_name,
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'confusion_matrix': np.zeros((2, 2))
        }
    
    y_true = results_df['Actual_Label']
    y_pred = results_df['Heuristic_Label']
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    return {
        'method': method_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix
    }

# Function to plot confusion matrix
def plot_confusion_matrix(conf_matrix, title):
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    classes = ['Not Depressed (0)', 'Depressed (1)']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = conf_matrix.max() / 2
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return plt

# -------------------------------
# Approach 1: At least N conditions met
# -------------------------------
def approach_1(min_conditions=4):
    results = []
    
    for pid in tqdm(valid_participants):
        file_path = os.path.join(base_dir, f"{pid}_facial", f"{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv")
        
        try:
            df = pd.read_csv(file_path)
            if df.empty or set(feature_cols).difference(df.columns):
                continue
                
            # Individual condition checks using Approach 1 thresholds
            is_sad_mouth = df['AU15_r'] > thresholds_approach1['AU15_r']
            is_tight_eyebrows = (df['AU01_r'] > thresholds_approach1['AU01_r']) & (df['AU04_r'] > thresholds_approach1['AU04_r'])
            is_dull_eyes = (df['gaze_angle_x'].abs() + df['gaze_angle_y'].abs()) < thresholds_approach1['gaze_sum']
            is_low_blink = df['AU45_r'] < thresholds_approach1['AU45_r']
            is_tearful = (df['AU01_r'] + df['AU04_r'] + df['AU45_r']) > thresholds_approach1['tear_thresh']
            is_no_smile = df['AU12_r'] < thresholds_approach1['AU12_r']
            is_sad_expression = (df['AU15_r'] + df['AU01_r'] + df['AU04_r'] - df['AU12_r']) > thresholds_approach1['sad_face_thresh']
            
            # Count conditions met for each frame
            conditions = pd.DataFrame({
                'is_sad_mouth': is_sad_mouth,
                'is_tight_eyebrows': is_tight_eyebrows,
                'is_dull_eyes': is_dull_eyes,
                'is_low_blink': is_low_blink,
                'is_tearful': is_tearful,
                'is_no_smile': is_no_smile,
                'is_sad_expression': is_sad_expression
            })
            
            conditions_met = conditions.sum(axis=1)
            depressed_frames = conditions_met >= min_conditions
            depressed_ratio = depressed_frames.mean()
            
            # Apply threshold to determine depression label
            heuristic_label = int(depressed_ratio > 0.4)
            actual_label = labels_dict.get(pid, None)
            
            if actual_label is not None:
                results.append({
                    "Participant": pid,
                    "Heuristic_Label": heuristic_label,
                    "Actual_Label": int(actual_label),
                    "Depressed_Frame_Ratio": depressed_ratio,
                    "Avg_Conditions_Met": conditions_met.mean()
                })
                
        except Exception as e:
            print(f"❌ Error processing {pid}: {e}")
            continue
            
    return pd.DataFrame(results)

# -------------------------------
# Approach 2: Inverse Logic
# -------------------------------
def approach_2():
    results = []
    
    for pid in tqdm(valid_participants):
        file_path = os.path.join(base_dir, f"{pid}_facial", f"{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv")
        
        try:
            df = pd.read_csv(file_path)
            if df.empty or set(feature_cols).difference(df.columns):
                continue
                
            # Inverse logic - if below threshold, not depressed
            not_sad_mouth = df['AU15_r'] <= thresholds['AU15_r']
            not_tight_eyebrows = (df['AU01_r'] <= thresholds['AU01_r']) | (df['AU04_r'] <= thresholds['AU04_r'])
            not_dull_eyes = (df['gaze_angle_x'].abs() + df['gaze_angle_y'].abs()) >= thresholds['gaze_sum']
            not_low_blink = df['AU45_r'] >= thresholds['AU45_r']
            not_tearful = (df['AU01_r'] + df['AU04_r'] + df['AU45_r']) <= thresholds['tear_thresh']
            has_smile = df['AU12_r'] >= thresholds['AU12_r']
            not_sad_expression = (df['AU15_r'] + df['AU01_r'] + df['AU04_r'] - df['AU12_r']) <= thresholds['sad_face_thresh']
            
            # Non-depressed frame if ANY condition indicates not depressed
            non_depressed_frames = (
                not_sad_mouth | 
                not_tight_eyebrows | 
                not_dull_eyes | 
                not_low_blink | 
                not_tearful | 
                has_smile | 
                not_sad_expression
            )
            
            # Inverse the logic - depressed frames are those NOT classified as non-depressed
            depressed_frames = ~non_depressed_frames
            depressed_ratio = depressed_frames.mean()
            
            # Apply threshold to determine depression label
            heuristic_label = int(depressed_ratio > 0.3)  # Lower threshold since this is more strict
            actual_label = labels_dict.get(pid, None)
            
            if actual_label is not None:
                results.append({
                    "Participant": pid,
                    "Heuristic_Label": heuristic_label,
                    "Actual_Label": int(actual_label),
                    "Depressed_Frame_Ratio": depressed_ratio
                })
                
        except Exception as e:
            print(f"❌ Error processing {pid}: {e}")
            continue
            
    return pd.DataFrame(results)

# -------------------------------
# Execute both approaches
# -------------------------------
print("\n🔍 Running Approach 1: At least N conditions met...")
results_approach1_a = approach_1(min_conditions=2)  # At least 2 conditions
results_approach1_b = approach_1(min_conditions=3)  # At least 3 conditions
results_approach1_c = approach_1(min_conditions=4)  # At least 4 conditions

print("\n🔍 Running Approach 2: Inverse Logic...")
results_approach2 = approach_2()

# -------------------------------
# Evaluate all models
# -------------------------------
evaluations = []
evaluations.append(evaluate_model(results_approach1_a, "Approach 1 (≥2 conditions)"))
evaluations.append(evaluate_model(results_approach1_b, "Approach 1 (≥3 conditions)"))
evaluations.append(evaluate_model(results_approach1_c, "Approach 1 (≥4 conditions)"))
evaluations.append(evaluate_model(results_approach2, "Approach 2 (Inverse Logic)"))

# -------------------------------
# Print results
# -------------------------------
print("\n===== MODEL EVALUATION RESULTS =====")
for eval_result in evaluations:
    print(f"\n{eval_result['method']}:")
    print(f"  Accuracy:  {eval_result['accuracy']:.4f}")
    print(f"  Precision: {eval_result['precision']:.4f}")
    print(f"  Recall:    {eval_result['recall']:.4f}")
    print(f"  F1 Score:  {eval_result['f1']:.4f}")
    print("  Confusion Matrix:")
    print(f"    TN: {eval_result['confusion_matrix'][0, 0]}, FP: {eval_result['confusion_matrix'][0, 1]}")
    print(f"    FN: {eval_result['confusion_matrix'][1, 0]}, TP: {eval_result['confusion_matrix'][1, 1]}")

# -------------------------------
# Visualization
# -------------------------------
# Plot distribution of depressed frame ratios
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.hist(results_approach1_a['Depressed_Frame_Ratio'], bins=20, alpha=0.7, color='blue')
plt.title('Approach 1 (≥2 conditions): Depressed Frame Ratio')
plt.xlabel('Ratio of Frames')
plt.ylabel('Count')

plt.subplot(2, 2, 2)
plt.hist(results_approach1_b['Depressed_Frame_Ratio'], bins=20, alpha=0.7, color='green')
plt.title('Approach 1 (≥3 conditions): Depressed Frame Ratio')
plt.xlabel('Ratio of Frames')
plt.ylabel('Count')

plt.subplot(2, 2, 3)
plt.hist(results_approach1_c['Depressed_Frame_Ratio'], bins=20, alpha=0.7, color='purple')
plt.title('Approach 1 (≥4 conditions): Depressed Frame Ratio')
plt.xlabel('Ratio of Frames')
plt.ylabel('Count')

plt.subplot(2, 2, 4)
plt.hist(results_approach2['Depressed_Frame_Ratio'], bins=20, alpha=0.7, color='red')
plt.title('Approach 2 (Inverse Logic): Depressed Frame Ratio')
plt.xlabel('Ratio of Frames')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('depression_ratio_distributions.png')
plt.close()

# Plot confusion matrices
for i, eval_result in enumerate(evaluations):
    plt_cm = plot_confusion_matrix(eval_result['confusion_matrix'], f"Confusion Matrix: {eval_result['method']}")
    plt_cm.savefig(f"confusion_matrix_{i+1}.png")
    plt_cm.close()

# Plot comparison of accuracies
methods = [e['method'] for e in evaluations]
accuracies = [e['accuracy'] for e in evaluations]
f1_scores = [e['f1'] for e in evaluations]
recalls = [e['recall'] for e in evaluations]
precisions = [e['precision'] for e in evaluations]

plt.figure(figsize=(12, 8))
width = 0.2
x = np.arange(len(methods))
plt.bar(x - 0.3, accuracies, width, label='Accuracy', color='blue')
plt.bar(x - 0.1, precisions, width, label='Precision', color='red')
plt.bar(x + 0.1, recalls, width, label='Recall', color='green')
plt.bar(x + 0.3, f1_scores, width, label='F1 Score', color='purple')
plt.xlabel('Method')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(x, [m.split(':')[0] for m in methods], rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('model_performance_comparison.png')
plt.close()

# Save results to CSV
results_approach1_a.to_csv("results_approach1_2conditions.csv", index=False)
results_approach1_b.to_csv("results_approach1_3conditions.csv", index=False)
results_approach1_c.to_csv("results_approach1_4conditions.csv", index=False)
results_approach2.to_csv("results_approach2_inverse.csv", index=False)

# Create summary dataframe
summary_df = pd.DataFrame({
    'Method': methods,
    'Accuracy': accuracies,
    'Precision': precisions,
    'Recall': recalls,
    'F1_Score': f1_scores,
    'TN': [e['confusion_matrix'][0, 0] for e in evaluations],
    'FP': [e['confusion_matrix'][0, 1] for e in evaluations],
    'FN': [e['confusion_matrix'][1, 0] for e in evaluations],
    'TP': [e['confusion_matrix'][1, 1] for e in evaluations],
})

summary_df.to_csv("depression_detection_summary.csv", index=False)
print("\n✅ Done. Results saved to CSV files and visualizations created.")

# Display summary table
print("\n===== SUMMARY TABLE =====")
print(summary_df.to_string(index=False))