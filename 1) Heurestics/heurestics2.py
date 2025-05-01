import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from itertools import product

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

        all_data.append(df[feature_cols])
        valid_participants.append(pid)

    except Exception as e:
        print(f"❌ Error loading {pid}: {e}")
        continue

if not all_data:
    raise ValueError("❌ No valid data files were loaded. Please check your file structure.")

combined_df = pd.concat(all_data, ignore_index=True)

# Function to evaluate model performance with focus on false negatives
def evaluate_model(results_df, method_name):
    if results_df.empty:
        return {
            'method': method_name,
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'false_negative_rate': 1.0,  # Worst possible
            'confusion_matrix': np.zeros((2, 2))
        }
    
    y_true = results_df['Actual_Label']
    y_pred = results_df['Heuristic_Label']
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    # Calculate false negative rate specifically
    if conf_matrix[1, 0] + conf_matrix[1, 1] > 0:
        false_negative_rate = conf_matrix[1, 0] / (conf_matrix[1, 0] + conf_matrix[1, 1])
    else:
        false_negative_rate = 0
    
    return {
        'method': method_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'false_negative_rate': false_negative_rate,
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
# APPROACH 3: Weighted Scoring System
# -------------------------------
def calculate_feature_weights_by_correlation():
    """Calculate feature weights based on correlation with depression labels"""
    feature_scores = {}
    depression_labels = []
    
    # Extract feature averages for each participant
    features_by_participant = {
        'AU15_r': [],  # sad mouth
        'AU01_r': [],  # inner brow raiser
        'AU04_r': [],  # brow lowerer
        'AU12_r': [],  # smile
        'AU45_r': [],  # blink
        'gaze_sum': []  # gaze angle
    }
    
    for pid in valid_participants:
        if pid not in labels_dict:
            continue
            
        depression_labels.append(labels_dict[pid])
        file_path = os.path.join(base_dir, f"{pid}_facial", f"{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv")
        
        try:
            df = pd.read_csv(file_path)
            if df.empty or set(feature_cols).difference(df.columns):
                continue
                
            # Calculate the average value for each feature
            features_by_participant['AU15_r'].append(df['AU15_r'].mean())
            features_by_participant['AU01_r'].append(df['AU01_r'].mean())
            features_by_participant['AU04_r'].append(df['AU04_r'].mean())
            features_by_participant['AU12_r'].append(df['AU12_r'].mean())
            features_by_participant['AU45_r'].append(df['AU45_r'].mean())
            features_by_participant['gaze_sum'].append((df['gaze_angle_x'].abs() + df['gaze_angle_y'].abs()).mean())
            
        except Exception as e:
            print(f"❌ Error processing {pid}: {e}")
            continue
    
    # Calculate correlations
    for feature, values in features_by_participant.items():
        if len(values) != len(depression_labels):
            continue
            
        correlation = np.corrcoef(values, depression_labels)[0, 1]
        
        # For AU12_r (smile), we expect negative correlation with depression
        # So we need to reverse the sign for proper weighting
        if feature == 'AU12_r':
            correlation = -correlation
            
        # For AU45_r (blink), we expect negative correlation with depression
        if feature == 'AU45_r':
            correlation = -correlation
            
        # For gaze_sum, we expect negative correlation (dull eyes)
        if feature == 'gaze_sum':
            correlation = -correlation
            
        # Store absolute value of correlation as weight
        feature_scores[feature] = abs(correlation)
    
    # Normalize weights to sum to 1
    total = sum(feature_scores.values())
    if total > 0:
        for feature in feature_scores:
            feature_scores[feature] /= total
    
    print("\nCalculated feature weights based on correlation:")
    for feature, weight in feature_scores.items():
        print(f"{feature}: {weight:.4f}")
    
    return feature_scores

# -------------------------------
# Approach 3: Weighted Scoring System with Grid Search
# -------------------------------
def approach_3_grid_search():
    """Try different combinations of weights and thresholds to minimize false negatives"""
    
    # Step 1: Get base weights from correlation analysis
    base_weights = calculate_feature_weights_by_correlation()
    
    # Step 2: Prepare weighted features for each participant
    all_results = []
    participant_features = {}
    
    print("\n🔍 Processing participant data for weighted scoring...")
    for pid in tqdm(valid_participants):
        if pid not in labels_dict:
            continue
            
        file_path = os.path.join(base_dir, f"{pid}_facial", f"{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv")
        
        try:
            df = pd.read_csv(file_path)
            if df.empty or set(feature_cols).difference(df.columns):
                continue
                
            # Pre-calculate weighted features
            weighted_features = {
                'sad_mouth': df['AU15_r'],
                'inner_brow': df['AU01_r'],
                'brow_lowerer': df['AU04_r'],
                'smile': -df['AU12_r'],  # Negative because less smile indicates depression
                'blink': -df['AU45_r'],  # Negative because less blinking indicates depression
                'gaze': -(df['gaze_angle_x'].abs() + df['gaze_angle_y'].abs()),  # Negative because dull eyes indicate depression
                'eyebrow_combined': df['AU01_r'] + df['AU04_r'],
                'sad_expression': df['AU15_r'] + df['AU01_r'] + df['AU04_r'] - df['AU12_r']
            }
            
            participant_features[pid] = {
                'features': weighted_features,
                'actual_label': int(labels_dict[pid])
            }
            
        except Exception as e:
            print(f"❌ Error processing {pid}: {e}")
            continue
    
    # Step 3: Grid search across different weighted combinations
    print("\n🔍 Running grid search for optimal weights and thresholds...")
    
    # Set up the grid search parameters
    weight_options = [0.5, 1.0, 1.5, 2.0]  # We'll multiply base weights by these factors
    frame_threshold_options = [0.25, 0.3, 0.35, 0.4]  # Thresholds for depressed frame ratio
    feature_score_threshold_options = [-0.2, -0.1, 0, 0.1, 0.2]  # Thresholds for feature scores
    
    best_config = None
    best_eval = {'false_negative_rate': float('inf')}
    
    total_combinations = (
        len(weight_options)**6 * 
        len(frame_threshold_options) * 
        len(feature_score_threshold_options)
    )
    print(f"Total combinations to try: {total_combinations}")
    
    # Track all configs with zero false negatives
    zero_fn_configs = []
    
    # Use a reduced set to make computation feasible
    weight_grid = list(product(
        weight_options[:2],  # sad_mouth
        weight_options[:2],  # inner_brow
        weight_options[:2],  # brow_lowerer
        weight_options[:2],  # smile
        weight_options[:2],  # blink
        weight_options[:2],  # gaze
    ))
    
    for weights in tqdm(weight_grid[:100]):  # Limit to first 100 combinations
        for frame_threshold in frame_threshold_options:
            for feature_score_threshold in feature_score_threshold_options:
                # Create weight dictionary
                weight_dict = {
                    'sad_mouth': weights[0] * base_weights.get('AU15_r', 1),
                    'inner_brow': weights[1] * base_weights.get('AU01_r', 1),
                    'brow_lowerer': weights[2] * base_weights.get('AU04_r', 1),
                    'smile': weights[3] * base_weights.get('AU12_r', 1),
                    'blink': weights[4] * base_weights.get('AU45_r', 1),
                    'gaze': weights[5] * base_weights.get('gaze_sum', 1),
                    'eyebrow_combined': (weights[1] + weights[2]) / 2 * (base_weights.get('AU01_r', 1) + base_weights.get('AU04_r', 1)) / 2,
                    'sad_expression': (weights[0] + weights[1] + weights[2] + weights[3]) / 4 * 
                                    (base_weights.get('AU15_r', 1) + base_weights.get('AU01_r', 1) + 
                                     base_weights.get('AU04_r', 1) + base_weights.get('AU12_r', 1)) / 4
                }
                
                # Normalize weights
                total_weight = sum(weight_dict.values())
                for key in weight_dict:
                    weight_dict[key] /= total_weight
                
                # Apply weights and thresholds to each participant
                results = []
                for pid, data in participant_features.items():
                    weighted_score = pd.Series(0.0, index=data['features']['sad_mouth'].index)
                    
                    # Apply weights to each feature
                    for feature_name, weight in weight_dict.items():
                        weighted_score += data['features'][feature_name] * weight
                    
                    # Determine if frame is indicative of depression
                    depressed_frames = weighted_score > feature_score_threshold
                    depressed_ratio = depressed_frames.mean()
                    
                    # Apply threshold to determine depression label
                    heuristic_label = int(depressed_ratio > frame_threshold)
                    
                    results.append({
                        "Participant": pid,
                        "Heuristic_Label": heuristic_label,
                        "Actual_Label": data['actual_label'],
                        "Depressed_Frame_Ratio": depressed_ratio
                    })
                
                # Evaluate this configuration
                results_df = pd.DataFrame(results)
                config_name = f"Weighted(SM:{weights[0]:.1f},IB:{weights[1]:.1f},BL:{weights[2]:.1f},S:{weights[3]:.1f},B:{weights[4]:.1f},G:{weights[5]:.1f})"
                eval_result = evaluate_model(results_df, config_name)
                
                # Check if this is the best configuration for reducing false negatives
                if eval_result['false_negative_rate'] < best_eval['false_negative_rate']:
                    best_eval = eval_result
                    best_config = {
                        'weights': weight_dict,
                        'frame_threshold': frame_threshold,
                        'feature_score_threshold': feature_score_threshold,
                        'results': results_df
                    }
                
                # Track configurations with zero false negatives
                if eval_result['false_negative_rate'] == 0:
                    zero_fn_configs.append({
                        'eval': eval_result,
                        'weights': weight_dict,
                        'frame_threshold': frame_threshold,
                        'feature_score_threshold': feature_score_threshold
                    })
    
    # Print results of configurations with zero false negatives
    print("\n===== CONFIGURATIONS WITH ZERO FALSE NEGATIVES =====")
    if zero_fn_configs:
        # Sort by accuracy (higher is better)
        zero_fn_configs.sort(key=lambda x: x['eval']['accuracy'], reverse=True)
        
        for i, config in enumerate(zero_fn_configs[:5]):  # Show top 5
            print(f"\nConfig #{i+1}:")
            print(f"  Method: {config['eval']['method']}")
            print(f"  Accuracy: {config['eval']['accuracy']:.4f}")
            print(f"  Precision: {config['eval']['precision']:.4f}")
            print(f"  Recall: {config['eval']['recall']:.4f}")
            print(f"  F1 Score: {config['eval']['f1']:.4f}")
            print(f"  Frame Threshold: {config['frame_threshold']}")
            print(f"  Feature Score Threshold: {config['feature_score_threshold']}")
            print("  Weights:")
            for feature, weight in config['weights'].items():
                print(f"    {feature}: {weight:.4f}")
            print("  Confusion Matrix:")
            print(f"    TN: {config['eval']['confusion_matrix'][0, 0]}, FP: {config['eval']['confusion_matrix'][0, 1]}")
            print(f"    FN: {config['eval']['confusion_matrix'][1, 0]}, TP: {config['eval']['confusion_matrix'][1, 1]}")
    else:
        print("No configurations achieved zero false negatives.")
    
    return best_config, zero_fn_configs

# -------------------------------
# Approach 4: Feature Ensemble with Adaptive Threshold
# -------------------------------
def approach_4_ensemble():
    """Create an ensemble of feature detectors with adaptive thresholding focused on reducing false negatives"""
    
    # We'll create multiple feature combinations and apply different thresholds
    # Then merge results with a bias towards positive detection to reduce false negatives
    
    # Step 1: Define quantile thresholds for each feature
    quantiles = {
        'AU15_r': [0.45, 0.50, 0.55],  # sad mouth
        'AU01_r': [0.45, 0.50, 0.55],  # inner brow
        'AU04_r': [0.45, 0.50, 0.55],  # brow lowerer
        'AU12_r': [0.25, 0.30, 0.35],  # smile (lower = less smile)
        'AU45_r': [0.25, 0.30, 0.35],  # blink (lower = less blink)
        'gaze_sum': [0.25, 0.30, 0.35]  # gaze (lower = dull eyes)
    }
    
    # Calculate actual threshold values based on quantiles
    thresholds = {}
    for feature, qtiles in quantiles.items():
        if feature == 'gaze_sum':
            values = (combined_df['gaze_angle_x'].abs() + combined_df['gaze_angle_y'].abs())
        else:
            values = combined_df[feature]
        
        thresholds[feature] = [values.quantile(q) for q in qtiles]
    
    # Step 2: Create multiple feature detector models
    print("\n🔍 Building feature ensemble with adaptive thresholds...")
    
    feature_detectors = [
        # Model 1: Sad mouth and eyebrows
        {
            'name': 'Sad Face Features',
            'conditions': [
                ('AU15_r', '>', 0),  # sad mouth
                ('AU01_r', '>', 0),  # inner brow
                ('AU04_r', '>', 0),  # brow lowerer
            ],
            'min_conditions': 2
        },
        # Model 2: Lack of positive expressions
        {
            'name': 'Lack of Positive',
            'conditions': [
                ('AU12_r', '<', 0),  # less smile
                ('AU45_r', '<', 0),  # less blink
                ('gaze_sum', '<', 0)  # dull eyes
            ],
            'min_conditions': 2
        },
        # Model 3: Combined facial expressions
        {
            'name': 'Combined Expression',
            'conditions': [
                ('AU15_r+AU01_r+AU04_r-AU12_r', '>', 0)  # sad expression
            ],
            'min_conditions': 1
        }
    ]
    
    # Track all results
    all_participant_results = {}
    
    # Apply all feature detectors with all threshold combinations
    for detector in feature_detectors:
        detector_name = detector['name']
        print(f"\nApplying detector: {detector_name}")
        
        # For each threshold level
        for threshold_level in range(3):  # 0=loose, 1=medium, 2=strict
            for pid in tqdm(valid_participants):
                if pid not in labels_dict:
                    continue
                    
                file_path = os.path.join(base_dir, f"{pid}_facial", f"{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv")
                
                try:
                    df = pd.read_csv(file_path)
                    if df.empty or set(feature_cols).difference(df.columns):
                        continue
                    
                    # Initialize conditions list
                    conditions_met = pd.Series(False, index=df.index)
                    
                    # Check each condition with the appropriate threshold
                    for condition in detector['conditions']:
                        feature, op, idx = condition
                        
                        # Check if this is a composite feature
                        if '+' in feature or '-' in feature:
                            # Split the expression
                            if '+' in feature and '-' in feature:
                                # Complex expression like AU15_r+AU01_r+AU04_r-AU12_r
                                parts = feature.replace('-', '+-').split('+')
                                expr = None
                                for part in parts:
                                    if part.startswith('-'):
                                        part = part[1:]  # Remove the minus sign
                                        if expr is None:
                                            expr = -df[part]
                                        else:
                                            expr -= df[part]
                                    else:
                                        if expr is None:
                                            expr = df[part]
                                        else:
                                            expr += df[part]
                                
                                # Use threshold_level=1 (medium) for composite features
                                threshold_value = combined_df[parts[0]].quantile(0.55)
                                
                                # Apply condition
                                if op == '>':
                                    condition_met = expr > threshold_value
                                else:
                                    condition_met = expr < threshold_value
                            else:
                                # Simple addition like AU01_r+AU04_r
                                parts = feature.split('+')
                                expr = df[parts[0]]
                                for part in parts[1:]:
                                    expr += df[part]
                                
                                # Use threshold_level=1 (medium) for composite features
                                threshold_value = combined_df[parts[0]].quantile(0.55)
                                
                                # Apply condition
                                if op == '>':
                                    condition_met = expr > threshold_value
                                else:
                                    condition_met = expr < threshold_value
                        else:
                            # Simple feature
                            if feature == 'gaze_sum':
                                values = df['gaze_angle_x'].abs() + df['gaze_angle_y'].abs()
                            else:
                                values = df[feature]
                            
                            threshold_value = thresholds[feature][threshold_level]
                            
                            # Apply condition
                            if op == '>':
                                condition_met = values > threshold_value
                            else:
                                condition_met = values < threshold_value
                        
                        # Update conditions met
                        conditions_met |= condition_met
                    
                    # Calculate depressed frame ratio
                    depressed_ratio = conditions_met.mean()
                    
                    # Use different frame_ratio thresholds based on sensitivity level
                    frame_thresholds = [0.25, 0.30, 0.35]  # Loose, medium, strict
                    
                    # Determine depression label (bias towards positive to reduce false negatives)
                    heuristic_label = int(depressed_ratio > frame_thresholds[threshold_level])
                    
                    # Store result
                    result_key = f"{detector_name}_Level{threshold_level}"
                    if pid not in all_participant_results:
                        all_participant_results[pid] = {
                            'Actual_Label': int(labels_dict[pid]),
                            'models': {}
                        }
                    all_participant_results[pid]['models'][result_key] = {
                        'label': heuristic_label,
                        'ratio': depressed_ratio
                    }
                    
                except Exception as e:
                    print(f"❌ Error processing {pid}: {e}")
                    continue
    
    # Step 3: Create ensemble results with different voting strategies
    voting_strategies = [
        # Even a single positive detection triggers a positive label
        {'name': 'Any-Positive', 'threshold': 1},
        # At least 25% of models must predict positive
        {'name': '25%-Positive', 'threshold': 0.25},
        # At least 50% of models must predict positive
        {'name': '50%-Positive', 'threshold': 0.5}
    ]
    
    ensemble_results = []
    
    for strategy in voting_strategies:
        results = []
        for pid, data in all_participant_results.items():
            # Collect votes
            votes = [model_data['label'] for model_data in data['models'].values()]
            
            # Calculate vote ratio
            if len(votes) > 0:
                positive_vote_ratio = sum(votes) / len(votes)
                ensemble_label = int(positive_vote_ratio >= strategy['threshold'])
                
                results.append({
                    "Participant": pid,
                    "Heuristic_Label": ensemble_label,
                    "Actual_Label": data['Actual_Label'],
                    "Positive_Vote_Ratio": positive_vote_ratio
                })
        
        # Evaluate ensemble
        results_df = pd.DataFrame(results)
        eval_result = evaluate_model(results_df, f"Ensemble-{strategy['name']}")
        
        ensemble_results.append({
            'strategy': strategy,
            'evaluation': eval_result,
            'results': results_df
        })
    
    # Print ensemble results
    print("\n===== ENSEMBLE MODEL EVALUATION =====")
    for ensemble in ensemble_results:
        eval_result = ensemble['evaluation']
        print(f"\n{eval_result['method']}:")
        print(f"  Accuracy:  {eval_result['accuracy']:.4f}")
        print(f"  Precision: {eval_result['precision']:.4f}")
        print(f"  Recall:    {eval_result['recall']:.4f}")
        print(f"  F1 Score:  {eval_result['f1']:.4f}")
        print(f"  False Negative Rate: {eval_result['false_negative_rate']:.4f}")
        print("  Confusion Matrix:")
        print(f"    TN: {eval_result['confusion_matrix'][0, 0]}, FP: {eval_result['confusion_matrix'][0, 1]}")
        print(f"    FN: {eval_result['confusion_matrix'][1, 0]}, TP: {eval_result['confusion_matrix'][1, 1]}")
    
    # Find the best ensemble (prioritizing minimizing false negatives)
    best_ensemble = min(ensemble_results, key=lambda x: x['evaluation']['false_negative_rate'])
    
    return ensemble_results, best_ensemble

# -------------------------------
# Run our new approaches that focus on minimizing false negatives
print("\n🔍 Running new Approach 3: Weighted Scoring with Grid Search...")
best_weighted_config, zero_fn_configs = approach_3_grid_search()

print("\n🔍 Running new Approach 4: Feature Ensemble with Adaptive Threshold...")
ensemble_results, best_ensemble = approach_4_ensemble()

# -------------------------------
# Compile and compare all results
# -------------------------------
evaluations = []

# New approaches
if best_weighted_config:
    evaluations.append(evaluate_model(best_weighted_config['results'], "New: Best Weighted Score"))

# Add best ensemble
if best_ensemble:
    evaluations.append(best_ensemble['evaluation'])

# Add configurations with zero false negatives
if zero_fn_configs:
    # Only add the best one (highest accuracy)
    best_zero_fn = zero_fn_configs[0]
    # We'd need to recompute results for this one, so we'll just add the evaluation
    evaluations.append(best_zero_fn['eval'])

# -------------------------------
# Print final comparative results
# -------------------------------
print("\n===== FINAL COMPARATIVE RESULTS =====")
for eval_result in evaluations:
    print(f"\n{eval_result['method']}:")
    print(f"  Accuracy:  {eval_result['accuracy']:.4f}")
    print(f"  Precision: {eval_result['precision']:.4f}")
    print(f"  Recall:    {eval_result['recall']:.4f}")
    print(f"  F1 Score:  {eval_result['f1']:.4f}")
    print(f"  False Negative Rate: {eval_result.get('false_negative_rate', 1-eval_result['recall']):.4f}")
    print("  Confusion Matrix:")
    print(f"    TN: {eval_result['confusion_matrix'][0, 0]}, FP: {eval_result['confusion_matrix'][0, 1]}")
    print(f"    FN: {eval_result['confusion_matrix'][1, 0]}, TP: {eval_result['confusion_matrix'][1, 1]}")

