import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt

# Load preprocessed audio and text features
audio_df = pd.read_csv('audio_data.csv')
text_df = pd.read_csv('text_data.csv')
print("✅ CSVs loaded.")
print(f"Audio DF shape: {audio_df.shape}, Text DF shape: {text_df.shape}")

# Merge on Participant_ID
merged_df = pd.merge(audio_df, text_df, on='Participant_ID', suffixes=('_audio', '_text'))
print(f"✅ Merged DF shape: {merged_df.shape}")

# Extract label
labels = merged_df['label'].values.astype(np.float32)
print(f"✅ Labels extracted. Shape: {labels.shape}, Unique: {np.unique(labels)}")

# Drop non-feature columns
non_feature_cols = ['Participant_ID', 'Participant_ID.1', 'label', 'depression_score']

# Audio features: all numeric features from audio CSV
audio_features = audio_df.drop(columns=non_feature_cols, errors='ignore')

# Text features: all numeric features from text CSV
text_features = text_df.drop(columns=non_feature_cols, errors='ignore')

# Combine them using merged_df for alignment
merged_df = pd.concat([audio_df, text_df.drop(columns=['Participant_ID'], errors='ignore')], axis=1)

# Get updated features after merge
audio_features = merged_df[audio_features.columns]
text_features = merged_df[text_features.columns]

# Standardize features
from sklearn.preprocessing import StandardScaler
scaler_audio = StandardScaler()
scaler_text = StandardScaler()
audio_features_scaled = scaler_audio.fit_transform(audio_features)
text_features_scaled = scaler_text.fit_transform(text_features)

# Combine audio and text features
X = np.hstack([audio_features_scaled, text_features_scaled])
y = labels

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
print("✅ Data split into training and test sets.")

# Apply SMOTE to the training set to handle class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print("✅ SMOTE applied. Resampled train size:", X_train_resampled.shape)

# Initialize XGBoost model
xgb_model = XGBClassifier()

# Define parameter grid for RandomizedSearchCV
param_dist = {
    'n_estimators': randint(100, 1000),  # Increase n_estimators range
    'max_depth': randint(3, 12),  # Increase max_depth range
    'learning_rate': uniform(0.001, 0.3),
    'subsample': uniform(0.5, 0.5),
    'colsample_bytree': uniform(0.5, 0.5)
}

# Initialize RandomizedSearchCV with more iterations
random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist, 
                                   n_iter=200, cv=3, verbose=2, n_jobs=-1, random_state=42)

# Fit RandomizedSearchCV
random_search.fit(X_train_resampled, y_train_resampled)

# Get the best XGBoost model
best_xgb_model = random_search.best_estimator_

# Evaluate the best model on training data
y_train_pred = best_xgb_model.predict(X_train_resampled)
train_accuracy = accuracy_score(y_train_resampled, y_train_pred)

# Evaluate the best model on test data
y_pred = best_xgb_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

# Print training and test accuracy
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Print best hyperparameters
print(f"Best Hyperparameters: {random_search.best_params_}")

# Plot feature importance
# import matplotlib.pyplot as plt
# xgb_model = best_xgb_model
# xgb.plot_importance(xgb_model, importance_type="weight")
# plt.show()
