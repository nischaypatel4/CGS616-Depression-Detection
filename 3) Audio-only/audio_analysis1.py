import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Define important features
important_features = [
    'Loudness_sma3', 'alphaRatio_sma3', 'hammarbergIndex_sma3',
    'slope0-500_sma3', 'spectralFlux_sma3', 'mfcc1_sma3', 'mfcc2_sma3',
    'mfcc3_sma3', 'mfcc4_sma3', 'F0semitoneFrom27.5Hz_sma3nz',
    'jitterLocal_sma3nz', 'shimmerLocaldB_sma3nz', 'HNRdBACF_sma3nz'
]
important_features1= [
    'pcm_fftMag_mfcc[1]', 'pcm_fftMag_mfcc[2]', 'pcm_fftMag_mfcc[3]', 'pcm_fftMag_mfcc[4]', 'pcm_fftMag_mfcc[5]',
    'pcm_fftMag_mfcc_de[1]', 'pcm_fftMag_mfcc_de[2]', 'pcm_fftMag_mfcc_de[3]', 'pcm_fftMag_mfcc_de[4]', 'pcm_fftMag_mfcc_de[5]',
    'pcm_fftMag_mfcc_de_de[1]', 'pcm_fftMag_mfcc_de_de[2]', 'pcm_fftMag_mfcc_de_de[3]', 'pcm_fftMag_mfcc_de_de[4]', 'pcm_fftMag_mfcc_de_de[5]'
]

# Path where your CSVs are stored
data_dir = "E:/ARSHIT/audio_features"

# Dictionary to store data
all_data = []

# Example depression label dictionary (replace with your actual PHQ labels)
# Format: {ParticipantID: 1 if PHQ >= 10 else 0}
depression_label = pd.read_csv("E:/ARSHIT/PHQ8_labels.csv")  # has columns ['Participant_ID', 'PHQ_8Total']
depression_label['label'] = (depression_label['PHQ_8Total'] >= 10).astype(int)
label_dict = dict(zip(depression_label['Participant_ID'], depression_label['label']))
participant_ids = depression_label['Participant_ID'].tolist()
# Process each participant file
for id in participant_ids:
        
        audio_file = f"E:/ARSHIT/processed_data/{id}_audio/{id}_OpenSMILE2.3.0_egemaps.csv"
        audio_file1= f"E:/ARSHIT/processed_data/{id}_audio/{id}_OpenSMILE2.3.0_mfcc.csv"
        print(audio_file) 
        if not os.path.exists(audio_file) or not os.path.exists(audio_file1) :
            print(f"Audio file for participant {id} not found. Skipping.")
            continue    
        try:
           
            df_raw = pd.read_csv(audio_file, header=None, dtype=str)
            df_raw1=pd.read_csv(audio_file1,header=None, dtype=str)
            # Split header and data
            header = df_raw.iloc[0, 0].split(';')
            df_data = df_raw.iloc[1:, 0].str.split(';', expand=True)
            df_data.columns = header
            df_data = df_data.iloc[:, 1:].astype(float)
            header1 = df_raw1.iloc[0, 0].split(';')
            df_data1 = df_raw1.iloc[1:, 0].str.split(';', expand=True)
            df_data1.columns = header1
            df_data1 = df_data1.iloc[:, 1:].astype(float)
            # Compute mean of selected features
            mean_features = df_data[important_features].mean().to_dict()
            mean_features1 = df_data1[important_features1].mean().to_dict()
            mean_features.update(mean_features1)
            mean_features['Participant_ID'] = int(id)
            mean_features['label'] = label_dict.get(int(id), None)
            
            if mean_features['label'] is not None:
                all_data.append(mean_features)
        except Exception as e:
            print(f"Error processing {id}: {e}")

# Convert to DataFrame
df_final = pd.DataFrame(all_data)
df_final.to_csv('E:/ARSHIT/audio_data.csv', index=False)

X = df_final.drop(columns=['Participant_ID', 'label'])
y = df_final['label']
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
# Train model
# clf = RandomForestClassifier(random_state=42)
clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
clf.fit(X_train_resampled, y_train_resampled)

# Evaluate
y_pred = clf.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
