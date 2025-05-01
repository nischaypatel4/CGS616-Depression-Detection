import os
import pandas as pd
from empath import Empath
from sklearn.feature_extraction.text import TfidfVectorizer
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()
# Path to folder where participant CSVs are stored
transcript_folder = "/path/to/transcript/files/"
phq8_file = "E:/ARSHIT/PHQ8_labels.csv"
df1 = pd.read_csv(phq8_file)

participant_ids = df1['Participant_ID'].tolist()
depression_score={}
for id in participant_ids:
    participant_row = df1[df1['Participant_ID'] == id]  # Access the row with this id
    if not participant_row.empty:
        depression_score[id] = (participant_row['PHQ_8Total'].values[0] >= 10).astype(int)
    else:
        depression_score[id] = None  # Or handle this case as you see fit
# Initialize the Empath lexicon
lexicon = Empath()

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=20, stop_words='english')

# This will store: {participant_id: full_transcript, features}
participant_data = {}

# Process each participant
for id in participant_ids:
    # Construct file path for participant's transcript
    transcript_file = f"E:/ARSHIT/processed_data/{id}_text/{id}_Transcript.csv"
    
    # Check if the transcript file exists
    if not os.path.exists(transcript_file):
        print(f"Transcript file for participant {id} not found. Skipping.")
        continue  # Skip to the next participant if the file does not exist
    
    # Try to read the participant's transcript CSV
    try:
        df = pd.read_csv(transcript_file)
        
        # Combine all text in the 'Text' column into one full transcript
        full_transcript = ' '.join(df['Text'].astype(str).tolist())
        
        # Get Empath features
        features = lexicon.analyze(full_transcript, categories=['sadness', 'positive_emotion', 'anger', 'death'], normalize=True)
        features['Participant_ID'] = id  # Add participant ID to features dictionary
        
        # Apply TF-IDF vectorization
        # X = vectorizer.fit_transform([full_transcript])  # Pass the transcript as a list
        # feature_names = vectorizer.get_feature_names_out()  # Get feature names
        
        # # Create a dictionary of feature names and their corresponding TF-IDF values
        # tfidf_values = X.toarray().flatten()  # Convert to dense array and flatten it
        # tfidf_dict = dict(zip(feature_names, tfidf_values))  # Create feature dictionary
        
        # # Add TF-IDF features to the features dictionary
        # for name in feature_names:
        #     features[name] = tfidf_dict.get(name, 0)  # Add the TF-IDF value, default to 0 if not found
        # readability_features = {
        #     'Participant_ID': id,
        #     'flesch': textstat.flesch_reading_ease(full_transcript),
        #     'fog': textstat.gunning_fog(full_transcript),
        #     'smog': textstat.smog_index(full_transcript),
        # }
        # features['flesch']=textstat.flesch_reading_ease(full_transcript)
        # features['fog']=textstat.gunning_fog(full_transcript)
        # features['smog']=textstat.smog_index(full_transcript)
        words = full_transcript.split()
        avg_word_len = sum(len(w) for w in words) / len(words)
        features['avg_word_len']=avg_word_len
        type_token_ratio = len(set(words)) / len(words)
        features['type_token_ratio']=type_token_ratio
        utterances = df['Text'].astype(str).tolist()
        avg_utt_len = sum(len(u.split()) for u in utterances) / len(utterances)
        features['avg_utt_len']=avg_utt_len
        sentiment = analyzer.polarity_scores(full_transcript)
        
        # Add sentiment scores to your features dictionary
        features['sentiment_neg'] = sentiment['neg']
        features['sentiment_neu'] = sentiment['neu']
        features['sentiment_pos'] = sentiment['pos']
        features['sentiment_compound'] = sentiment['compound']
        # Store the participant data
        features['depression_score']=depression_score[id]
        participant_data[id] = features
        
    except Exception as e:
        print(f"Error processing transcript for participant {id}: {e}")
        continue  # Skip to the next participant if there is any error

# Check the participant_data dictionary
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
# 1. Convert the participant_data dictionary to a DataFrame
df_features = pd.DataFrame.from_dict(participant_data, orient='index')
df_features.to_csv('E:/ARSHIT/text_data.csv', index=False)
# 2. Separate features and target
X = df_features.drop(columns=['Participant_ID', 'depression_score'])
y = df_features['depression_score']

# 3. Handle any missing values (optional: you can also impute)
X = X.fillna(0)

# 4. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42, stratify=y)
from sklearn.linear_model import LogisticRegression

# clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
# clf.fit(X_train, y_train)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train the classifier
# clf = RandomForestClassifier(random_state=42)
# clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
# clf.fit(X_train_resampled, y_train_resampled)
# # 6. Train a simple classifier
# # clf = RandomForestClassifier(random_state=42)
# # clf.fit(X_train, y_train)

# # 7. Evaluate
# y_pred = clf.predict(X_test)
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
#best model till  now
# Define the logistic regression model
logreg = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)

# Define the hyperparameters grid
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],          # Regularization strength
    'penalty': ['l1', 'l2'],               # Type of regularization
}

# Perform Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_resampled, y_train_resampled)

# Best model and parameters
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Evaluate on test set
y_pred = best_model.predict(X_test)
# from xgboost import XGBClassifier
# from sklearn.model_selection import GridSearchCV

# # Define the XGBoost classifier
# xgb_model = XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss', random_state=42)

# # Define the hyperparameters grid
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [3, 5, 7],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'subsample': [0.8, 1.0],
# }

# # Perform Grid Search with 5-fold cross-validation
# grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
# grid_search.fit(X_train_resampled, y_train_resampled)

# # Best model and parameters
# best_model = grid_search.best_estimator_
# print("Best Parameters:", grid_search.best_params_)

# # Evaluate on test set
# y_pred = best_model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))




