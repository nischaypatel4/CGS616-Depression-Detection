import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv("audio_data.csv")  # replace with your actual file name

# Drop the Participant_ID column
df_features = df.drop(columns=['Participant_ID'])

# Compute the correlation matrix
correlation_matrix = df_features.corr()

# Print the correlation matrix
print(correlation_matrix)

# Optional: visualize it as a heatmap
plt.figure(figsize=(30, 25))
sns.heatmap(correlation_matrix, annot=True, fmt=".1f", cmap='coolwarm', square=True)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()
