import pandas as pd
import os

def extract_features(df):
    features = {}
    for col in ['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ']:
        features[f'{col}_mean'] = df[col].mean()
        features[f'{col}_std'] = df[col].std()
        features[f'{col}_min'] = df[col].min()
        features[f'{col}_max'] = df[col].max()
    return features


data_dir = 'recordings'
all_features = []
output_dir = 'features'
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(data_dir):
    if file.endswith('.csv'):
        # Load the CSV file to extract features
        df = pd.read_csv(os.path.join(data_dir, file), delimiter=';')
        features = extract_features(df)

        # Add the label
        if 'Stone' in file:
            features['label'] = 'stone'
        elif 'Paper' in file:
            features['label'] = 'paper'
        elif 'Scissors' in file:
            features['label'] = 'scissors'
        else:
            continue

        all_features.append(features)


df_all = pd.DataFrame(all_features)
# Save the DataFrame to a CSV file
df_all.to_csv(os.path.join(output_dir, 'features.csv'), index=False)
print("Features saved to features.csv")

