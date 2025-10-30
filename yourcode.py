# IMPORTANT: Please make all your changes to this file
import joblib
import pandas as pd
from random import randrange

spellname1 = "stone"
spellname2 = "paper"
spellname3 = "scissors"

# Load the model once
model = joblib.load('random_forest_model.pkl')

def extract_features(df):
    features = {}
    for col in ['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ']:
        features[f'{col}_mean'] = df[col].mean()
        features[f'{col}_std'] = df[col].std()
        features[f'{col}_min'] = df[col].min()
        features[f'{col}_max'] = df[col].max()
    return pd.DataFrame([features])

def process_spell(pandas_df: pd.DataFrame):
    # give a number for predicted spell (1 to 3; with 1 wins over 2, 2 wins over 3, and 3 wins over 1)
    predicted_spell = randrange(3) + 1

    print("Incoming Dataframe:")
    print(pandas_df.head(5))

    # TODO use your preprocessing from your Jupyter Notebook here and load your model
    #
    # look here for how to save and load a model:
    # https://www.geeksforgeeks.org/saving-a-machine-learning-model/

    X = extract_features(pandas_df)

    prediction = model.predict(X)[0]

    label_to_id = {'stone': 1, 'paper': 2, 'scissors': 3}

    spell_id = label_to_id.get(prediction, 0)

    return (spell_id, get_spellname(spell_id))

def get_spellname(id):
    if id == 1:
        return spellname1
    if id == 2:
        return spellname2
    if id == 3:
        return spellname3
    return "Unknown Spell"


if __name__ == "__main__":
    df = pd.read_csv('recordings/New_Gesture-20250513-001234.csv', sep=';')
    result = process_spell(df)
    print("🔮 Prediction :", result)