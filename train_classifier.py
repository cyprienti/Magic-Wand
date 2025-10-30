import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load the CSV file
df = pd.read_csv('features/features.csv')
X = df.drop(columns=['label'])
y = df['label']

# Split the data into training sets(80%) and testing sets(20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(model, 'random_forest_model.pkl')
print("Model saved to random_forest_model.pkl")

print("Real :", list(y_test))
print("Predict :", list(y_pred))


# precision := how many predictions were correct in total
# recall := how many correct predictions were made
# f1-score := precision and recall combined