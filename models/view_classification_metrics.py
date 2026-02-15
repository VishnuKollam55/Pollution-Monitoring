import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, f1_score, accuracy_score

# Load dataset
df = pd.read_csv("../data/air_data.csv")

print("Columns:", df.columns)

# Features and class label
X = df[["pm25", "pm10", "co2"]]
y = df["level"]   # classification target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load trained classification model
with open("../models/air_level_model.pkl", "rb") as f:   # change name if needed
    clf = pickle.load(f)

# Predict classes
y_pred = clf.predict(X_test)

# Print all metrics
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Precision (macro):", precision_score(y_test, y_pred, average="macro"))
print("F1 Score (macro):", f1_score(y_test, y_pred, average="macro"))

print("\nDetailed Report:\n")
print(classification_report(y_test, y_pred))
