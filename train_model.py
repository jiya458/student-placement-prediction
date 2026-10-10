
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("student_placement.csv")

# Input features
X = df[
    [
        "cgpa",
        "aptitude_score",
        "communication_skill",
        "internship",
        "projects"
    ]
]

# Target output
y = df["placement_status"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Create and train the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

print("Student Placement Prediction Model")
print("----------------------------------")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Save the trained model
joblib.dump(model, "placement_model.pkl")

print("\nModel saved successfully as placement_model.pkl")