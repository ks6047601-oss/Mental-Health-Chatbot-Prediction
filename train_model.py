import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load your survey dataset
df = pd.read_csv("survey.csv")

# Drop unused or problematic columns
df = df.drop(columns=["Timestamp", "comments", "Country", "state"], errors="ignore")

# Separate target and features
y = df["treatment"]        # the column you are predicting
X = df.drop("treatment", axis=1)

# One-hot encode features
X_encoded = pd.get_dummies(X, drop_first=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train the model
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Save model and expected columns
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("expected_columns.pkl", "wb") as f:
    pickle.dump(list(X_encoded.columns), f)

print("Training complete. Files saved: model.pkl, expected_columns.pkl")
