import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

print("Loading data...")

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

target = "diagnosed_diabetes"
print("Target column:", target)

# =========================
# Separate features and target
# =========================
X = train.drop(columns=[target, "id"])
y = train[target]

# =========================
# Handle categorical columns
# =========================
cat_cols = X.select_dtypes(include="object").columns

for c in cat_cols:
    X[c] = X[c].astype("category").cat.codes
    test[c] = test[c].astype("category").cat.codes

# =========================
# Fill missing values safely
# =========================
X = X.replace([np.inf, -np.inf], np.nan)
test_X = test.drop(columns=["id"]).replace([np.inf, -np.inf], np.nan)

X = X.fillna(X.median(numeric_only=True))
test_X = test_X.fillna(test_X.median(numeric_only=True))

# =========================
# Train simple strong baseline model
# =========================
print("Training RandomForest baseline model...")

model = RandomForestClassifier(
    n_estimators=600,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

# =========================
# Predict test probabilities
# =========================
print("Predicting test set...")

test_pred = model.predict_proba(test_X)[:, 1]

# =========================
# Create submission file
# =========================
submission = pd.DataFrame({
    "id": test["id"],
    "diagnosed_diabetes": test_pred
})

print(submission.head())
print(submission.describe())

submission.to_csv("submission.csv", index=False)

print("\nSubmission file saved as submission.csv\n")
print("Upload this file to Kaggle.")
