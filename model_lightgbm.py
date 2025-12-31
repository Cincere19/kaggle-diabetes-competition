import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

print("Loading data...")

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

target = "diagnosed_diabetes"

# ==========================
# Split features and label
# ==========================
X = train.drop(columns=[target, "id"])
y = train[target]

# ==========================
# Encode categoricals
# ==========================
cat_cols = X.select_dtypes(include="object").columns

for c in cat_cols:
    X[c] = X[c].astype("category").cat.codes
    test[c] = test[c].astype("category").cat.codes

# ==========================
# Handle missing and inf values
# ==========================
X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True))
test_X = test.drop(columns=["id"]).replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True))

# ==========================
# Train-validation split
# ==========================
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid)

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.02,
    "num_leaves": 64,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "verbose": -1
}

print("Training LightGBM with callbacks...")

model = lgb.train(
    params,
    train_data,
    valid_sets=[valid_data],
    num_boost_round=4000,
    callbacks=[
        lgb.early_stopping(stopping_rounds=200),
        lgb.log_evaluation(period=200),
    ]
)

print("Best iteration:", model.best_iteration)

# ==========================
# Train final model on full data
# ==========================
final_model = lgb.LGBMClassifier(
    **params,
    n_estimators=model.best_iteration
)

final_model.fit(X, y)

# ==========================
# Predict test
# ==========================
print("Predicting test...")

test_pred = final_model.predict_proba(test_X)[:, 1]

# ==========================
# Save submission
# ==========================
submission = pd.DataFrame({
    "id": test["id"],
    "diagnosed_diabetes": test_pred
})

submission.to_csv("submission.csv", index=False)

print("\nsubmission.csv created successfully")
