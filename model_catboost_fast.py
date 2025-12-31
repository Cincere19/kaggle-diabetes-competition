import pandas as pd
from catboost import CatBoostClassifier

print("Loading data...")

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

TARGET = "diagnosed_diabetes"

# Drop id everywhere (important fix)
train = train.drop(columns=["id"])
test_ids = test["id"]
test = test.drop(columns=["id"])

# Split X and y
X = train.drop(columns=[TARGET])
y = train[TARGET]

# Identify categorical columns
cat_cols = X.select_dtypes(include="object").columns.tolist()

print("Training CatBoost (fast mode)...")

model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    od_type="Iter",
    od_wait=200,
    verbose=200,
    task_type="CPU"
)

model.fit(
    X,
    y,
    cat_features=cat_cols
)

print("Predicting...")

preds = model.predict_proba(test)[:, 1]

submission = pd.DataFrame({
    "id": test_ids,
    "diagnosed_diabetes": preds
})

submission.to_csv("submission_catboost_fast.csv", index=False)

print("\nSUCCESS: submission_catboost_fast.csv created")
