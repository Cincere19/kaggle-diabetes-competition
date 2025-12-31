import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

TARGET = "diagnosed_diabetes"

# -----------------------------
# Drop id from BOTH train & test
# -----------------------------
train_id = train["id"]
test_id = test["id"]

train = train.drop(columns=["id"])
test = test.drop(columns=["id"])

# -----------------------------
# Label encode categoricals
# -----------------------------
cat_cols = train.select_dtypes(include="object").columns

for col in cat_cols:
    le = LabelEncoder()
    full_col = pd.concat([train[col], test[col]], axis=0)
    le.fit(full_col.astype(str))

    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# -----------------------------
# Train/validation split
# -----------------------------
X = train.drop(columns=[TARGET])
y = train[TARGET]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)
dtest = xgb.DMatrix(test)

# -----------------------------
# FAST XGBoost params
# -----------------------------
params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "eta": 0.05,
    "max_depth": 5,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "lambda": 2.0,
    "alpha": 0.0,
    "tree_method": "hist",
}

watchlist = [(dtrain, "train"), (dvalid, "valid")]

model = xgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    evals=watchlist,
    early_stopping_rounds=100,
    verbose_eval=200,
)

# -----------------------------
# Validation AUC
# -----------------------------
valid_pred = model.predict(dvalid)
auc = roc_auc_score(y_valid, valid_pred)
print("Validation AUC:", auc)

# -----------------------------
# Predict test set
# -----------------------------
test_pred = model.predict(dtest)

# -----------------------------
# Save submission
# -----------------------------
submission = pd.DataFrame({
    "id": test_id,
    "diagnosed_diabetes": test_pred
})

submission.to_csv("submission_xgboost_fast.csv", index=False)

print("SUCCESS: submission_xgboost_fast.csv created")
