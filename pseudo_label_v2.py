import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

print("Loading data...")

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# -----------------------------
# Detect target column safely
# -----------------------------
for possible_target in ["diagnosed_diabetes", "diabetes", "target"]:
    if possible_target in train.columns:
        target_col = possible_target
        break
else:
    raise ValueError("Target column not found in training data")

print(f"Using target column: {target_col}")

# -----------------------------
# Remove ID column safely
# -----------------------------
id_col = None
for c in ["id", "ID", "Id"]:
    if c in train.columns:
        id_col = c

if id_col:
    test_ids = test[id_col]
    train = train.drop(columns=[id_col])
    test = test.drop(columns=[id_col])
else:
    test_ids = None

# -----------------------------
# Separate features & labels
# -----------------------------
y = train[target_col]
X = train.drop(columns=[target_col])
X_test = test.copy()

# -----------------------------
# Handle categorical columns
# -----------------------------
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

print("Encoding categoricals (Label Encoding style)...")

for col in cat_cols:
    # combine to avoid unseen label problems
    full = pd.concat([X[col], X_test[col]], axis=0)

    mapping = {v: i for i, v in enumerate(full.astype(str).unique())}
    X[col] = X[col].astype(str).map(mapping)
    X_test[col] = X_test[col].astype(str).map(mapping)

# confirm numeric
assert all([np.issubdtype(dt, np.number) for dt in X.dtypes]), "Non-numeric dtypes remain"

# -----------------------------
# Train/validation split
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

dtrain = lgb.Dataset(X_train, label=y_train)
dvalid = lgb.Dataset(X_val, label=y_val)

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 64,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.9,
    "bagging_freq": 3,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "verbose": -1,
}

print("Training base model...")

base_model = lgb.train(
    params,
    dtrain,
    valid_sets=[dvalid],
    num_boost_round=1500,
    callbacks=[lgb.early_stopping(200)]
)

# -----------------------------
# Predict test set
# -----------------------------
test_pred = base_model.predict(X_test)

# -----------------------------
# Create pseudo-labels
# -----------------------------
high_conf_mask = (test_pred > 0.95) | (test_pred < 0.05)

pseudo = X_test.loc[high_conf_mask].copy()
pseudo[target_col] = (test_pred[high_conf_mask] > 0.5).astype(int)

print(f"Pseudo-labeled rows added: {len(pseudo)}")

# -----------------------------
# Build augmented dataset
# -----------------------------
train_aug = pd.concat([X,], axis=0)
label_aug = pd.concat([y,], axis=0)

if len(pseudo) > 0:
    train_aug = pd.concat([train_aug, pseudo.drop(columns=[target_col])])
    label_aug = pd.concat([label_aug, pseudo[target_col]])

# -----------------------------
# Train FINAL model
# -----------------------------
dfinal = lgb.Dataset(train_aug, label=label_aug)

print("Training final model on augmented data...")

final_model = lgb.train(
    params,
    dfinal,
    num_boost_round=2000
)

# -----------------------------
# Predict final submission
# -----------------------------
final_pred = final_model.predict(X_test)

sub = pd.DataFrame({
    "id": test_ids if test_ids is not None else np.arange(len(final_pred)),
    target_col: final_pred
})

sub.to_csv("submission_pseudo_v2.csv", index=False)

print("SUCCESS: submission_pseudo_v2.csv created")
