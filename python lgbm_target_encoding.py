import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# ======================
# CONFIG
# ======================
TARGET = "diagnosed_diabetes"
ID_COL = "id"
N_FOLDS = 5
SEED = 42

np.random.seed(SEED)

print("Loading data...")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# ======================
# Identify column types
# ======================
categorical_cols = [
    "gender",
    "ethnicity",
    "education_level",
    "income_level",
    "smoking_status",
    "employment_status",
]

numeric_cols = [c for c in train.columns if c not in categorical_cols + [TARGET, ID_COL]]

# ======================
# Target Encoding (Leakage-safe)
# ======================
def add_target_encoding(train, test, col, target):
    global_mean = train[target].mean()

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    te_train = pd.Series(np.zeros(train.shape[0]), index=train.index)
    
    for tr_idx, val_idx in skf.split(train, train[target]):
        tr_data = train.iloc[tr_idx]
        val_data = train.iloc[val_idx]

        means = tr_data.groupby(col)[target].mean()

        # smoothing
        counts = tr_data.groupby(col)[target].count()
        smoothing = 1 / (1 + np.exp(-(counts - 5)))

        means_smooth = global_mean * (1 - smoothing) + means * smoothing

        te_train.iloc[val_idx] = val_data[col].map(means_smooth).fillna(global_mean)

    # Process test the same way
    means_full = train.groupby(col)[target].mean()
    counts_full = train.groupby(col)[target].count()
    smoothing_full = 1 / (1 + np.exp(-(counts_full - 5)))
    means_full_smooth = global_mean * (1 - smoothing_full) + means_full * smoothing_full

    te_test = test[col].map(means_full_smooth).fillna(global_mean)

    new_name = f"{col}_te"
    return te_train.values, te_test.values, new_name


print("Applying target encoding...")

for col in categorical_cols:
    te_tr, te_te, new_col = add_target_encoding(train, test, col, TARGET)
    train[new_col] = te_tr
    test[new_col] = te_te

# Remove original categorical columns
train = train.drop(columns=categorical_cols)
test = test.drop(columns=categorical_cols)

features = [c for c in train.columns if c not in [TARGET, ID_COL]]

# ======================
# LightGBM Model
# ======================
params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 4,
    "min_data_in_leaf": 80,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": SEED,
}

print("Training LightGBM with target-encoded features...")

kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof = np.zeros(train.shape[0])
test_preds = np.zeros(test.shape[0])

for fold, (tr_idx, val_idx) in enumerate(kf.split(train[features], train[TARGET])):
    print(f"Fold {fold+1}")

    X_tr = train.iloc[tr_idx][features]
    y_tr = train.iloc[tr_idx][TARGET]
    X_val = train.iloc[val_idx][features]
    y_val = train.iloc[val_idx][TARGET]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val)

    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dval],
        valid_names=["train", "valid"],
        num_boost_round=5000,
        callbacks=[lgb.early_stopping(200)],
    )

    oof[val_idx] = model.predict(X_val)

    test_preds += model.predict(test[features]) / N_FOLDS

auc = roc_auc_score(train[TARGET], oof)
print("===========================")
print(f"OOF AUC: {auc}")
print("===========================")

# ======================
# Create submission
# ======================
submission = pd.DataFrame({
    ID_COL: test[ID_COL],
    TARGET: test_preds
})

submission_file = "submission_lgbm_target_encoding.csv"
submission.to_csv(submission_file, index=False)

print(f"SUCCESS: {submission_file} created")
