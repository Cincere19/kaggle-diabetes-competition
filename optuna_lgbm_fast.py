import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

TARGET = "diagnosed_diabetes"

print("Loading data...")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# =====================================
# Drop ID consistently (IMPORTANT FIX)
# =====================================
train_id = train["id"]
test_id = test["id"]

# we will NOT use id as a feature
train = train.drop(columns=["id"])
test = test.drop(columns=["id"])

# =====================================
# Encode categorical features
# =====================================
cat_cols = [c for c in train.columns if train[c].dtype == "object"]

for c in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[c], test[c]]).astype(str)
    le.fit(combined)
    train[c] = le.transform(train[c].astype(str))
    test[c] = le.transform(test[c].astype(str))

# =====================================
# Fill missing values
# =====================================
train = train.fillna(-999)
test = test.fillna(-999)

# =====================================
# Split features and target
# =====================================
X = train.drop(columns=[TARGET])
y = train[TARGET]

X_test = test.copy()

print(f"Training feature count: {X.shape[1]}")
print(f"Test feature count:     {X_test.shape[1]}")


# =====================================
# OPTUNA OBJECTIVE (FAST)
# =====================================
def objective(trial):

    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_threads": 6,

        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.08),
        "num_leaves": trial.suggest_int("num_leaves", 32, 128),
        "max_depth": trial.suggest_int("max_depth", 4, 9),

        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 30, 200),

        "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 5),

        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 3.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 6.0),
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof = np.zeros(len(train))

    for tr_idx, va_idx in skf.split(X, y):

        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        dtrain = lgb.Dataset(X_tr, y_tr)
        dvalid = lgb.Dataset(X_va, y_va)

        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dvalid],
            num_boost_round=2000,
            callbacks=[lgb.early_stopping(80)],
        )

        preds = model.predict(X_va)
        oof[va_idx] = preds

        trial.report(roc_auc_score(y_va, preds), step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return roc_auc_score(y, oof)


print("Starting FAST Optuna tuning...")

study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner()
)

study.optimize(objective, n_trials=15, show_progress_bar=True)

print("\nBest AUC:", study.best_value)
print("Best params:", study.best_params)

# =====================================
# Train FINAL model on all data
# =====================================
best_params = study.best_params
best_params.update({
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "num_threads": 6
})

final_model = lgb.LGBMClassifier(
    **best_params,
    n_estimators=1800
)

final_model.fit(X, y)

test_pred = final_model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    "id": test_id,
    "diagnosed_diabetes": test_pred
})

submission.to_csv("submission_optuna_fast.csv", index=False)

print("\nSUCCESS: submission_optuna_fast.csv created")
