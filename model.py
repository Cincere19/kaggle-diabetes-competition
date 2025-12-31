import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
from itertools import combinations


# ============================
# LOAD DATA
# ============================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")

target = "diagnosed_diabetes"
print("Target column:", target)


# ============================
# LABEL ENCODE CATEGORICALS
# ============================
cat_cols = train.select_dtypes(include=["object"]).columns.tolist()

for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))


# ============================
# MEMORY REDUCTION
# ============================
def reduce_memory(df):
    for col in df.columns:
        if df[col].dtype == object:
            continue

        c_min = df[col].min()
        c_max = df[col].max()

        if str(df[col].dtype)[:3] == "int":
            if c_min > -128 and c_max < 127:
                df[col] = df[col].astype(np.int8)
            elif c_min > -32768 and c_max < 32767:
                df[col] = df[col].astype(np.int16)
            else:
                df[col] = df[col].astype(np.int32)
        else:
            df[col] = df[col].astype(np.float32)

    return df


train = reduce_memory(train)
test = reduce_memory(test)


# ============================
# DOMAIN FEATURE ENGINEERING
# ============================
def domain_features(df):

    if {"systolic_bp", "diastolic_bp"}.issubset(df.columns):
        df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
        df["map_bp"] = df["diastolic_bp"] + df["pulse_pressure"] / 3

    if {"cholesterol_total", "hdl_cholesterol"}.issubset(df.columns):
        df["chol_hdl_ratio"] = df["cholesterol_total"] / (df["hdl_cholesterol"] + 1e-5)

    if {"ldl_cholesterol", "hdl_cholesterol"}.issubset(df.columns):
        df["ldl_hdl_ratio"] = df["ldl_cholesterol"] / (df["hdl_cholesterol"] + 1e-5)

    if {"triglycerides", "hdl_cholesterol"}.issubset(df.columns):
        df["tg_hdl_ratio"] = df["triglycerides"] / (df["hdl_cholesterol"] + 1e-5)

    if {"bmi", "age"}.issubset(df.columns):
        df["bmi_age"] = df["bmi"] * df["age"]

    if {"bmi", "waist_to_hip_ratio"}.issubset(df.columns):
        df["bmi_whr"] = df["bmi"] * df["waist_to_hip_ratio"]

    return df


train = domain_features(train)
test = domain_features(test)


# ============================
# AUTO FEATURE GENERATION
# ============================
def auto_features(df):

    num_cols = df.select_dtypes(include=['float32', 'float64', 'int32', 'int64']).columns

    # multiplicative interactions
    for c1, c2 in combinations(num_cols, 2):
        df[f"{c1}_times_{c2}"] = df[c1] * df[c2]

    # log transforms
    for c in num_cols:
        df[f"log_{c}"] = np.log1p(np.abs(df[c]))

    return df


train_use = train.drop(columns=["id"])
test_use = test.drop(columns=["id"])

train_use = auto_features(train_use)
test_use = auto_features(test_use)

train_use = reduce_memory(train_use)
test_use = reduce_memory(test_use)


# ============================
# HANDLE NaN / INF (STRONG GUARANTEE)
# ============================
train_use = train_use.replace([np.inf, -np.inf], np.nan)
test_use = test_use.replace([np.inf, -np.inf], np.nan)

# fill by median
train_use = train_use.fillna(train_use.median(numeric_only=True))
test_use = test_use.fillna(test_use.median(numeric_only=True))

# fallback fill
train_use = train_use.fillna(0)
test_use = test_use.fillna(0)


# ============================
# SPLIT
# ============================
X = train_use.drop(columns=[target])
y = train_use[target]
X_test = test_use.copy()

# FINAL guaranteed cleanup (no NaN / inf anywhere)
X = pd.DataFrame(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), columns=X.columns)
X_test = pd.DataFrame(np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0), columns=X_test.columns)


# ============================
# ADVERSARIAL VALIDATION (NaN-tolerant)
# ============================
adv_train = X.copy()
adv_train["label"] = 0

adv_test = X_test.copy()
adv_test["label"] = 1

adv = pd.concat([adv_train, adv_test])
y_adv = adv["label"]
X_adv = adv.drop(columns=["label"])

skf_adv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_adv = np.zeros(len(adv))

for tr, va in skf_adv.split(X_adv, y_adv):
    X_tr, X_va = X_adv.iloc[tr], X_adv.iloc[va]
    y_tr, y_va = y_adv.iloc[tr], y_adv.iloc[va]

    clf = HistGradientBoostingClassifier(max_depth=6)
    clf.fit(X_tr, y_tr)

    oof_adv[va] = clf.predict_proba(X_va)[:, 1]

adv_auc = roc_auc_score(y_adv, oof_adv)
print("Adversarial validation AUC:", adv_auc)


# ============================
# OPTUNA TUNING FOR LIGHTGBM
# ============================
def objective(trial):

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.03),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 3.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 3.0),
        "verbose": -1
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof = np.zeros(len(X))

    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_va, label=y_va)

        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dvalid],
            num_boost_round=2000,
            callbacks=[lgb.early_stopping(100)],
        )

        oof[va_idx] = model.predict(X_va)

    return roc_auc_score(y, oof)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

best_lgb_params = study.best_params
best_lgb_params["objective"] = "binary"
best_lgb_params["metric"] = "auc"
best_lgb_params["verbose"] = -1

print("Best LGBM params:", best_lgb_params)


# ============================
# BASE MODELS
# ============================
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

pred_lgb = np.zeros(len(X_test))
pred_xgb = np.zeros(len(X_test))
pred_cat = np.zeros(len(X_test))


# ---------- LightGBM ----------
for tr, va in skf.split(X, y):
    X_tr, X_va = X.iloc[tr], X.iloc[va]
    y_tr, y_va = y.iloc[tr], y.iloc[va]

    model = lgb.LGBMClassifier(**best_lgb_params, n_estimators=4000)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    oof_lgb[va] = model.predict_proba(X_va)[:, 1]
    pred_lgb += model.predict_proba(X_test)[:, 1] / skf.n_splits


# ---------- XGBoost ----------
for tr, va in skf.split(X, y):
    X_tr, X_va = X.iloc[tr], X.iloc[va]
    y_tr, y_va = y.iloc[tr], y.iloc[va]

    model = xgb.XGBClassifier(
        n_estimators=4000,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="auc",
        tree_method="hist"
    )

    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    oof_xgb[va] = model.predict_proba(X_va)[:, 1]
    pred_xgb += model.predict_proba(X_test)[:, 1] / skf.n_splits


# ---------- CatBoost ----------
for tr, va in skf.split(X, y):
    X_tr, X_va = X.iloc[tr], X.iloc[va]
    y_tr, y_va = y.iloc[tr], y.iloc[va]

    model = CatBoostClassifier(
        iterations=4000,
        learning_rate=0.01,
        depth=6,
        eval_metric="AUC",
        verbose=False
    )

    model.fit(X_tr, y_tr, eval_set=(X_va, y_va))

    oof_cat[va] = model.predict_proba(X_va)[:, 1]
    pred_cat += model.predict_proba(X_test)[:, 1] / skf.n_splits


print("LightGBM AUC:", roc_auc_score(y, oof_lgb))
print("XGBoost AUC:", roc_auc_score(y, oof_xgb))
print("CatBoost AUC:", roc_auc_score(y, oof_cat))


# ============================
# STACKING META-MODEL (NaN tolerant)
# ============================
stack_train = np.vstack([oof_lgb, oof_xgb, oof_cat]).T
stack_test = np.vstack([pred_lgb, pred_xgb, pred_cat]).T

meta = HistGradientBoostingClassifier(max_depth=6)
meta.fit(stack_train, y)
meta_pred = meta.predict_proba(stack_test)[:, 1]


# ============================
# PSEUDO-LABELING
# ============================
pseudo = test.copy()
pseudo[target] = meta_pred

high_conf = pseudo[(pseudo[target] > 0.99) | (pseudo[target] < 0.01)]
print("Pseudo-labels added:", len(high_conf))

augmented = pd.concat([train, high_conf])

aug_y = augmented[target]
aug_X = augmented.drop(columns=[target, "id"])

final_model = lgb.LGBMClassifier(**best_lgb_params, n_estimators=5000)
final_model.fit(aug_X, aug_y)

final_pred = final_model.predict_proba(test_use.drop(columns=[target], errors="ignore"))[:, 1]


# ============================
# SAVE SUBMISSION
# ============================
submission = sample_submission.copy()
submission[target] = final_pred
submission.to_csv("submission.csv", index=False)

print("\nsubmission.csv created successfully\n")
