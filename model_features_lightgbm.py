import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

print("Loading data...")

target = "diagnosed_diabetes"

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

y = train[target]
train = train.drop(columns=[target])

df = pd.concat([train, test], axis=0)

# ================================
# DOWNCAST NUMERICS
# ================================
def downcast(df):
    for c in df.select_dtypes(include=["float64", "int64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    return df

df = downcast(df)

# ================================
# FEATURE ENGINEERING
# ================================
def add_features(df):

    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["map_pressure"] = df["diastolic_bp"] + df["pulse_pressure"] / 3

    df["chol_hdl_ratio"] = df["cholesterol_total"] / (df["hdl_cholesterol"] + 1)
    df["ldl_hdl_ratio"] = df["ldl_cholesterol"] / (df["hdl_cholesterol"] + 1)
    df["tg_hdl_ratio"] = df["triglycerides"] / (df["hdl_cholesterol"] + 1)

    df["bmi_age_interaction"] = df["bmi"] * df["age"]
    df["waist_bmi_ratio"] = df["waist_to_hip_ratio"] * df["bmi"]

    df["lifestyle_risk_index"] = (
        df["alcohol_consumption_per_week"]
        + df["screen_time_hours_per_day"]
        - df["physical_activity_minutes_per_week"] / 60
        - df["sleep_hours_per_day"]
    )

    for col in ["triglycerides", "cholesterol_total", "bmi"]:
        df[f"log_{col}"] = np.log1p(np.abs(df[col]))

    return df


print("Adding engineered features...")
df = add_features(df)

# ================================
# HANDLE CATEGORICAL NATIVE
# ================================
cat_cols = df.select_dtypes(include="object").columns.tolist()
for c in cat_cols:
    df[c] = df[c].astype("category")

df = df.replace([np.inf, -np.inf], np.nan).fillna(df.median(numeric_only=True))

train_processed = df.iloc[: len(train)]
test_processed = df.iloc[len(train):]

# ================================
# LIGHTGBM PARAMS (FAST + FIXED)
# ================================
params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 64,
    "boosting_type": "goss",     # keep GOSS
    "feature_fraction": 0.9,
    "max_bin": 255,
    "num_threads": 8,
    "lambda_l2": 1.0,
    "verbose": -1,
    "first_metric_only": True,
}

print("Training LightGBM (fast mode with GOSS)...")

# ================================
# STRATIFIED 3-FOLD CV
# ================================
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

oof = np.zeros(len(train))
pred_test = np.zeros(len(test))

for fold, (tr, va) in enumerate(skf.split(train_processed, y)):
    print(f"\n===== Fold {fold+1} =====")

    X_tr, X_va = train_processed.iloc[tr], train_processed.iloc[va]
    y_tr, y_va = y.iloc[tr], y.iloc[va]

    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
    dva = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_cols)

    model = lgb.train(
        params,
        dtr,
        valid_sets=[dva],
        num_boost_round=1500,
        callbacks=[lgb.early_stopping(50)],
    )

    oof[va] = model.predict(X_va)
    pred_test += model.predict(test_processed) / skf.n_splits

cv_auc = roc_auc_score(y, oof)
print("\n==============================")
print("FAST Feature-engineered CV AUC:", cv_auc)
print("==============================")

submission = pd.DataFrame({
    "id": test["id"],
    "diagnosed_diabetes": pred_test
})

submission.to_csv("submission_features_fast.csv", index=False)
print("\nsubmission_features_fast.csv created successfully")
