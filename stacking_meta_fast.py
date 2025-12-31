import pandas as pd

# === Load base model submissions ===
lgb = pd.read_csv("submission_optuna_fast.csv")
cat = pd.read_csv("submission_catboost_fast.csv")
xgb = pd.read_csv("submission_xgboost_fast.csv")

# Merge on id to keep alignment correct
df = lgb.merge(cat, on="id", suffixes=("_lgb", "_cat"))
df = df.merge(xgb, on="id")

df.columns = ["id", "lgb", "cat", "xgb"]

# ========= BLENDING WEIGHTS =========
# LightGBM strongest   -> highest weight
# CatBoost second best
# XGBoost stabilizes tails
blend = (
    0.50 * df["lgb"] +
    0.30 * df["cat"] +
    0.20 * df["xgb"]
)

# Clip to valid probability range
blend = blend.clip(0.00001, 0.99999)

# Create stacked submission
submission = pd.DataFrame({
    "id": df["id"],
    "diabetes": blend
})

submission.to_csv("submission_stacking_meta_fast.csv", index=False)

print("SUCCESS: submission_stacking_meta_fast.csv created")
