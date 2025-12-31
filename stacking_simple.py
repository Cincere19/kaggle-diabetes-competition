import pandas as pd

lgb = pd.read_csv("submission_optuna_fast.csv")
cat = pd.read_csv("submission_catboost_fast.csv")
xgb = pd.read_csv("submission_xgboost_fast.csv")

sub = lgb.copy()

sub["diagnosed_diabetes"] = (
    0.4 * lgb["diagnosed_diabetes"] +
    0.3 * cat["diagnosed_diabetes"] +
    0.3 * xgb["diagnosed_diabetes"]
)

sub.to_csv("submission_stack_soft.csv", index=False)

print("Saved submission_stack_soft.csv")
