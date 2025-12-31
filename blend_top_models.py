import pandas as pd
import numpy as np

print("Loading predictions...")

files = [
    "submission_optuna_fast.csv",
    "submission_catboost_fast.csv",
    "submission_features_fast.csv",
    "submission_lgbm_target_encoding.csv",
    "submission_stacking_meta_fast.csv"
]

dfs = []
for f in files:
    try:
        df = pd.read_csv(f)
        dfs.append(df)
        print("Loaded:", f)
    except Exception as e:
        print("Skipping (cannot load):", f, "| Error:", e)

# function to detect prediction column
def get_pred_column(df):
    cols = [c.lower() for c in df.columns]

    # preferred names
    for name in ["diabetes", "prediction", "pred", "prob", "probability"]:
        for c in df.columns:
            if c.lower() == name:
                return c

    # otherwise assume last numeric column
    return df.columns[-1]

# base submission id
sub = dfs[0][["id"]].copy()

preds = np.zeros(len(sub))

for df in dfs:
    col = get_pred_column(df)
    print("Using column:", col)
    preds += df[col].astype(float) / len(dfs)

sub["diabetes"] = preds

sub.to_csv("submission_blend_top.csv", index=False)

print("SUCCESS: submission_blend_top.csv created")
