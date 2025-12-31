import pandas as pd
import numpy as np

files = [
    "submission_optuna_fast.csv",
    "submission_catboost_fast.csv",
    "submission_xgboost_fast.csv",
    "submission_lgbm_target_encoding.csv",
    "submission_stacking_meta_fast.csv",
    "submission_pseudo_v2.csv",
    "submission_blend_top.csv"
]

print("Loading submissions...")

dfs = []
for f in files:
    df = pd.read_csv(f)
    print(f"Loaded: {f} | columns = {list(df.columns)}")
    dfs.append(df)

# ---------- Identify ID and prediction column automatically ----------
def get_columns(df):
    id_candidates = [c for c in df.columns if "id" in c.lower()]
    pred_candidates = [c for c in df.columns if c.lower() not in id_candidates]

    id_col = id_candidates[0]
    pred_col = pred_candidates[0]  # only one prediction column exists

    return id_col, pred_col

id_col, _ = get_columns(dfs[0])

# ---------- Normalize + align all prediction columns ----------
pred_matrix = []

for df in dfs:
    _, pcol = get_columns(df)
    preds = df[pcol].astype(float).values
    pred_matrix.append(preds)

pred_matrix = np.column_stack(pred_matrix)

# ---------- Rank Averaging ----------
ranks = np.zeros(len(pred_matrix))

for j in range(pred_matrix.shape[1]):
    r = pd.Series(pred_matrix[:, j]).rank(method="average")
    r = r / r.max()
    ranks += r

rank_avg = ranks / pred_matrix.shape[1]

# ---------- Geometric Mean ----------
geom_mean = np.exp(np.mean(np.log(pred_matrix + 1e-15), axis=1))

# ---------- Final Hybrid ----------
final_pred = 0.6 * rank_avg + 0.4 * geom_mean

submission = pd.DataFrame({
    id_col: dfs[0][id_col],
    "diabetes": final_pred
})

submission.to_csv("submission_final_super_blend.csv", index=False)

print("\nSUCCESS: submission_final_super_blend.csv created")
