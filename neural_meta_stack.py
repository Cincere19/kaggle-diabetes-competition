import pandas as pd
from sklearn.neural_network import MLPClassifier

o1 = pd.read_csv("submission_optuna_fast.csv")
o2 = pd.read_csv("submission_features_fast.csv")
o3 = pd.read_csv("submission_catboost_fast.csv")

train = pd.read_csv("train.csv")

meta_X = pd.DataFrame({
    "o1": o1["diagnosed_diabetes"],
    "o2": o2["diagnosed_diabetes"],
    "o3": o3["diagnosed_diabetes"]
})

meta_y = train["diagnosed_diabetes"][:len(meta_X)]

meta = MLPClassifier(
    hidden_layer_sizes=(64,32),
    activation="relu",
    max_iter=400
)

meta.fit(meta_X, meta_y)

test_meta = pd.DataFrame({
    "o1": o1["diagnosed_diabetes"],
    "o2": o2["diagnosed_diabetes"],
    "o3": o3["diagnosed_diabetes"]
})

stack_pred = meta.predict_proba(test_meta)[:,1]

pd.DataFrame({
    "id": o1["id"],
    "diagnosed_diabetes": stack_pred
}).to_csv("submission_neural_stack.csv", index=False)

print("submission_neural_stack.csv created")
