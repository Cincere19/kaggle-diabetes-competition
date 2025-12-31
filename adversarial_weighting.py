import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import lightgbm as lgb

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train["is_train"] = 1
test["is_train"] = 0

combined = pd.concat([train.drop(columns=["diagnosed_diabetes"]), test])

X = combined.drop(columns=["is_train"])
y = combined["is_train"]

adv = XGBClassifier(
    eval_metric="auc",
    max_depth=5,
    learning_rate=0.05,
    n_estimators=700,
    subsample=0.8,
    colsample_bytree=0.8,
)

adv.fit(X, y)
weights = adv.predict_proba(train.drop(columns=["is_train","diagnosed_diabetes"]))[:, 1]
weights = 1 - weights

dtrain = lgb.Dataset(
    train.drop(columns=["is_train","diagnosed_diabetes"]),
    label=train["diagnosed_diabetes"],
    weight=weights
)

params = {"objective":"binary","metric":"auc"}
model = lgb.train(params, dtrain, num_boost_round=1500)

test_pred = model.predict(test.drop(columns=["is_train"]))

pd.DataFrame({
    "id": test["id"],
    "diagnosed_diabetes": test_pred
}).to_csv("submission_adv_weight.csv", index=False)

print("Created submission_adv_weight.csv")
