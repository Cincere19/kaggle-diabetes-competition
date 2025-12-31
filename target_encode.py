import pandas as pd
from category_encoders import TargetEncoder
from sklearn.model_selection import KFold
import lightgbm as lgb

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

TARGET = "diagnosed_diabetes"
cat_cols = [
    "gender","ethnicity","education_level","income_level",
    "smoking_status","employment_status"
]

X = train.drop(columns=[TARGET])
y = train[TARGET]

te = TargetEncoder(cols=cat_cols, smoothing=0.2)
te.fit(X[cat_cols], y)

X_enc = X.copy()
test_enc = test.copy()

X_enc[cat_cols] = te.transform(X[cat_cols])
test_enc[cat_cols] = te.transform(test[cat_cols])

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 96,
    "feature_fraction": 0.75
}

dtrain = lgb.Dataset(X_enc, label=y)
model = lgb.train(params, dtrain, num_boost_round=1800)

pred = model.predict(test_enc)

pd.DataFrame({
    "id": test["id"],
    "diagnosed_diabetes": pred
}).to_csv("submission_target_encode.csv", index=False)

print("DONE: submission_target_encode.csv created")
