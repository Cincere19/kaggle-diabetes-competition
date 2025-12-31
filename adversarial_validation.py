import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

target = "diagnosed_diabetes"

X_train = train.drop(columns=[target, "id"])
X_test = test.drop(columns=["id"])

X = pd.concat([X_train, X_test], axis=0)
y = np.concatenate([
    np.zeros(len(X_train)),
    np.ones(len(X_test))
])

# label encode categoricals
cat_cols = X.select_dtypes(include="object").columns
for c in cat_cols:
    X[c] = X[c].astype("category").cat.codes

# handle missing
X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True))

X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = GradientBoostingClassifier()
clf.fit(X_tr, y_tr)

preds = clf.predict_proba(X_va)[:, 1]
auc = roc_auc_score(y_va, preds)

print("Adversarial validation AUC:", auc)
