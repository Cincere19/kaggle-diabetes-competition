import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
preds = pd.read_csv("submission_stack_soft.csv")

high = preds.copy()

high["pseudo"] = -1
high.loc[high["diagnosed_diabetes"] > 0.95, "pseudo"] = 1
high.loc[high["diagnosed_diabetes"] < 0.05, "pseudo"] = 0

pseudo = high[high["pseudo"] != -1][["id", "pseudo"]]
pseudo = pseudo.merge(test, on="id")

pseudo.columns = ["id", "diagnosed_diabetes"] + list(test.columns[1:])

new_train = pd.concat([train, pseudo], axis=0)

new_train.to_csv("train_pseudo.csv", index=False)

print("Saved train_pseudo.csv")
