# 🩺 Kaggle – Diabetes Prediction Competition

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-green)
![CatBoost](https://img.shields.io/badge/CatBoost-Boosting-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Tree%20Boosting-red)
![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-purple)
![Status](https://img.shields.io/badge/Competition-Active-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

This repository contains my **end-to-end machine learning pipeline** for the **Kaggle Diabetes Prediction** competition.  
The objective is to predict the likelihood of diabetes using **demographic, lifestyle, and clinical features**.

I built:

- Strong single-model baselines
- Multiple gradient boosting models
- Advanced ensembling system
- Pseudo-labeling & meta-stacking
- Final rank-based blending strategy

---

## 🏆 Leaderboard Performance

| Metric | Score |
|--------|-------|
| **Best Public Leaderboard Score** | **0.69869** |
| Models Used | LGBM, CatBoost, XGBoost, Blending |

> Currently aiming for Top-3 finish.

---

## 🧠 Machine Learning Methods Used

- LightGBM (baseline & tuned)
- XGBoost with LR decay
- CatBoost categorical boosting
- Target Encoding for high-cardinality variables
- Pseudo-Labeling (semi-supervised learning)
- Model Stacking (meta learner)
- Rank Averaging & Weighted Blending
- Optuna Bayesian hyperparameter tuning
- Adversarial validation to detect leakage & drift

---

## 🧰 Tech Stack

- Python
- Pandas / NumPy
- LightGBM
- XGBoost
- CatBoost
- Optuna
- Scikit-learn
- Matplotlib / Seaborn (EDA)

---

## 📂 Repository Structure

```
kaggle_diabetes_competition/
│
├── train.csv
├── test.csv
│
├── optuna_lgbm_fast.py
├── model_lightgbm.py
├── model_xgboost_fast.py
├── model_catboost_fast.py
│
├── stacking_simple.py
├── stacking_meta_fast.py
├── blend_top_models.py
├── final_super_blend.py
│
├── pseudo_label_v2.py
│
└── README.md
```

---

## 🚀 Reproducibility – How to Run

```
pip install -r requirements.txt
python optuna_lgbm_fast.py
python model_catboost_fast.py
python model_xgboost_fast.py
python stacking_meta_fast.py
python final_super_blend.py
```

---

## 🧾 Results Summary

| Model | Score |
|------|------|
| Optuna + LGBM | 0.697 |
| CatBoost tuned | 0.697 |
| XGBoost tuned | 0.694 |
| Target Encoding + LGBM | 0.697 |
| Meta-Stacking | 0.6973 |
| **Final Blend** | **0.69869** |

---

## 📈 Key Insights

- Categorical encoding and stacked blending provide major lift
- Simple models with smart ensembling outperform deep nets
- Pseudo-labeling improves generalization
- Public LB variance is significant — rank-based blending stabilizes score

---

## 👨‍💻 Author

**Tonumay Bhattacharya**  
📍 India  

---

## 📝 License

This project is licensed under the MIT License.
