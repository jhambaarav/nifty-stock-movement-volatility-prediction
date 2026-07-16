"""
nifty_02_train_models.py
--------------------------
Time-series-aware train/test split (NO shuffling, NO random k-fold CV)
and two models: Logistic Regression (baseline) vs Random Forest (main).

Why a single chronological split instead of shuffled k-fold:
Shuffling would let the model train on data from AFTER the day it's
predicting (e.g. training on March 2024 to predict January 2023),
which is lookahead bias - the classic way financial ML projects
quietly cheat. A single date cutoff (train = all data before date X,
test = everything after) guarantees the model never sees the future
during training, at the cost of a smaller effective test set than
k-fold would give you. That tradeoff is the right one here.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv("/sessions/peaceful-youthful-maxwell/mnt/outputs/nifty_features.csv", parse_dates=["date"])
df["target_up"] = df["target_up"].astype(int)

FEATURE_COLS = [
    "price_to_ma20", "price_to_ma50", "ma20_to_ma50", "rsi14",
    "macd", "macd_signal", "macd_hist", "bb_pctb", "bb_width",
    "volume_ratio", "ret_1d", "ret_5d", "ret_10d", "realized_vol_10d",
]

# ---- Chronological split: last 20% of CALENDAR DATES held out (applies to every ticker at once) ----
unique_dates = np.sort(df["date"].unique())
cutoff_date = unique_dates[int(len(unique_dates) * 0.8)]
print(f"Train: dates < {pd.Timestamp(cutoff_date).date()} | Test: dates >= {pd.Timestamp(cutoff_date).date()}")

train = df[df["date"] < cutoff_date].copy()
test = df[df["date"] >= cutoff_date].copy()
print(f"Train rows: {len(train)} | Test rows: {len(test)}")

X_train, y_train = train[FEATURE_COLS], train["target_up"]
X_test, y_test = test[FEATURE_COLS], test["target_up"]

# Logistic Regression needs scaled features; tree models don't, but scaling doesn't hurt them either
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}

# ---- Baseline: Logistic Regression ----
logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
logreg.fit(X_train_scaled, y_train)
pred_lr = logreg.predict(X_test_scaled)
proba_lr = logreg.predict_proba(X_test_scaled)[:, 1]

# ---- Main model: Random Forest ----
rf = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=30,
                             class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
proba_rf = rf.predict_proba(X_test)[:, 1]

def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print(f"Confusion matrix [[TN, FP],[FN, TP]]:\n{cm}")
    print(f"Naive baseline (always predict majority class, 'up'): "
          f"{max(y_true.mean(), 1-y_true.mean()):.4f} accuracy")
    return {"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "tn": int(cm[0,0]), "fp": int(cm[0,1]), "fn": int(cm[1,0]), "tp": int(cm[1,1])}

results["logreg"] = evaluate("Logistic Regression (baseline)", y_test, pred_lr)
results["rf"] = evaluate("Random Forest (main model)", y_test, pred_rf)

# save predictions (needed for backtest + "predicted vs actual" chart)
test["next_day_return"] = test["next_close"] / test["close"] - 1
test_out = test[["ticker", "date", "close", "next_close", "next_day_return", "target_up"]].copy()
test_out["pred_logreg"] = pred_lr
test_out["proba_logreg"] = proba_lr
test_out["pred_rf"] = pred_rf
test_out["proba_rf"] = proba_rf
test_out.to_csv("/sessions/peaceful-youthful-maxwell/mnt/outputs/nifty_test_predictions.csv", index=False)

# save feature importances (Random Forest)
importances = pd.DataFrame({"feature": FEATURE_COLS, "importance": rf.feature_importances_}) \
    .sort_values("importance", ascending=False)
importances.to_csv("/sessions/peaceful-youthful-maxwell/mnt/outputs/nifty_feature_importance.csv", index=False)
print("\nTop feature importances (Random Forest):")
print(importances.to_string(index=False))

pd.DataFrame(results).T.to_csv("/sessions/peaceful-youthful-maxwell/mnt/outputs/nifty_model_metrics.csv")
print("\nSaved nifty_test_predictions.csv, nifty_feature_importance.csv, nifty_model_metrics.csv")
