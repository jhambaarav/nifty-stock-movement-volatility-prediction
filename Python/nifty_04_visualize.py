"""
nifty_04_visualize.py
------------------------
Three required charts: feature importance, predicted vs actual
movement over time (for one representative stock), and the backtest
equity curve (ML strategy vs. buy-and-hold).
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
OUT = "/sessions/peaceful-youthful-maxwell/mnt/outputs/"

# ---- 1. Feature importance ----
imp = pd.read_csv(OUT + "nifty_feature_importance.csv")
plt.figure(figsize=(8, 6))
sns.barplot(data=imp, x="importance", y="feature", palette="crest")
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance"); plt.ylabel("")
plt.tight_layout(); plt.savefig(OUT + "nifty_chart1_feature_importance.png", dpi=140); plt.close()

# ---- 2. Predicted vs actual movement over time (one representative stock) ----
preds = pd.read_csv(OUT + "nifty_test_predictions.csv", parse_dates=["date"])
rel = preds[preds["ticker"] == "RELIANCE"].sort_values("date").reset_index(drop=True)
rel["cum_actual"] = (1 + rel["next_day_return"]).cumprod()
rel["correct"] = (rel["pred_rf"] == rel["target_up"]).astype(int)

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(rel["date"], rel["close"], color="#2c6e91", label="RELIANCE Close Price")
ax1.set_ylabel("Price", color="#2c6e91")
correct_pts = rel[rel["correct"] == 1]
wrong_pts = rel[rel["correct"] == 0]
ax1.scatter(correct_pts["date"], correct_pts["close"], color="#3ecf8e", s=14, label="RF correct", zorder=5)
ax1.scatter(wrong_pts["date"], wrong_pts["close"], color="#e05656", s=14, label="RF wrong", zorder=5)
ax1.legend(loc="upper left", fontsize=8)
plt.title("RELIANCE: Price with Random Forest Next-Day Direction Correctness (test period)")
plt.tight_layout(); plt.savefig(OUT + "nifty_chart2_predicted_vs_actual.png", dpi=140); plt.close()

# ---- 3. Backtest equity curve ----
port = pd.read_csv(OUT + "nifty_backtest_portfolio.csv", parse_dates=["date"])
plt.figure(figsize=(10, 5))
plt.plot(port["date"], port["strategy_equity"], label="ML Strategy (RF signal, long/flat)", color="#3ecf8e", linewidth=2)
plt.plot(port["date"], port["bh_equity"], label="Buy & Hold (equal-weight, 5 stocks)", color="#4f8ff7", linewidth=2)
plt.title("Backtest Equity Curve: ML Strategy vs. Buy & Hold (test period)")
plt.xlabel("Date"); plt.ylabel("Growth of Rs.1 invested")
plt.legend()
plt.tight_layout(); plt.savefig(OUT + "nifty_chart3_equity_curve.png", dpi=140); plt.close()

print("Saved 3 charts: feature importance, predicted-vs-actual, equity curve.")
