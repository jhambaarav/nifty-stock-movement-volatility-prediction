"""
nifty_03_backtest.py
-----------------------
Turns Random Forest predictions into a simple long/flat trading signal
and backtests it against buy-and-hold, on the held-out test period only
(the same period the model never trained on).

Strategy: for each stock, each day: if the model predicts "up" for
tomorrow, hold a long position sized 1/N_tickers of the portfolio for
that one day; otherwise stay flat (cash, 0% return) for that stock that
day. Rebalanced daily across the 5-stock equal-weight portfolio.
No leverage, no shorting, no transaction costs modeled (flagged as a
limitation in the writeup).
"""

import pandas as pd
import numpy as np

df = pd.read_csv("/sessions/peaceful-youthful-maxwell/mnt/outputs/nifty_test_predictions.csv", parse_dates=["date"])

N_TICKERS = df["ticker"].nunique()
WEIGHT = 1.0 / N_TICKERS

# ---- per-stock daily strategy return: only earn the next-day return if the model said "up" ----
df["strategy_ret"] = np.where(df["pred_rf"] == 1, df["next_day_return"], 0.0)
df["bh_ret"] = df["next_day_return"]  # buy-and-hold: always exposed

# ---- portfolio-level daily return: equal-weight average across the 5 tickers ----
portfolio = df.groupby("date").agg(strategy_ret=("strategy_ret", "mean"), bh_ret=("bh_ret", "mean")).reset_index()
portfolio = portfolio.sort_values("date").reset_index(drop=True)

portfolio["strategy_equity"] = (1 + portfolio["strategy_ret"]).cumprod()
portfolio["bh_equity"] = (1 + portfolio["bh_ret"]).cumprod()

def sharpe_ratio(returns, periods_per_year=252, rf_annual=0.06):
    rf_daily = rf_annual / periods_per_year
    excess = returns - rf_daily
    if excess.std() == 0:
        return np.nan
    return (excess.mean() / excess.std()) * np.sqrt(periods_per_year)

def max_drawdown(equity_curve):
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return drawdown.min()

strat_sharpe = sharpe_ratio(portfolio["strategy_ret"])
bh_sharpe = sharpe_ratio(portfolio["bh_ret"])
strat_mdd = max_drawdown(portfolio["strategy_equity"])
bh_mdd = max_drawdown(portfolio["bh_equity"])

strat_total_return = portfolio["strategy_equity"].iloc[-1] - 1
bh_total_return = portfolio["bh_equity"].iloc[-1] - 1

n_days = len(portfolio)
strat_ann_return = (1 + strat_total_return) ** (252 / n_days) - 1
bh_ann_return = (1 + bh_total_return) ** (252 / n_days) - 1

print("=" * 70)
print(f"BACKTEST: {portfolio['date'].min().date()} to {portfolio['date'].max().date()} ({n_days} trading days)")
print("=" * 70)
print(f"{'Metric':<30}{'ML Strategy (RF signal)':<26}{'Buy & Hold':<20}")
print(f"{'Total return':<30}{strat_total_return*100:>10.2f}%{'':<14}{bh_total_return*100:>10.2f}%")
print(f"{'Annualized return':<30}{strat_ann_return*100:>10.2f}%{'':<14}{bh_ann_return*100:>10.2f}%")
print(f"{'Sharpe ratio (rf=6%)':<30}{strat_sharpe:>10.3f}{'':<14}{bh_sharpe:>10.3f}")
print(f"{'Max drawdown':<30}{strat_mdd*100:>10.2f}%{'':<14}{bh_mdd*100:>10.2f}%")
print(f"{'Avg daily exposure (long)':<30}{(df['pred_rf'].mean())*100:>10.1f}%{'':<14}{'100.0%':<10}")

verdict = ("The ML signal underperformed buy-and-hold" if strat_total_return < bh_total_return
           else "The ML signal outperformed buy-and-hold")
print(f"\nVerdict: {verdict} on total return over this test window.")
print("Statistically 'better than a coinflip' (52.7% accuracy) did NOT necessarily translate into "
      "beating a passive strategy once actual position sizing and being out of the market on 'flat' "
      "days is accounted for - see the writeup for why these two things are not the same question.")

portfolio.to_csv("/sessions/peaceful-youthful-maxwell/mnt/outputs/nifty_backtest_portfolio.csv", index=False)

summary = {
    "strat_total_return_pct": round(strat_total_return*100, 2), "bh_total_return_pct": round(bh_total_return*100, 2),
    "strat_ann_return_pct": round(strat_ann_return*100, 2), "bh_ann_return_pct": round(bh_ann_return*100, 2),
    "strat_sharpe": round(strat_sharpe, 3), "bh_sharpe": round(bh_sharpe, 3),
    "strat_mdd_pct": round(strat_mdd*100, 2), "bh_mdd_pct": round(bh_mdd*100, 2),
    "avg_exposure_pct": round(df["pred_rf"].mean()*100, 1),
}
pd.Series(summary).to_csv("/sessions/peaceful-youthful-maxwell/mnt/outputs/nifty_backtest_summary.csv")
print("\nSaved nifty_backtest_portfolio.csv, nifty_backtest_summary.csv")
