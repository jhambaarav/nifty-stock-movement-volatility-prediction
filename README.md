# Nifty Stock Movement & Volatility Prediction

An ML-for-markets project built for a quant/trading portfolio — the goal is to show whether a model's predictions would have actually made money, not just how accurate they were.

## Overview

Most stock-prediction ML projects stop at a confusion matrix and call 54% accuracy a win. That's not the question that matters — the real question is whether trading on the model's predictions would have been profitable after accounting for being wrong sometimes and sitting out other days. This project builds a next-day direction classifier for 5 liquid Nifty-50 stocks from technical indicators, then converts it into an actual long/flat trading signal and backtests it against simply buying and holding.

## Data

yfinance and direct NSE/Yahoo historical-data endpoints weren't reachable from the build environment. `nifty_00_simulate_data.py` instead simulates ~5.2 years of daily OHLCV (1,300 trading days × 5 stocks: Reliance, TCS, HDFC Bank, Infosys, ICICI Bank) using a stochastic-volatility random walk with volatility clustering and a deliberately weak short-term momentum effect — sized so the resulting directional edge lands in a believable 51–54% range rather than an unrealistically high one.

## Pipeline

1. **`nifty_00_simulate_data.py`** — generates the OHLCV dataset
2. **`nifty_01_features.py`** — builds technical indicator features (RSI, MACD, moving averages, Bollinger Bands, volume ratio, lagged returns) and the next-day direction target
3. **`nifty_02_train_models.py`** — trains and compares Logistic Regression (baseline) vs. Random Forest (main model) with a strict chronological train/test split
4. **`nifty_03_backtest.py`** — converts predictions into a long/flat trading signal and backtests it against buy-and-hold
5. **`nifty_04_visualize.py`** — feature importance chart, predicted-vs-actual movement, and the backtest equity curve

## Why classification, not volatility regression

A binary up/down prediction converts directly into a trading position (long if "up", flat otherwise) with no extra modeling step, and classification metrics (precision, recall) map directly onto trading questions — precision on the "up" class is literally the strategy's daily hit rate. Volatility regression is a reasonable alternative but doesn't produce a position by itself.

## Model results

Chronological split: train on 2021–2024 (5,000 rows), test on 2025 (1,255 rows) — no shuffling, so the model never trains on data from after the day it's predicting.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 53.15% | 55.04% | 59.55% | 57.21% |
| Random Forest | 52.67% | 54.44% | 61.36% | 57.69% |
| Naive baseline (always "up") | 52.59% | – | – | – |

Both models land just barely above the naive baseline — a modest, honest result consistent with how weak real short-horizon equity signals actually are.

## Backtest: the part that matters

Signal: long (equal-weight) when Random Forest predicts "up" for tomorrow, flat otherwise. No leverage, no shorting, no transaction costs modeled.

| Metric | ML Strategy | Buy & Hold |
|---|---|---|
| Total return | 26.42% | 31.95% |
| Sharpe ratio | 2.32 | 2.20 |
| Max drawdown | -3.54% | -7.88% |
| Avg. market exposure | 59.4% | 100% |

The model was "right" more often than a coinflip, but the strategy made **less** money than doing nothing — 2025 was a strong bull run in this simulation, and sitting out ~40% of days meant missing some of the best up-days along with the down-days. The strategy did win on risk-adjusted terms (higher Sharpe, much shallower drawdown), which is a real result for a risk-averse mandate, just not a return-maximizing one.

## Project structure

```
nifty_00_simulate_data.py          # OHLCV data generation
nifty_01_features.py                # technical indicators + target
nifty_02_train_models.py            # model training + evaluation
nifty_03_backtest.py                # signal backtest vs. buy-and-hold
nifty_04_visualize.py               # charts
nifty_ohlcv.csv                     # raw OHLCV data
nifty_test_predictions.csv          # held-out test predictions
nifty_feature_importance.csv        # Random Forest feature importances
nifty_model_metrics.csv             # model evaluation metrics
nifty_backtest_portfolio.csv        # daily portfolio equity (strategy vs. buy-and-hold)
nifty_backtest_summary.csv          # backtest summary stats
nifty_chart1_feature_importance.png
nifty_chart2_predicted_vs_actual.png
nifty_chart3_equity_curve.png
```

## Limitations

Statistical accuracy is not the same question as profitability — this project's own results prove it. No transaction costs or slippage are modeled, which would meaningfully erode the already-thin edge given ~40% of days involve a position change. The Random Forest was deliberately kept shallow to resist overfitting a modest signal, but one held-out year is a thin basis for a real capital-allocation decision — a production version would need walk-forward validation across multiple market regimes (bull, bear, sideways, high-volatility) before it earns any trust.
