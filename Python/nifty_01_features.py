"""
nifty_01_features.py
----------------------
Technical indicator feature engineering + target construction.

TARGET CHOICE: classification (next-day up/down), not regression on
volatility. Reasoning (also explained in the writeup):
  - A binary up/down prediction converts directly into a trading signal
    (long if predicted up, flat otherwise) with no extra modeling step,
    which is exactly what the backtest in step 4 needs.
  - Classification metrics (precision/recall/F1, confusion matrix) map
    cleanly onto trading-relevant questions ("when we say up, how often
    are we right" = precision = how often the strategy is actually long
    on a winning day).
  - Volatility regression is a legitimate alternative (and arguably
    easier to get a high R^2 on, since volatility is more persistent
    than returns), but it doesn't produce a directional signal by
    itself - you'd still need a second rule to turn a vol forecast into
    a position. Direction classification is the more honest, single-step
    choice for a "does this make money" project.
"""

import pandas as pd
import numpy as np

df = pd.read_csv("/sessions/peaceful-youthful-maxwell/mnt/outputs/nifty_ohlcv.csv", parse_dates=["date"])
df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

feature_frames = []
for ticker, g in df.groupby("ticker"):
    g = g.copy().sort_values("date")
    close = g["close"]

    # --- Moving averages ---
    g["ma20"] = close.rolling(20).mean()
    g["ma50"] = close.rolling(50).mean()
    g["ma200"] = close.rolling(200).mean()
    g["price_to_ma20"] = close / g["ma20"] - 1
    g["price_to_ma50"] = close / g["ma50"] - 1
    g["ma20_to_ma50"] = g["ma20"] / g["ma50"] - 1

    # --- RSI(14) ---
    g["rsi14"] = rsi(close, 14)

    # --- MACD(12,26,9) ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    g["macd"] = ema12 - ema26
    g["macd_signal"] = g["macd"].ewm(span=9, adjust=False).mean()
    g["macd_hist"] = g["macd"] - g["macd_signal"]

    # --- Bollinger Bands(20, 2) ---
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    g["bb_upper"] = bb_mid + 2 * bb_std
    g["bb_lower"] = bb_mid - 2 * bb_std
    g["bb_pctb"] = (close - g["bb_lower"]) / (g["bb_upper"] - g["bb_lower"])  # 0=at lower band, 1=at upper band
    g["bb_width"] = (g["bb_upper"] - g["bb_lower"]) / bb_mid

    # --- Volume trend ratio ---
    g["vol_ma20"] = g["volume"].rolling(20).mean()
    g["volume_ratio"] = g["volume"] / g["vol_ma20"]

    # --- Lagged returns (momentum features) ---
    g["ret_1d"] = close.pct_change(1)
    g["ret_5d"] = close.pct_change(5)
    g["ret_10d"] = close.pct_change(10)
    g["realized_vol_10d"] = g["ret_1d"].rolling(10).std()

    # --- Target: next-day direction (1 = up, 0 = down/flat) ---
    g["next_close"] = close.shift(-1)
    g["target_up"] = np.where(g["next_close"].isna(), np.nan, (g["next_close"] > close).astype(float))

    feature_frames.append(g)

full = pd.concat(feature_frames, ignore_index=True)

FEATURE_COLS = [
    "price_to_ma20", "price_to_ma50", "ma20_to_ma50", "rsi14",
    "macd", "macd_signal", "macd_hist", "bb_pctb", "bb_width",
    "volume_ratio", "ret_1d", "ret_5d", "ret_10d", "realized_vol_10d",
]

# drop warm-up rows (MA200/rolling windows need history) and the last row per ticker (no next_close)
full_clean = full.dropna(subset=FEATURE_COLS + ["target_up"]).reset_index(drop=True)

full_clean.to_csv("/sessions/peaceful-youthful-maxwell/mnt/outputs/nifty_features.csv", index=False)
print(f"Feature matrix: {full_clean.shape[0]} rows x {len(FEATURE_COLS)} features (from {full.shape[0]} raw rows)")
print("Target balance (share of up-days):", round(full_clean["target_up"].mean(), 4))
print(full_clean[["ticker", "date"] + FEATURE_COLS + ["target_up"]].head(3).to_string())
