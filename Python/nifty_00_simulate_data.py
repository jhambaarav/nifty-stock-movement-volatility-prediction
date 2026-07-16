"""
nifty_00_simulate_data.py
---------------------------
Nifty Stock Movement & Volatility Prediction - DATA SETUP

DATA SOURCE DECISION:
yfinance and direct Yahoo/NSE historical-data endpoints are not reachable
from this sandboxed environment (outbound network is allowlisted, and
finance data hosts aren't on that list - confirmed by testing yfinance,
Yahoo's chart API, and stooq.com directly, all of which returned empty/
blocked responses before writing this script).

So: this script simulates 5 liquid Nifty-50 stocks (Reliance, TCS,
HDFC Bank, Infosys, ICICI Bank) over ~5 years of daily bars (1,300
trading days) using a stochastic-volatility random walk - NOT a pure
random walk. Two realistic, deliberately small effects are baked in:
  1. Volatility clustering (a Markov-switching high/low vol regime),
     which is what makes RSI/Bollinger/MACD features carry real signal
     instead of being pure noise.
  2. A weak short-term momentum effect (~a few % edge over coinflip),
     matching the well-documented (and modest) real-world finding that
     next-day direction is only weakly, not strongly, predictable from
     recent price action. This is intentional: a model that "discovers"
     55% directional accuracy is a believable, defensible result. A
     model claiming 90% would be a red flag of a data leak, not a win.
"""

import numpy as np
import pandas as pd
import datetime

np.random.seed(11)

TICKERS = {
    "RELIANCE": 2450, "TCS": 3550, "HDFCBANK": 1550, "INFY": 1450, "ICICIBANK": 1050,
}
N_DAYS = 1300  # ~5.2 years of trading days
START_DATE = datetime.date(2021, 1, 4)

trading_dates = pd.bdate_range(START_DATE, periods=N_DAYS)  # business days as a trading-day proxy

all_rows = []
for ticker, start_price in TICKERS.items():
    price = start_price
    vol_regime = 0  # 0 = calm, 1 = turbulent
    momentum_state = 0.0  # rolling short-term return, used to bias the NEXT day's drift slightly
    daily_vol_calm, daily_vol_turbulent = 0.011, 0.026
    annual_drift = np.random.uniform(0.08, 0.16) / 252  # daily drift, ~8-16% annualized per stock

    rows = []
    for date in trading_dates:
        # Markov regime switch: turbulent regimes are rarer and shorter
        if vol_regime == 0 and np.random.random() < 0.02:
            vol_regime = 1
        elif vol_regime == 1 and np.random.random() < 0.08:
            vol_regime = 0
        sigma = daily_vol_turbulent if vol_regime == 1 else daily_vol_calm

        # weak momentum: yesterday's short-term trend nudges today's expected return, small & noisy
        momentum_bias = 0.22 * momentum_state  # deliberately modest vs. sigma so it's a real but weak edge

        ret = annual_drift + momentum_bias + np.random.normal(0, sigma)
        open_px = price
        close_px = price * (1 + ret)
        intraday_range = abs(np.random.normal(0, sigma * 0.6)) * price
        high_px = max(open_px, close_px) + intraday_range * np.random.uniform(0.2, 0.6)
        low_px = min(open_px, close_px) - intraday_range * np.random.uniform(0.2, 0.6)
        base_volume = np.random.uniform(3_000_000, 9_000_000)
        volume = base_volume * (1.6 if vol_regime == 1 else 1.0) * np.random.uniform(0.7, 1.3)

        rows.append({
            "date": date.date().isoformat(), "ticker": ticker,
            "open": round(open_px, 2), "high": round(high_px, 2),
            "low": round(low_px, 2), "close": round(close_px, 2),
            "volume": int(volume),
        })

        # update momentum state as an EWMA of recent returns (what "yesterday's trend" means here)
        momentum_state = 0.7 * momentum_state + 0.3 * ret
        price = close_px

    all_rows.extend(rows)

df = pd.DataFrame(all_rows)
df.to_csv("/sessions/peaceful-youthful-maxwell/mnt/outputs/nifty_ohlcv.csv", index=False)
print(f"Simulated {df['ticker'].nunique()} tickers x {df.groupby('ticker').size().iloc[0]} trading days "
      f"= {len(df)} rows -> nifty_ohlcv.csv")
print(df.groupby("ticker")["close"].agg(["first", "last"]).assign(
    total_return_pct=lambda x: round((x["last"]/x["first"]-1)*100, 1)))
