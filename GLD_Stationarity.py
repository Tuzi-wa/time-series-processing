"""
GLD_Stationarity.py

Standalone script to analyse the stationarity of GLD (gold ETF) prices.
This script:
- downloads daily GLD prices from Yahoo Finance
- creates a train / test split (80% / 20%)
- applies log transform and first difference on the TRAIN set
- runs ADF tests on original, log, and differenced log series
- plots:
    1) Original vs log price (full sample)
    2) Differenced log price (train only)
    3) ACF of differenced log price (train only)
    4) PACF of differenced log price (train only)
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller


def adf_test(series: pd.Series, name: str = "") -> None:
    """
    Run Augmented Dickey-Fuller test and print results.
    """
    series = series.dropna()
    result = adfuller(series)
    print(f"\nADF Test for {name}")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print("→ Stationary" if result[1] < 0.05 else "→ Non-stationary")


def main():
    # =========================
    # 1. Download GLD data
    # =========================
    ticker = "GLD"
    print(f"Downloading data for {ticker} from Yahoo Finance...")
    df = yf.download(ticker, start="2004-01-01", progress=False)

    if "Adj Close" in df.columns:
        ts = df["Adj Close"].dropna().copy()
    else:
        ts = df["Close"].dropna().copy()

    ts.name = "GLD"
    print(f"Sample size: {len(ts)} observations")

    # =========================
    # 2. Train / Test split (80% / 20%)
    # =========================
    split = int(len(ts) * 0.8)
    train = ts.iloc[:split]
    test = ts.iloc[split:]

    print(f"Train size: {len(train)}, Test size: {len(test)}")

    # =========================
    # 3. Stationarity diagnostics
    # =========================
    # Log transform on full series (for visualization)
    ts_log = np.log(ts)

    # Log transform and difference on TRAIN ONLY (for ARIMA component)
    ts_log_train = np.log(train)
    ts_log_diff_train = ts_log_train.diff().dropna()

    # ADF tests
    adf_test(ts, "Original Price")
    adf_test(ts_log, "Log Price (full sample)")
    adf_test(ts_log_diff_train, "Log Price Diff (TRAIN only, 1st diff)")

    # =========================
    # 4. Plots
    # =========================

    # Figure 1: Original vs Log price
    plt.figure(figsize=(10, 5))
    plt.plot(ts.index, ts.values, label="Original Price")
    plt.plot(ts_log.index, ts_log.values, label="Log Price")
    plt.title("GLD Original Price vs Log Price")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Figure 2: Differenced Log Price (TRAIN only)
    plt.figure(figsize=(10, 5))
    plt.plot(ts_log_diff_train.index, ts_log_diff_train.values, label="Δ Log Price (TRAIN)")
    plt.title("Differenced Log Price (TRAIN, stationary target for ARIMA)")
    plt.xlabel("Date")
    plt.ylabel("Differenced Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Figure 3: ACF of differenced log price (TRAIN)
    plt.figure(figsize=(12, 4))
    plot_acf(ts_log_diff_train, lags=40, ax=plt.gca())
    plt.title("ACF (Differenced Log Price - TRAIN)")
    plt.tight_layout()
    plt.show()

    # Figure 4: PACF of differenced log price (TRAIN)
    plt.figure(figsize=(12, 4))
    plot_pacf(ts_log_diff_train, lags=40, method="ywm", ax=plt.gca())
    plt.title("PACF (Differenced Log Price - TRAIN)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
