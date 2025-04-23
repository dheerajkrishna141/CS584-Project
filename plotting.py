import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

# -------- CONFIG --------
DATA_FILE = "stock_data_20250422.csv"
RESULTS_FILE = "optimized_results.csv"
START_DATE = "2020-01-01"
PLOT_DIR = "plots_compare_signals"
# ------------------------

os.makedirs(PLOT_DIR, exist_ok=True)

def compute_rolling_slope(series, window):
    """Compute rolling regression slope (scaled)"""
    x = np.arange(window).reshape(-1, 1)
    slopes = [np.nan] * (window - 1)

    for i in range(window, len(series) + 1):
        y = series.iloc[i - window:i].values.reshape(-1, 1)
        if np.isnan(y).any():
            slopes.append(np.nan)
        else:
            model = LinearRegression().fit(x, y)
            slope = model.coef_[0][0]
            scaled_slope = slope * 10
            slopes.append(scaled_slope)

    return pd.Series(slopes, index=series.index)

# Load and filter data
df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
df = df[df["Date"] >= pd.to_datetime(START_DATE)]
df.set_index("Date", inplace=True)

# Load optimized SMA results
results = pd.read_csv(RESULTS_FILE)
symbols = results["Symbol"].tolist()
available = [sym for sym in symbols if f"{sym}_Close" in df.columns]

for symbol in available:
    col = f"{symbol}_Close"
    prices = df[col].dropna()

    row = results[results["Symbol"] == symbol]
    sma_s = int(row["Best_SMA_S"].values[0])
    sma_l = int(row["Best_SMA_L"].values[0])

    # Indicators
    sma_short = prices.rolling(window=sma_s).mean()
    sma_long = prices.rolling(window=sma_l).mean()
    slope_short = compute_rolling_slope(prices, sma_s)
    slope_long = compute_rolling_slope(prices, sma_l)

    # Signal logic — SMA
    sma_buy = (sma_short > sma_long) & (sma_short.shift(1) <= sma_long.shift(1))
    sma_sell = (sma_short < sma_long) & (sma_short.shift(1) >= sma_long.shift(1))

    # Signal logic — Slope
    slope_buy = (slope_short > slope_long) & (slope_short.shift(1) <= slope_long.shift(1))
    slope_sell = (slope_short < slope_long) & (slope_short.shift(1) >= slope_long.shift(1))

    # Plot
    fig, ax1 = plt.subplots(figsize=(16, 10))
    ax1.set_title(f"{symbol} — SMA vs Slope Signals", fontsize=14)

    # Price and SMA
    ax1.plot(prices.index, prices, label="Close Price", color="black", alpha=0.6)
    ax1.plot(prices.index, sma_short, label=f"SMA {sma_s}", linestyle="--", color="blue")
    ax1.plot(prices.index, sma_long, label=f"SMA {sma_l}", linestyle="-.", color="orange")

    # SMA Signals
    ax1.scatter(prices.index[sma_buy], prices[sma_buy], marker="^", color="green", label="SMA Buy", s=70)
    ax1.scatter(prices.index[sma_sell], prices[sma_sell], marker="v", color="red", label="SMA Sell", s=70)

    # Slope Signals
    ax1.scatter(prices.index[slope_buy], prices[slope_buy], marker="^", color="#FF8C00", label="Slope Buy", s=70)
    ax1.scatter(prices.index[slope_sell], prices[slope_sell], marker="v", color="#4169E1", label="Slope Sell", s=70)

    # Plot slopes on second axis
    ax2 = ax1.twinx()
    ax2.plot(slope_short.index, slope_short, label=f"Slope {sma_s}", color="blue", alpha=0.3)
    ax2.plot(slope_long.index, slope_long, label=f"Slope {sma_l}", color="red", alpha=0.3)
    ax2.set_ylabel("Slope Value (scaled)", color="gray")

    # Merge legends from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(handles1 + handles2, labels1 + labels2, loc="upper left", bbox_to_anchor=(0.1, 0.97))

    ax1.grid(True)
    plt.tight_layout()
    out_file = os.path.join(PLOT_DIR, f"{symbol}_sma_vs_slope_signals.png")
    plt.savefig(out_file, dpi=300)
    plt.close()

print("✅ Plots with SMA and Slope crossover signals saved in:", PLOT_DIR)
