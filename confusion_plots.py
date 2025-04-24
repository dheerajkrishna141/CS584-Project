
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
import os

# -------- CONFIG --------
DATA_FILE = "stock_data_20250423.csv"
RESULTS_FILE = "optimized_results.csv"
START_DATE = "2020-01-01"
PLOT_DIR = "plots_confusion_annotated"
# ------------------------

os.makedirs(PLOT_DIR, exist_ok=True)

def compute_rolling_slope(series, window):
    x = np.arange(window).reshape(-1, 1)
    slopes = [np.nan] * (window - 1)
    for i in range(window, len(series) + 1):
        y = series.iloc[i - window:i].values.reshape(-1, 1)
        if np.isnan(y).any():
            slopes.append(np.nan)
        else:
            model = LinearRegression().fit(x, y)
            slopes.append(model.coef_[0][0] * 10)
    return pd.Series(slopes, index=series.index)

df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
df = df[df["Date"] >= pd.to_datetime(START_DATE)]
df.set_index("Date", inplace=True)

results = pd.read_csv(RESULTS_FILE)
symbols = results["Symbol"].tolist()
available = [sym for sym in symbols if f"{sym}_Close" in df.columns]

for symbol in available[:100]:
    col = f"{symbol}_Close"
    prices = df[col].dropna()
    row = results[results["Symbol"] == symbol]
    sma_s = int(row["Best_SMA_S"].values[0])
    sma_l = int(row["Best_SMA_L"].values[0])

    sma_short = prices.rolling(window=sma_s).mean()
    sma_long = prices.rolling(window=sma_l).mean()
    slope_short = compute_rolling_slope(prices, sma_s)
    slope_long = compute_rolling_slope(prices, sma_l)

    slope_buy = (slope_short > slope_long) & (slope_short.shift(1) <= slope_long.shift(1))
    slope_sell = (slope_short < slope_long) & (slope_short.shift(1) >= slope_long.shift(1))
    slope_signals = pd.Series(np.where(slope_buy, 1, np.where(slope_sell, -1, 0)), index=prices.index)
    returns = np.log(prices / prices.shift(1))
    actuals = pd.Series(np.where(returns > 0, 1, -1), index=returns.index)
    mask = slope_signals != 0
    y_true = actuals[mask]
    y_pred = slope_signals[mask]

    y_true_bin = (y_true == 1).astype(int)
    y_pred_bin = (y_pred == 1).astype(int)

    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    cm = confusion_matrix(y_true_bin, y_pred_bin)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Down", "Up"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{symbol} Confusion Matrix\nAcc={acc:.2f} Prec={prec:.2f} Rec={rec:.2f} F1={f1:.2f}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{symbol}_confusion_matrix.png"), dpi=300)
    plt.close()
