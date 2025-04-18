import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def get_latest_data_file():
    """Retrieve the latest stock_data_*.csv file"""
    files = sorted(glob.glob("stock_data_*.csv"), reverse=True)
    return files[0] if files else None

def load_stock_data(filename):
    """Load and reshape stock data for multi-ticker access"""
    df = pd.read_csv(filename, index_col=0, parse_dates=True)
    tickers = set(col.split('_')[0] for col in df.columns if '_' in col)
    stock_data = {}

    for ticker in tickers:
        cols = [f"{ticker}_{col}" for col in ['Open', 'High', 'Low', 'Close', 'Volume']]
        cols = [col for col in cols if col in df.columns]
        if cols:
            sub_df = df[cols].copy()
            sub_df.columns = [col.split('_')[1] for col in cols]
            sub_df.dropna(inplace=True)
            stock_data[ticker] = sub_df

    return stock_data

def add_smas_and_signals(df, sma_s, sma_l):
    df["SMA_S"] = df["Close"].rolling(window=sma_s).mean()
    df["SMA_L"] = df["Close"].rolling(window=sma_l).mean()
    df["Signal"] = 0
    df.loc[df["SMA_S"] > df["SMA_L"], "Signal"] = 1
    df["Buy_Signal"] = (df["Signal"].diff() == 1)
    df["Sell_Signal"] = (df["Signal"].diff() == -1)
    return df

def plot_ticker_group(tickers, results_df, stock_data, group_idx):
    fig, axs = plt.subplots(5, 1, figsize=(14, 20))
    fig.suptitle(f"SMA Signals: Tickers {group_idx * 5 + 1}–{group_idx * 5 + len(tickers)}", fontsize=16)

    for ax, ticker in zip(axs, tickers):
        row = results_df[results_df["Symbol"] == ticker].iloc[0]
        df = stock_data[ticker].copy()
        df = add_smas_and_signals(df, row["Best_SMA_S"], row["Best_SMA_L"])

        ax.plot(df.index, df["Close"], label="Close", alpha=0.6)
        ax.plot(df.index, df["SMA_S"], label=f"SMA {row['Best_SMA_S']}", linestyle="--")
        ax.plot(df.index, df["SMA_L"], label=f"SMA {row['Best_SMA_L']}", linestyle=":")
        ax.scatter(df[df["Buy_Signal"]].index, df[df["Buy_Signal"]]["Close"], color='green', label="Buy Signal", marker="^")
        ax.scatter(df[df["Sell_Signal"]].index, df[df["Sell_Signal"]]["Close"], color='red', marker='v', label="Sell Signal")
        
        ax.set_title(f"{ticker}")
        ax.legend()
        ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def main():
    # Load files
    results_file = "optimized_results.csv"
    data_file = get_latest_data_file()

    if not os.path.exists(results_file):
        print("Error: 'optimized_results.csv' not found.")
        return
    if not data_file:
        print("Error: No 'stock_data_*.csv' files found.")
        return

    results_df = pd.read_csv(results_file)
    stock_data = load_stock_data(data_file)

    # Filter tickers that have matching data
    filtered_df = results_df[results_df["Symbol"].isin(stock_data.keys())].reset_index(drop=True)

    # Plot in groups of 5
    for i in range(0, len(filtered_df), 5):
        group_tickers = filtered_df["Symbol"].iloc[i:i+5].tolist()
        plot_ticker_group(group_tickers, filtered_df, stock_data, i // 5)

if __name__ == "__main__":
    main()
