import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import math
import os

def load_optimized_results():
    return pd.read_csv("optimized_results.csv")

def load_price_data():
    files = [f for f in os.listdir() if f.startswith("stock_data_") and f.endswith(".csv")]
    files.sort(reverse=True)
    if not files:
        raise FileNotFoundError("No stock_data_*.csv file found")
    return pd.read_csv(files[0], parse_dates=["Date"])

def add_smas_and_signals(df, sma_s, sma_l):
    df["SMA_S"] = df["Close"].rolling(window=sma_s).mean()
    df["SMA_L"] = df["Close"].rolling(window=sma_l).mean()
    df["Signal"] = 0
    df.loc[df["SMA_S"] > df["SMA_L"], "Signal"] = 1
    df["Buy_Signal"] = (df["Signal"].diff() == 1)
    df["Sell_Signal"] = (df["Signal"].diff() == -1)
    return df

def plot_interactive_tickers(data, results, max_per_plot=5):
    tickers = results["Symbol"].tolist()
    total_pages = math.ceil(len(tickers) / max_per_plot)

    for page in range(total_pages):
        fig = make_subplots(rows=max_per_plot, cols=1,
                            shared_xaxes=True,
                            subplot_titles=tickers[page * max_per_plot:(page + 1) * max_per_plot])

        for i, ticker in enumerate(tickers[page * max_per_plot:(page + 1) * max_per_plot]):
            row_data = results[results["Symbol"] == ticker].iloc[0]
            df = data.get(ticker)
            if df is None:
                continue

            df = df.copy()
            df.index = pd.to_datetime(df.index)
            df = add_smas_and_signals(df, row_data["Best_SMA_S"], row_data["Best_SMA_L"])

            fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name=f"{ticker} Close", line=dict(width=1)),
                          row=i + 1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA_S"], mode="lines", name=f"{ticker} SMA_S", line=dict(dash='dot')),
                          row=i + 1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA_L"], mode="lines", name=f"{ticker} SMA_L", line=dict(dash='dash')),
                          row=i + 1, col=1)
            fig.add_trace(go.Scatter(x=df[df["Buy_Signal"]].index, y=df[df["Buy_Signal"]]["Close"],
                                     mode="markers", marker=dict(symbol="triangle-up", color="green", size=8),
                                     name="Buy Signal"), row=i + 1, col=1)
            fig.add_trace(go.Scatter(x=df[df["Sell_Signal"]].index, y=df[df["Sell_Signal"]]["Close"],
                                     mode="markers", marker=dict(symbol="triangle-down", color="red", size=8),
                                     name="Sell Signal"), row=i + 1, col=1)

        fig.update_layout(height=300 * max_per_plot, title_text=f"Page {page + 1} - SMA Strategy Signals", showlegend=False)
        fig.update_xaxes(rangeslider_visible=True)

        fig.show()

# Run the visualization
if __name__ == "__main__":
    results_df = load_optimized_results()
    price_df = load_price_data()

    # Convert price_df to a dictionary {ticker: DataFrame}
    tickers = set(col.split("_")[0] for col in price_df.columns if "_" in col)
    data_dict = {}
    for ticker in tickers:
        sub_cols = [f"{ticker}_Open", f"{ticker}_High", f"{ticker}_Low", f"{ticker}_Close", f"{ticker}_Volume"]
        if all(col in price_df.columns for col in sub_cols):
            df_ticker = price_df[["Date"] + sub_cols].copy()
            df_ticker.set_index("Date", inplace=True)
            df_ticker.columns = ["Open", "High", "Low", "Close", "Volume"]
            data_dict[ticker] = df_ticker.dropna()

    plot_interactive_tickers(data_dict, results_df, max_per_plot=5)
