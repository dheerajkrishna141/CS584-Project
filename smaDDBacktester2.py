import yfinance as yf
import pandas as pd
import numpy as np
from warnings import filterwarnings
import requests
from datetime import datetime,timedelta
import os
import schedule
import time

filterwarnings("ignore")

def fetch_symbols():
    """ Fetches symbols for NASDAQ-100 and S&P 500 indices """

    def fetch_sp500_symbols():
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        resp = requests.get(url)
        soup = pd.read_html(resp.text)
        return soup[0]['Symbol'].tolist()

    def fetch_nasdaq_100_symbols():
        headers = {"User-Agent": "Mozilla/5.0"}
        url = "https://api.nasdaq.com/api/quote/list-type/nasdaq100"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            main_data = response.json().get('data', {}).get('rows', [])
            return [item['symbol'] for item in main_data]
        return []

    return list(set(fetch_sp500_symbols() + fetch_nasdaq_100_symbols()))

def get_latest_file():
    """Find the most recent stock data file and extract the latest recorded date."""
    files = [f for f in os.listdir() if f.startswith("stock_data_") and f.endswith(".csv")]
    if not files:
        return None, None  # No previous files found

    files.sort(reverse=True, key=lambda x: x.split("_")[-1].split(".csv")[0])

    latest_file = files[0]
    try:
        df = pd.read_csv(latest_file, parse_dates=["Date"])
        latest_date = df["Date"].max()  # Get the most recent date in the data
        return latest_file, latest_date
    except Exception as e:
        print(f"Error reading {latest_file}: {e}")
        return None, None

def download_data(symbols):
    """Download OHLCV data only for missing dates and append to the latest dataset."""
    today = datetime.now().strftime("%Y%m%d")
    filename = f"stock_data_{today}.csv"

    # If today's file exists, exit
    if os.path.exists(filename):
        print(f"File for today already exists: {filename}")
        return filename

    # Get latest available data file
    latest_file, latest_date = get_latest_file()

    if latest_date is None:
        print("No existing data found. Downloading full dataset from 2000...")
        start_date = "2000-01-01"
    else:
        start_date = (latest_date + timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"Latest available date: {latest_date.strftime('%Y-%m-%d')}")
        print(f"Downloading missing data from {start_date} to today...")

    end_date = datetime.today().strftime("%Y-%m-%d")

    # Fetch missing data
    df_new = yf.download(symbols, start=start_date, end=end_date, group_by='ticker')

    # Identify failed tickers
    failed_tickers = [ticker for ticker in symbols if ticker not in df_new.columns.get_level_values(0)]
    if failed_tickers:
        print(f"Failed downloads: {failed_tickers}")

    # Debugging: Check if data was retrieved
    print(f'Data Downloaded: {df_new.shape[0]} rows, {df_new.shape[1]} columns')

    if df_new.empty:
        print("No new data retrieved.")
        return latest_file  # Return existing latest file

    # Convert MultiIndex into a flat DataFrame
    if isinstance(df_new.columns, pd.MultiIndex):
        df_new.columns = ['_'.join(col) for col in df_new.columns]  # Convert MultiIndex to single index

    df_new.reset_index(inplace=True)

    # Remove columns related to failed tickers
    for ticker in failed_tickers:
        df_new = df_new.loc[:, ~df_new.columns.str.startswith(f"{ticker}_")]

    # Load existing file if available
    if latest_file:
        df_existing = pd.read_csv(latest_file, parse_dates=["Date"])

        # Remove failed tickers from existing data
        for ticker in failed_tickers:
            df_existing = df_existing.loc[:, ~df_existing.columns.str.startswith(f"{ticker}_")]

        # Ensure existing data does NOT have 'Symbol' column
        if "Symbol" in df_existing.columns:
            df_existing = df_existing.drop(columns=["Symbol"])

        # Ensure new data does NOT have 'Symbol' column
        if "Symbol" in df_new.columns:
            df_new = df_new.drop(columns=["Symbol"])

        # Ensure column order matches before merging
        df_new = df_new[df_existing.columns]

        # Show tail of the last available data before merging
        print("\nExisting Data (Last 5 Rows):")
        print(df_existing.tail())

        # Show tail of newly downloaded data
        print("\nNew Data (Last 5 Rows):")
        print(df_new.tail())

        # Merge old and new data, removing duplicates
        df_combined = pd.concat([df_existing, df_new], ignore_index=True).drop_duplicates(subset=["Date"]).sort_values("Date")
    else:
        df_combined = df_new

    # Show tail of final merged dataset
    print("\nCombined Data (Last 5 Rows After Merging):")
    print(df_combined.tail())

    # Save updated data
    df_combined.to_csv(filename, index=False)
    print(f"Data saved correctly to {filename}")

    return filename

def load_data_from_file(filename):
    """ Load stock data from CSV into a dictionary with each ticker as a key. """
    try:
        df = pd.read_csv(filename, index_col=0, parse_dates=True)

        # Identify unique tickers from column names
        tickers = set(col.split('_')[0] for col in df.columns if '_' in col)

        data = {}

        for ticker in tickers:
            try:
                # Extract only this ticker's columns
                columns = [f"{ticker}_Open", f"{ticker}_High", f"{ticker}_Low", f"{ticker}_Close", f"{ticker}_Volume"]

                # Ensure only valid columns are selected
                columns = [col for col in columns if col in df.columns]

                if columns:
                    df_ticker = df[columns].copy()
                    df_ticker.columns = ['Open', 'High', 'Low', 'Close', 'Volume']  # Rename columns back to standard
                    data[ticker] = df_ticker.dropna()  # Drop missing values

            except Exception as e:
                print(f"Error processing {ticker}: {e}")

        print(f"Loaded data for {len(data)} tickers from {filename}")
        return data

    except Exception as e:
        print(f"Error loading file: {e}")
        return {}


class SMABacktester:
    """ SMA Backtesting strategy """

    def __init__(self, symbol, data, sma_s, sma_l):
        self.symbol = symbol
        self.data = data.copy()  # Avoid modifying original data
        self.SMA_S = sma_s
        self.SMA_L = sma_l
        self.results = None
        self.prepare_data()

    def prepare_data(self):
        """ Prepare the data by calculating log returns and SMAs """
        self.data["returns"] = np.log(self.data["Close"] / self.data["Close"].shift(1))
        self.data["SMA_S"] = self.data["Close"].rolling(self.SMA_S).mean()
        self.data["SMA_L"] = self.data["Close"].rolling(self.SMA_L).mean()
        self.data.dropna(inplace=True)

    def test_results(self):
        """ Backtests the SMA strategy and returns performance metrics """

        # Generate trading signals (1 = Buy, -1 = Sell)
        self.data["position"] = np.where(self.data["SMA_S"] > self.data["SMA_L"], 1, -1)
        self.data["strategy"] = self.data["returns"] * self.data["position"].shift(1)

        # Calculate strategy cumulative performance
        total_return = round(self.data["strategy"].sum(), 2)

        # Benchmark performance (buy & hold return)
        benchmark_return = round(self.data["returns"].sum(), 2)

        # Outperformance (strategy return - benchmark return)
        outperf = round(total_return - benchmark_return, 2)

        # Max Drawdown Calculation
        cum_returns = self.data["strategy"].cumsum()
        peak = cum_returns.cummax()
        drawdown = peak - cum_returns
        max_drawdown = round(drawdown.max(), 2)  # Worst drawdown observed

        # Determine last trade signal (Buy/Sell)
        last_signal = "Buy" if self.data["position"].iloc[-1] == 1 else "Sell"

        return outperf, total_return, max_drawdown, last_signal


class Optimizer:
    """ Optimization class for finding the best SMA parameters and generating buy/sell signals. """

    def __init__(self, symbols, data):
        self.symbols = symbols
        self.data = data

    def find_best_sma(self):
        """ Find the best SMA parameters for each symbol, optimizing for performance and drawdown. """
        results = []

        for symbol in self.symbols:
            print(f"Processing {symbol}...")  # Debugging Output

            if symbol not in self.data:
                print(f"Skipping {symbol} (No Data Found)")
                continue

            symbol_data = self.data[symbol]
            best_config = {
                "SMA_S": None, "SMA_L": None, "Performance": -float('inf'),
                "Max_Drawdown": float('inf'), "Last_Signal": None
            }

            try:
                for SMA_S in range(20, 51):
                    for SMA_L in range(100, 201, 10):
                        tester = SMABacktester(symbol, symbol_data.copy(), SMA_S, SMA_L)

                        # Unpack results from the backtester
                        outperf, perf, max_drawdown, signal = tester.test_results()

                        # Update best configuration based on performance and drawdown criteria
                        if outperf > best_config["Performance"] or (
                                outperf == best_config["Performance"] and max_drawdown < best_config["Max_Drawdown"]):
                            best_config.update({
                                "SMA_S": SMA_S, "SMA_L": SMA_L, "Performance": outperf,
                                "Max_Drawdown": max_drawdown, "Last_Signal": signal
                            })

            except Exception as e:
                print(f"Failed to process {symbol} due to error: {e}")
                continue

            # Append only if performance > 0.20 and drawdown > -0.50
            if best_config["Performance"] > 0.20 and best_config["Max_Drawdown"] > -0.50:
                results.append({
                    "Symbol": symbol,
                    "Best_SMA_S": best_config["SMA_S"],
                    "Best_SMA_L": best_config["SMA_L"],
                    "Max_Performance": best_config["Performance"],
                    "Max_Drawdown": best_config["Max_Drawdown"],
                    "Last_Signal": best_config["Last_Signal"]
                })

        # Convert results to a DataFrame
        results_df = pd.DataFrame(results)

        print("\nFinal Optimized Results:")
        print(results_df)  # Debugging Output

        return results_df




def run_code():
    symbols = fetch_symbols()
    filename = download_data(symbols)

    if filename:
        data = load_data_from_file(filename)
        optimizer = Optimizer(symbols, data)
        df_results = optimizer.find_best_sma()
        df_results.to_csv("optimized_results.csv", index=False)
        print("Results saved:", df_results)


if __name__ == "__main__":
    run_code()
