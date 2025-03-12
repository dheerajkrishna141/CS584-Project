import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from warnings import filterwarnings
import requests
from datetime import datetime, timedelta,timezone
from random import randint
import pytz
import schedule
import time
import os


filterwarnings("ignore")

def fetch_symbols():
    """ Fetches symbols for NASDAQ-100 and S&P 500 indices """
    def fetch_nasdaq_100_symbols():
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
        }
        url = "https://api.nasdaq.com/api/quote/list-type/nasdaq100"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            main_data = response.json()['data']['data']['rows']
            return [item['symbol'] for item in main_data]
        else:
            print(f"Failed to fetch NASDAQ-100 data: HTTP {response.status_code}")
            return []

    def fetch_sp500_symbols():
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        resp = requests.get(url)
        soup = pd.read_html(resp.text)
        return soup[0]['Symbol'].tolist()

    nasdaq_100_symbols = fetch_nasdaq_100_symbols()
    sp500_symbols = fetch_sp500_symbols()
    combined_symbols = list(set(nasdaq_100_symbols + sp500_symbols))

    return combined_symbols

def download_data(symbols, start_date, end_date):
    """ Download data for a list of symbols, handling errors for delisted symbols gracefully. """
    data = {}
    for symbol in symbols:
            try:
                print(f"Downloading data for {symbol}...")
                df = yf.download(symbol, end=end_date, group_by='column')
                if df.empty:
                    print(f"No data found for {symbol}. It may be delisted.")
                else:
                    data[symbol] = df[['Close']]
            except Exception as e:
                print(f"Failed to download data for {symbol}. Error: {e}")
    return data


class SMABacktester:
    """ SMA Backtesting strategy """
    def __init__(self, symbol, data, sma_s, sma_l):
        self.symbol = symbol
        self.data = data
        self.SMA_S = sma_s
        self.SMA_L = sma_l
        self.results = None
        self.prepare_data()

    def prepare_data(self):
        """ Prepare data with SMA calculations """
        self.data["returns"] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        self.data["SMA_S"] = self.data['Close'].rolling(self.SMA_S).mean()
        self.data["SMA_L"] = self.data['Close'].rolling(self.SMA_L).mean()
        self.data.dropna(inplace=True)

    def test_results(self):
        """ Test SMA strategy performance and determine the last trading signal """
        self.data["position"] = np.where(self.data["SMA_S"] > self.data["SMA_L"], 1, -1)
        self.data["strategy"] = self.data["returns"] * self.data["position"].shift(1)
        self.data["cumulative_returns"] = self.data["returns"].cumsum().apply(np.exp)
        self.data["cumulative_strategy_returns"] = self.data["strategy"].cumsum().apply(np.exp)
        self.calculate_drawdowns()
        perf = self.data["cumulative_strategy_returns"].iloc[-1]
        max_drawdown = self.data["drawdown"].min()
        outperf = perf - self.data["cumulative_returns"].iloc[-1]
        self.results = self.data
        last_position = self.data['position'].iloc[-1]
        last_signal = "Buy" if last_position == 1 else "Sell"
        return round(outperf, 2), round(perf, 2), max_drawdown, last_signal

    def calculate_drawdowns(self):
        """ Calculate drawdowns and add to data """
        self.data["cumulative_max"] = self.data["cumulative_strategy_returns"].cummax()
        self.data["drawdown"] = self.data["cumulative_strategy_returns"] - self.data["cumulative_max"]
        self.data["drawdown"] /= self.data["cumulative_max"]


    def plot_results(self):
        """ Plot the backtesting results """
        if self.results is None:
            print("No results to plot. Please run the test.")
        else:
            title = f"{self.symbol} | SMA_S={self.SMA_S} | SMA_L={self.SMA_L}"
            self.results[["cumulative_returns", "cumulative_strategy_returns"]].plot(title=title, figsize=(12, 8))
            plt.show()


class Optimizer:
    """ Optimization class for finding the best SMA parameters """

    def __init__(self, symbols, data):
        self.symbols = symbols
        self.data = data

    def find_best_sma(self):
        """ Find the best SMA parameters for each symbol, considering both max performance and min drawdown. """
        results = []
        for symbol in self.symbols:
            print(f"Processing {symbol}...")
            if symbol not in self.data:
                print(f"Skipping {symbol} as no data is available.")
                continue
            symbol_data = self.data[symbol]
            best_config = {"SMA_S": None, "SMA_L": None, "Performance": -float('inf'), "Max_Drawdown": float('inf')}
            try:
                for SMA_S in range(20, 51):
                    for SMA_L in range(100, 201, 10):
                        tester = SMABacktester(symbol, symbol_data.copy(), SMA_S, SMA_L)
                        outperf, perf, max_drawdown, signal = tester.test_results()
                        # Update the best configuration if current performance is better,
                        # or if performance ties and drawdown is less severe.
                        if outperf > best_config["Performance"] or (
                                outperf == best_config["Performance"] and max_drawdown < best_config["Max_Drawdown"]):
                            best_config.update(
                                {"SMA_S": SMA_S, "SMA_L": SMA_L, "Performance": outperf, "Max_Drawdown": max_drawdown,
                                 "Last_Signal": signal})
            except Exception as e:
                print(f"Failed to process {symbol} due to error: {e}")
                continue

            # Append only if performance is greater than 0.20 and drawdown less severe than -0.50
            if best_config["Performance"] > 0.20 and best_config["Max_Drawdown"] > -0.50:
                results.append({
                    "Symbol": symbol,
                    "Best_SMA_S": best_config["SMA_S"],
                    "Best_SMA_L": best_config["SMA_L"],
                    "Max_Performance": best_config["Performance"],
                    "Max_Drawdown": best_config["Max_Drawdown"],
                    "Last_Signal": best_config["Last_Signal"]
                })
        return pd.DataFrame(results)


def run_code():
    symbols = fetch_symbols()
    random_number = randint(1, 25)
    start_date = datetime.today() - timedelta(days=365 * random_number)  # Adjusted for 25 years
    end_date = datetime.today()
    data = download_data(symbols, start_date, end_date)
    optimizer = Optimizer(symbols, data)
    df_results = optimizer.find_best_sma()

    file_path = "backtest_results_inter_day.csv"
    file_exists = os.path.exists(file_path)

    # Append mode ('a') used to continuously update the file without overwriting
    df_results.to_csv(file_path, mode='w', header=True ,  index=False)

    print(f"Results saved at {datetime.now()}:\n", df_results)

def get_local_timezone_stdlib():
    # Python 3.9 and newer
    local_tz_name = datetime.now(timezone.utc).astimezone().tzinfo
    return local_tz_name

def schedule_task():
    """ Schedules run_code to run daily at 09:00 AM EST """
    est = pytz.timezone('America/New_York')  # Timezone for Eastern Standard Time

    # Calculate next 09:00 AM EST
    now_est = datetime.now(est)  # Make 'now' offset-aware by setting the timezone to EST
    next_run = now_est.replace(hour=9, minute=0, second=0, microsecond=0)
    if now_est.hour >= 9 or (now_est.hour == 9 and now_est.minute > 0):  # Adjust for after 9:00 AM
        next_run += timedelta(days=1)


    # Set up a schedule to run at 9 AM EST
    schedule_time = next_run.strftime("%H:%M")
    schedule.every().day.at(schedule_time).do(run_code)
    print(f"Scheduled to run daily at {schedule_time} EST.")

    # Main loop to handle scheduling
    while True:
        now = datetime.now(est)  # Continuously update 'now' as offset-aware
        if now.hour >= 9 and now.minute >= 0 and now.second < 60:
            run_code()  # Ensure it runs if the script starts exactly at 9:00 AM
        time_until_next_run = int((next_run - now).total_seconds())
        time.sleep(max(0, time_until_next_run))  # Sleep until the next scheduled run time

if __name__ == "__main__":
    while True:#schedule_task()
        run_code()
        time.sleep(60*10)
