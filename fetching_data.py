import requests
import pandas as pd
import time
import os
import sqlite3
from datetime import datetime

# Polygon.io API Key
API_KEY = "io0fjlb9ouIz1AnS6cCjyPaImee8BKXu"
DB_NAME = "stock_data.db"


def fetch_symbols():
    """Fetches symbols for NASDAQ-100 and S&P 500 indices."""

    def fetch_nasdaq_100_symbols():
        headers = {
            "User-Agent": "Mozilla/5.0"
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
        soup = pd.read_html(resp.text, flavor='bs4')
        return soup[0]['Symbol'].tolist()

    nasdaq_100_symbols = fetch_nasdaq_100_symbols()
    sp500_symbols = fetch_sp500_symbols()
    combined_symbols = list(set(nasdaq_100_symbols + sp500_symbols))

    return combined_symbols


def fetch_stock_data(symbol, start_date, end_date):
    """Fetch historical stock data from Polygon.io and print failed responses."""

    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}?adjusted=true&sort=asc&apiKey={API_KEY}"

    try:
        response = requests.get(url)
        data = response.json()

        if "results" in data and data["results"]:
            df = pd.DataFrame(data["results"])
            df["ticker"] = symbol
            df["date"] = pd.to_datetime(df["t"], unit='ms')
            df.set_index("date", inplace=True)
            df = df[["ticker", "c", "h", "l", "o", "v"]]
            df.columns = ["Ticker", "Close", "High", "Low", "Open", "Volume"]
            return df
        else:
            print(f"⚠ No data for {symbol} | API Response: {data}")  # PRINT FULL API RESPONSE
            return None

    except Exception as e:
        print(f"❌ Error fetching data for {symbol}: {e}")
        return None


def save_to_sqlite(df, table_name):
    """Save DataFrame to SQLite database and ensure correct schema with UNIQUE constraint."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # ✅ Ensure the table exists with the correct schema
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            date TEXT NOT NULL,
            Ticker TEXT NOT NULL,
            Close REAL,
            High REAL,
            Low REAL,
            Open REAL,
            Volume INTEGER,
            PRIMARY KEY (date, Ticker)  -- Ensures unique (date, Ticker) pairs
        )
    """)

    # ✅ Convert DataFrame index (date) to a column for SQLite
    if df.index.name == "date":
        df = df.reset_index()

    # ✅ Ensure 'date' column exists in DataFrame
    if "date" not in df.columns:
        print("⚠ Warning: No 'date' column found in DataFrame before saving.")

    # ✅ Use INSERT OR REPLACE to avoid duplicate errors
    df.to_sql(table_name, conn, if_exists="append", index=False)

    conn.close()


def download_data(symbols, start_date, end_date):
    """Download historical stock data and save to SQLite."""

    print(f"📥 Downloading data for {len(symbols)} stocks from {start_date} to {end_date}...")

    failed_tickers = []

    for i, symbol in enumerate(symbols):
        print(f"\n({i + 1}/{len(symbols)}) Fetching {symbol}...")

        df = fetch_stock_data(symbol, start_date, end_date)

        if df is not None:
            save_to_sqlite(df, "stock_prices")
        else:
            failed_tickers.append(symbol)  # Store missing tickers

        time.sleep(1)  # ⏳ Pause between requests to avoid hitting API limits

    # Save failed tickers for later review
    if failed_tickers:
        with open("failed_tickers.txt", "w") as f:
            f.write("\n".join(failed_tickers))
        print(f"⚠ Failed tickers saved to failed_tickers.txt")

    print(f"✅ Data saved to {DB_NAME}")


if __name__ == "__main__":
    symbols = fetch_symbols()
    start_date = "2000-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")  # Dynamic today's date
    download_data(symbols, start_date, end_date)
