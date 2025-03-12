import datetime
import pandas as pd
import pytz
import time
from warnings import filterwarnings
import configparser
from alpaca_trade_api.rest import REST, APIError
import holidays

# Suppress warnings
filterwarnings("ignore")

def load_config(config_path='../Keys/config.ini'):
    """ Load API configuration settings. """
    config = configparser.ConfigParser()
    config.read(config_path)
    return {
        'api_key': config['alpaca']['api_key'],
        'api_secret': config['alpaca']['api_secret'],
        'base_url': 'https://paper-api.alpaca.markets'
    }

def init_api(config):
    """ Initialize and return the Alpaca API client. """
    return REST(config['api_key'], config['api_secret'], config['base_url'])

def fetch_historical_data(file_path="optimized_results.csv"):
    """ Load historical data from a CSV file. """
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print("No data file found")
        return pd.DataFrame()


def has_pending_orders(api, symbol):
    """ Check if there are pending orders for the symbol. """
    open_orders = api.list_orders(status='open')
    symbol_orders = [order for order in open_orders if order.symbol == symbol]

    if len(symbol_orders) > 0:
        print(f"Open orders detected for {symbol}: {len(symbol_orders)} orders pending.")

    return len(symbol_orders) > 0

def execute_trade(api, symbol, qty, side):
    """ Execute a trade order. """
    order_type = 'market'
    try:
        if qty > 0:
            print(f"{side.capitalize()}ing {qty} shares of {symbol}.")
            api.submit_order(symbol=symbol, qty=qty, side=side, type=order_type, time_in_force='gtc')
        else:
            print(f"Order skipped: Quantity is zero for {symbol}.")
    except APIError as e:
        print(f"API Error on {side} order for {symbol}: {e}")
        print(f"API Response: {e.response.content if hasattr(e, 'response') else 'No response'}")

def run_strategy(api, symbol, last_signal, funds_per_stock):
    """ Run strategy for a given symbol based on last recorded signal using the latest market price. """
    try:
        if has_pending_orders(api, symbol):
            print(f"Skipping {symbol} due to pending orders.")
            return

        try:
            position = api.get_position(symbol)
            position_qty = int(position.qty)
        except APIError as error:
            if 'position does not exist' in str(error):
                position_qty = 0
            else:
                print(f"Error retrieving position for {symbol}: {error}")
                return

        latest_quote = api.get_latest_quote(symbol)
        current_price = latest_quote.bp if latest_quote.bp else latest_quote.ap if latest_quote.ap else 0  # Fallback price

        # Fetch buying power
        account = api.get_account()
        buying_power = float(account.buying_power)

        print(f"{symbol}: Price = {current_price}, Position = {position_qty}, Signal = {last_signal}, Buying Power = ${buying_power:.2f}")

        if current_price == 0:
            print(f"No valid price for {symbol}, skipping trade.")
            return

        # Ensure funds_per_stock does not exceed actual buying power
        adjusted_funds = min(funds_per_stock, buying_power)
        max_qty = int(adjusted_funds / current_price)  # Calculate maximum affordable shares

        # **NEW LOGIC: Allow trading even if already in portfolio**
        if last_signal == 'Buy':
            if position_qty < 0:  # Cover a short position
                print(f"Covering {abs(position_qty)} shares of {symbol}.")
                execute_trade(api, symbol, abs(position_qty), 'buy')
            elif max_qty > 0:
                print(f"Buying {max_qty} shares of {symbol}.")
                execute_trade(api, symbol, max_qty, 'buy')
            else:
                print(f"Not enough funds to buy {symbol}.")

        elif last_signal == 'Sell':
            if position_qty > 0:  # Sell existing position
                print(f"Selling {position_qty} shares of {symbol}.")
                execute_trade(api, symbol, position_qty, 'sell')
            elif max_qty > 0:  # Short Sell new position
                print(f"Short Selling {max_qty} shares of {symbol}.")
                execute_trade(api, symbol, max_qty, 'sell')
            else:
                print(f"Not enough buying power to short {symbol}.")

        else:
            print(f"No trade executed for {symbol}. Signal: {last_signal}, Position: {position_qty}")

    except Exception as e:
        print(f"Error processing {symbol}: {e}")

def is_weekday(date):
    """Check if the date is a weekday."""
    return date.weekday() < 5  # Monday to Friday are < 5


def is_holiday(date):
    """Check if the date is a holiday."""
    us_holidays = holidays.US(years=date.year)
    return date in us_holidays


def get_next_market_open(now):
    """Calculate the next market open time considering weekends and holidays."""
    # Ensure 'now' is set with the correct hour and minute for comparison
    open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)

    # If it's past the opening time today or today isn't a trading day, move to the next day
    if now >= open_time or not is_weekday(now) or is_holiday(now):
        now += datetime.timedelta(days=1)
        now = now.replace(hour=9, minute=30, second=0, microsecond=0)
    else:
        # If it's not past the opening time and today is a trading day
        return open_time

    # Ensure we find the next valid trading day
    while not is_weekday(now) or is_holiday(now):
        now += datetime.timedelta(days=1)

    return now

def print_progress_bar(total_seconds):
    """Print a progress bar showing remaining time within square brackets with a rotating marker."""
    marker_positions = '|/-\\'  # Rotating marker symbols
    start_time = datetime.datetime.now()
    end_time = start_time + datetime.timedelta(seconds=total_seconds)

    while datetime.datetime.now() < end_time:
        elapsed_time = datetime.datetime.now() - start_time
        elapsed_seconds = elapsed_time.total_seconds()
        remaining_seconds = total_seconds - elapsed_seconds
        hours, remainder = divmod(remaining_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        progress_marker = marker_positions[int(elapsed_seconds) % len(marker_positions)]
        print(f"\r[{progress_marker}] Time remaining: {int(hours):02}:{int(minutes):02}h", end='')
        time.sleep(1)
    print()  # Print newline at the end

def check_account_value_and_close_positions(api):
    """Check account value and close all positions if it exceeds $100,000."""
    # Get account information
    account = api.get_account()
    equity = float(account.equity)

    # If account value exceeds $100,000, close all positions
    if equity > 100000:
        print(f"Portfolio value is ${equity:.2f}, exceeding $100,000. Closing all positions and stopping the bot.")

        # Close all open positions
        positions = api.list_positions()
        for position in positions:
            try:
                api.close_position(position.symbol)
                print(f"Closed position for {position.symbol}")
            except APIError as e:
                print(f"Failed to close position for {position.symbol}: {e}")

        print("All positions closed. Stopping the bot.")
        return True  # Signal to stop execution
    else:
        print(f"Portfolio Value is {equity:.2f}. Continuing trading.")
        return False  # Continue execution


def main():
    config = load_config()
    api = init_api(config)
    data = fetch_historical_data()
    total_cash = float(api.get_account().cash)
    buying_power = float(api.get_account().buying_power)
    print(f"Total Cash Available for Trading: ${total_cash:.2f}")

    # Check if account value exceeds $100,000 and stop if necessary
    if check_account_value_and_close_positions(api):
        exit(0)  # Stop execution

    if data.empty:
        print("No trading data available.")
        return

    # Get current open positions
    positions = api.list_positions()
    tickers_in_portfolio = {position.symbol: int(position.qty) for position in positions}  # Store quantity too

    # Get tickers from the CSV data
    tickers_in_data = set(data['Symbol'].unique())  # Unique tickers from dataset

    print(f"Total number of Tickers in Data: {len(tickers_in_data)}")
    print(f"Current Open Positions: {len(tickers_in_portfolio)}")

    # Separate tickers for buy and sell signals
    buy_tickers = set()
    sell_tickers = set()

    for index, row in data.iterrows():
        symbol = row['Symbol']
        last_signal = row['Last_Signal']

        if last_signal == "Buy":
            buy_tickers.add(symbol)
        elif last_signal == "Sell":
            sell_tickers.add(symbol)

    print(f"Buy Signals: {len(buy_tickers)} | Sell Signals: {len(sell_tickers)}")

    #  **Process only stocks that are currently held**
    if sell_tickers:
        print(f"Executing sell orders for {len(sell_tickers)} tickers...")

        for symbol in sell_tickers:
            if symbol in tickers_in_portfolio and tickers_in_portfolio[symbol] > 0:
                run_strategy(api, symbol, "Sell", 0)  # No funds needed for sell
            else:
                print(f"Skipping {symbol}: Not in portfolio or already fully sold.")

    #  **Process buy orders only if buying power is available**
    if buying_power > 0 and buy_tickers:
        remaining_tickers = buy_tickers - tickers_in_portfolio.keys()  # Only buy new stocks
        print(f"Buying Power Available: ${buying_power:.2f}")
        print(f"Potential New Buys: {len(remaining_tickers)}")

        if remaining_tickers:
            funds_per_stock = abs(buying_power * 0.9 / len(remaining_tickers))  # Allocate funds per new stock
            print(f"Allocated Funds Per Stock: ${funds_per_stock:.2f}")

            for symbol in remaining_tickers:
                run_strategy(api, symbol, "Buy", funds_per_stock)
        else:
            print("No new tickers to buy.")
    else:
        print("Insufficient buying power for new purchases. Skipping buy orders.")

    # Print daily account summary
    account = api.get_account()
    equity = float(account.equity)
    last_equity = float(account.last_equity)
    daily_change = round((equity - last_equity), 2)

    print(f"\nCurrent Time (HH:MM): {datetime.datetime.now().strftime('%H:%M')}")
    print(f"Daily Change: ${daily_change:.2f}")

def close_all_short_positions():
    """
    Fetches all open positions and closes only short positions by placing market buy orders.
    """
    try:
        positions = api.list_positions()
        short_positions = [pos for pos in positions if int(pos.qty) < 0]  # Find short positions

        if not short_positions:
            print("No short positions to close.")
            return

        print(f"Found {len(short_positions)} short positions. Closing them now...")

        for pos in short_positions:
            symbol = pos.symbol
            qty = abs(int(pos.qty))  # Convert negative qty to positive for buying
            print(f"Closing short position: {symbol} ({qty} shares)")

            # Submit market buy order to close the short position
            api.submit_order(
                symbol=symbol,
                qty=qty,
                side='buy',  # Buying to cover short position
                type='market',
                time_in_force='gtc'  # Good-Til-Canceled
            )
            print(f"✅ Order placed to buy {qty} shares of {symbol} (covering short position).")

    except Exception as e:
        print(f"Error closing short positions: {e}")

def sell_to_cover_cash_deficit(deficit_amount):
    """
    Sells long positions to cover the negative cash balance.
    """
    positions = api.list_positions()
    long_positions = [pos for pos in positions if int(pos.qty) > 0]  # Filter only long positions

    if not long_positions:
        print("No long positions to sell.")
        return

    print(f"Need to sell ${deficit_amount:.2f} worth of stocks to restore cash.")

    total_value = sum(float(pos.market_value) for pos in long_positions)
    sell_ratio = min(1, abs(deficit_amount) / total_value)  # Ensure we don't sell everything unnecessarily

    for pos in long_positions:
        symbol = pos.symbol
        qty = int(float(pos.qty) * sell_ratio)  # Calculate proportional qty to sell

        if qty > 0:
            print(f"Selling {qty} shares of {symbol} to cover margin debt.")
            api.submit_order(
                symbol=symbol,
                qty=qty,
                side='sell',
                type='market',
                time_in_force='gtc'
            )

    print("Orders placed to sell stocks and restore cash.")

def clear_negative_cash():
    """
    Checks cash balance and sells a small amount of stocks to fully restore cash to positive.
    """
    account = api.get_account()
    cash_balance = float(account.cash)  # Get current cash balance

    if cash_balance >= 0:
        print("Cash balance is already positive. No action needed.")
        return

    deficit = abs(cash_balance)
    print(f"Cash balance is negative (-${deficit:.2f}). Selling stocks to restore positive cash.")

    positions = api.list_positions()
    long_positions = [pos for pos in positions if int(pos.qty) > 0]

    if not long_positions:
        print("No stocks left to sell. Deposit funds to restore balance.")
        return

    for pos in long_positions:
        symbol = pos.symbol
        price = float(pos.current_price)
        qty_needed = int(deficit / price) + 1  # Calculate how many shares to sell

        if qty_needed > int(pos.qty):
            qty_needed = int(pos.qty)  # Don't oversell

        print(f"Selling {qty_needed} shares of {symbol} to restore cash balance.")
        api.submit_order(
            symbol=symbol,
            qty=qty_needed,
            side='sell',
            type='market',
            time_in_force='gtc'
        )

    print("Order placed to bring cash balance to positive.")

if __name__ == "__main__":
    config = load_config()
    api = init_api(config)
    clear_negative_cash()

    while True:
        try:
            ny_tz = pytz.timezone('America/New_York')
            now = datetime.datetime.now(ny_tz)
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

            current_time = datetime.datetime.now()
            formatted_time = current_time.strftime("%H:%M")
            print("\nCurrent Time (HH:MM): ", formatted_time,"\n\n")

            config = load_config()
            api = init_api(config)

            # Check account value before running the bot
            #if check_account_value_and_close_positions(api):
            #    break  # Stop execution

            if market_open <= now < market_close and is_weekday(now) and not is_holiday(now):
                print("Market is open. Running the trading bot...")
                main()
                time.sleep(300)  # Wait 5 min before checking again
            else:
                next_open = get_next_market_open(now)
                sleep_seconds = int((next_open - now).total_seconds())
                print(f"Market is closed. Next opening at {next_open.strftime('%Y-%m-%d %H:%M:%S %Z')}.")
                print_progress_bar(sleep_seconds)  # Sleep until the market opens graphically

        except KeyboardInterrupt:
            print("Script terminated by user.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            raise
