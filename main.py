import datetime
import pandas as pd
import pytz
import time
import os
from warnings import filterwarnings
import configparser
from alpaca_trade_api.rest import REST, APIError
import holidays

# Suppress warnings
filterwarnings("ignore")

def load_config(config_path='../../Keys/config.ini'):
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


def has_pending_orders(api, symbol, side=None):
    """ Check if there are pending orders for the symbol, optionally filtered by side. """
    open_orders = api.list_orders(status='open')
    for order in open_orders:
        if order.symbol == symbol and (side is None or order.side == side):
            print(f"Open order detected for {symbol} ({order.side.upper()})")
            return True
    return False

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

def log_limit_adjustment(symbol, old_signal, new_signal, reason):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "limit_adjustment_log.txt")

    with open(log_file, "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp} - {symbol} - Signal changed: {old_signal} → {new_signal} - {reason}\n")

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

def close_profitable_shorts_before_close():
    """
    At 3:55 PM, close only those short positions that are currently profitable.
    """
    try:
        positions = api.list_positions()
        short_positions = [pos for pos in positions if int(pos.qty) < 0]

        profitable_shorts = [
            pos for pos in short_positions if float(pos.unrealized_pl) > 0
        ]

        if not profitable_shorts:
            print("No profitable short positions to cover.")
            return

        print(f"📈 Found {len(profitable_shorts)} profitable shorts. Covering them now...")

        for pos in profitable_shorts:
            symbol = pos.symbol
            qty = abs(int(pos.qty))
            print(f"✅ Covering profitable short: {symbol} ({qty} shares) | P/L: ${float(pos.unrealized_pl):.2f}")

            api.submit_order(
                symbol=symbol,
                qty=qty,
                side='buy',
                type='market',
                time_in_force='gtc'
            )

    except Exception as e:
        print(f"Error covering profitable shorts: {e}")

def close_profitable_positions(api):
    """
    Closes all long or short positions with a positive unrealized profit.
    """
    try:
        positions = api.list_positions()
        profitable_positions = []

        for pos in positions:
            unrealized_pl = float(pos.unrealized_pl)
            qty = int(pos.qty)

            if unrealized_pl > 0 and qty != 0:
                profitable_positions.append((pos.symbol, qty, pos.side))

        if not profitable_positions:
            print("No profitable positions to close.")
            return

        print(f"Found {len(profitable_positions)} profitable positions. Closing them now...")

        for symbol, qty, side in profitable_positions:
            print(f"Closing profitable position: {symbol} ({qty} shares, {side})")
            try:
                api.close_position(symbol)
                print(f"✅ Closed {symbol}")
            except APIError as e:
                print(f"Failed to close {symbol}: {e}")

    except Exception as e:
        print(f"Error while closing profitable positions: {e}")

# Global tracker for original limit prices (symbol -> original limit price)
original_limit_prices = {}


def submit_limit_order(api, symbol, qty, side, current_price):
    """
    Submit a limit order with 5% offset from market price.
    Buy -> 5% below market price
    Sell -> 5% above market price
    """
    try:
        offset = 0.02
        limit_price = round(
            current_price * (1 - offset) if side == 'buy' else current_price * (1 + offset), 2
        )

        # Check for existing open orders to prevent duplicates
        open_orders = api.list_orders(status='open')
        for order in open_orders:
            if order.symbol == symbol and order.side == side:
                print(f"Skipping new order: Existing {side.upper()} order found for {symbol}")
                return

        api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='limit',
            time_in_force='gtc',
            limit_price=limit_price
        )

        print(f"Placed {side.upper()} LIMIT order: {qty} shares of {symbol} @ ${limit_price:.2f}")
        original_limit_prices[symbol] = limit_price

    except APIError as e:
        print(f"API Error placing {side.upper()} order for {symbol}: {e}")





def handle_signal_change(api, symbol, new_signal):
    open_orders = api.list_orders(status='open', symbols=[symbol])

    if not open_orders:
        return False  # No pending orders to check

    for order in open_orders:
        existing_signal = 'Buy' if order.side == 'buy' else 'Sell'

        if existing_signal.lower() != new_signal.lower():
            print(f"Signal changed for {symbol}: {existing_signal} → {new_signal}")

            # Cancel existing order
            try:
                api.cancel_order(order.id)
                print(f"Cancelled existing limit order for {symbol}.")
                log_limit_adjustment(symbol, existing_signal, new_signal, "Signal changed - order cancelled")
            except Exception as e:
                print(f"Failed to cancel order for {symbol}: {e}")

            return True  # Signal changed and order cancelled

    return False  # Signal was same as pending order

def adjust_limit_orders(api):
    """
    Adjust open limit orders once, 5 minutes before market close, if market price
    has deviated more than 5% from the original limit price.
    """
    log_entries = []
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    open_orders = api.list_orders(status='open')
    for order in open_orders:
        symbol = order.symbol
        side = order.side
        qty = int(order.qty)

        if symbol not in original_limit_prices:
            continue

        original_price = float(original_limit_prices[symbol])

        try:
            quote = api.get_latest_quote(symbol)
            market_price = float(quote.bp or quote.ap or 0)
            if market_price == 0:
                continue

            deviation = abs(original_price - market_price) / market_price
            if deviation >= 0.05:
                # Cancel and replace
                api.cancel_order(order.id)
                time.sleep(1)

                new_limit = round(
                    market_price * (1 - 0.02) if side == 'buy' else market_price * (1 + 0.02), 2
                )

                api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    type='limit',
                    time_in_force='gtc',
                    limit_price=new_limit
                )

                original_limit_prices[symbol] = new_limit

                log_line = (
                    f"[{now_str}] Adjusted {side.upper()} LIMIT for {symbol} from ${original_price:.2f} "
                    f"to ${new_limit:.2f} (Market: ${market_price:.2f})"
                )
                print(log_line)
                log_entries.append(log_line)

        except Exception as e:
            print(f"Error adjusting order for {symbol}: {e}")

    if log_entries:
        with open("limit_adjustment_log.txt", "a") as f:
            for line in log_entries:
                f.write(line + "\n")

def run_strategy(api, symbol, last_signal, funds_per_stock):
    """Run strategy for a given symbol based on last recorded signal."""
    try:
        side = 'buy' if last_signal.lower() == 'buy' else 'sell'

        if has_pending_orders(api, symbol, side):
            print(f"{symbol}: Skipping {side.upper()} — existing {side.upper()} order is pending.")
            return

        # Get position and price info
        try:
            position = api.get_position(symbol)
            position_qty = int(position.qty)
        except APIError as error:
            if 'position does not exist' in str(error).lower():
                position_qty = 0
            else:
                print(f"Error retrieving position for {symbol}: {error}")
                return

        quote = api.get_latest_quote(symbol)
        current_price = quote.bp or quote.ap or 0
        account = api.get_account()
        cash = float(account.cash)
        buying_power = float(account.buying_power)

        print(f"{symbol}: Price=${current_price:.2f}, Pos={position_qty}, Signal={last_signal}, Cash=${cash:.2f}, BP=${buying_power:.2f}")

        if current_price == 0:
            print(f"Skipping {symbol}: No valid price.")
            return

        # Determine order side, limit price, and quantity
        if last_signal == 'Buy':
            limit_price = round(current_price * 0.98, 2)
            if position_qty < 0:
                print(f"Covering short position: {abs(position_qty)} shares of {symbol}")
                qty = abs(position_qty)
            else:
                adjusted_funds = min(funds_per_stock, cash)
                qty = int(adjusted_funds / limit_price)

        elif last_signal == 'Sell':
            limit_price = round(current_price * 1.02, 2)
            if position_qty > 0:
                print(f"Selling long position: {position_qty} shares of {symbol}")
                qty = position_qty
            elif position_qty == 0:
                adjusted_funds = min(funds_per_stock, buying_power)
                qty = int(adjusted_funds / limit_price)
            else:
                print(f"{symbol}: Already shorted. Holding.")
                return
        else:
            print(f"Unknown signal type '{last_signal}' for {symbol}.")
            return

        if qty < 1:
            print(f"Skipping {symbol}: Quantity too small.")
            return

        print(f"Placing {side.upper()} LIMIT order for {symbol}: {qty} shares @ ${limit_price:.2f} (60-day GTC)")

        api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='limit',
            limit_price=limit_price,
            time_in_force='gtc'
        )

    except Exception as e:
        print(f"Error running strategy for {symbol}: {e}")



def main():
    data = fetch_historical_data()
    account = api.get_account()
    cash_available = float(account.cash)
    buying_power = float(account.buying_power)

    print(f"Total Cash Available for Trading: ${cash_available:.2f}")
    print(f"Total Buying Power: ${buying_power:.2f}")

    # Stop if account too large
    #if check_account_value_and_close_positions(api):
    #    exit(0)

    if data.empty:
        print("No trading data available.")
        return

    # Get current positions
    positions = api.list_positions()
    tickers_in_portfolio = {}
    for position in positions:
        tickers_in_portfolio[position.symbol] = int(position.qty)

    tickers_in_data = set(data['Symbol'].unique())
    print(f"Total Tickers in Data: {len(tickers_in_data)}")
    print(f"Open Positions: {len(tickers_in_portfolio)}")

    # Categorize signals
    buy_tickers = set()
    sell_tickers = set()

    for _, row in data.iterrows():
        symbol = row['Symbol']
        signal = row['Last_Signal']
        if signal == "Buy":
            buy_tickers.add(symbol)
        elif signal == "Sell":
            sell_tickers.add(symbol)

    print(f"Buy Signals: {len(buy_tickers)} | Sell Signals: {len(sell_tickers)}")

    # ---- SELL & SHORT ----
    if sell_tickers:
        print(f"\nProcessing {len(sell_tickers)} SELL/SHORT signals...")

        new_shorts = sell_tickers - set(tickers_in_portfolio.keys())
        num_new_shorts = len(new_shorts)
        per_short_funds = (buying_power * 0.9 / num_new_shorts) if num_new_shorts > 0 else 0

        for symbol in sell_tickers:
            if symbol in tickers_in_portfolio and tickers_in_portfolio[symbol] != 0:
                run_strategy(api, symbol, "Sell", 0)  # Just liquidate held positions
            else:
                if has_pending_orders(api, symbol):
                    print(f"Skipping {symbol}: Already has a pending order.")
                    continue

                try:
                    quote = api.get_latest_quote(symbol)
                    current_price = quote.bp or quote.ap or 0
                    if current_price == 0:
                        print(f"Skipping {symbol}: No valid market price.")
                        continue

                    limit_price = round(current_price * 1.05, 2)  # for shorting
                    if per_short_funds < limit_price:
                        print(f"Skipping {symbol}: Not enough buying power to short even 1 share at ${limit_price:.2f}")
                        continue

                    print(f"{symbol} not in portfolio. Allocating ${per_short_funds:.2f} for shorting.")
                    run_strategy(api, symbol, "Sell", per_short_funds)

                except Exception as e:
                    print(f"Error preparing short for {symbol}: {e}")

    # ---- BUY ----
    if cash_available > 0 and buy_tickers:
        new_buys = buy_tickers - tickers_in_portfolio.keys()
        print(f"\nProcessing {len(new_buys)} BUY signals (new)...")

        # Use the actual spendable amount (cash + valid buying power)
        effective_buying_power = min(cash_available, buying_power)

        if effective_buying_power < 100:  # Minimum needed to attempt 1 or 2 shares
            print(f"Skipping buys: Effective buying power is too low (${effective_buying_power:.2f})")
        elif new_buys:
            per_buy_funds = effective_buying_power * 0.9 / len(new_buys)
            print(f"Allocated Funds Per BUY Stock: ${per_buy_funds:.2f}")
            for symbol in new_buys:
                run_strategy(api, symbol, "Buy", per_buy_funds)
        else:
            print("No new tickers to buy.")
    else:
        print("Insufficient cash for buys. Skipping new buy orders.")

    # ---- Daily Summary ----
    equity = float(account.equity)
    last_equity = float(account.last_equity)
    delta = round(equity - last_equity, 2)

    print(f"\nCurrent Time: {datetime.datetime.now().strftime('%H:%M')}")
    print(f"Daily P/L Change: ${delta:.2f}")

if __name__ == "__main__":
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

            if market_open <= now < market_close and is_weekday(now) and not is_holiday(now):
                print("Market is open. Running the trading bot...")
                main()
                if now >= market_close - datetime.timedelta(minutes=5):
                    print("\nChecking for limit order adjustments (5 mins before close)...")
                    adjust_limit_orders(api)
                    print("\n⏱️ 3:55 PM ET: Closing profitable positions and restoring cash...")
                    close_profitable_positions(api)
                    print("\nClosing negative cash positions before market close (3:55 PM ET)...")
                    clear_negative_cash()
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
