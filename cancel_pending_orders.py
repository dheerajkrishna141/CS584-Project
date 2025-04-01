import configparser
from alpaca_trade_api.rest import REST, APIError

def load_config(config_path='../../Keys/config.ini'):
    config = configparser.ConfigParser()
    config.read(config_path)
    return {
        'api_key': config['alpaca']['api_key'],
        'api_secret': config['alpaca']['api_secret'],
        'base_url': 'https://paper-api.alpaca.markets'
    }

def cancel_all_pending_orders():
    config = load_config()
    api = REST(config['api_key'], config['api_secret'], config['base_url'])

    try:
        open_orders = api.list_orders(status='open')
        if not open_orders:
            print("✅ No pending orders found.")
            return

        print(f"Found {len(open_orders)} open order(s). Cancelling them...")

        for order in open_orders:
            try:
                api.cancel_order(order.id)
                print(f"❌ Cancelled order: {order.symbol} ({order.side.upper()} {order.qty} @ {order.type})")
            except APIError as e:
                print(f"⚠️ Failed to cancel order for {order.symbol}: {e}")

        print("✅ All pending orders processed.")
    except Exception as e:
        print(f"❌ Error fetching/cancelling orders: {e}")

if __name__ == "__main__":
    cancel_all_pending_orders()
