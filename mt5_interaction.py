import MetaTrader5 as mt5

def initialize_mt5(username, password, server, path):
    """Initialize MetaTrader 5."""
    mt5.initialize(path)
    authorized = mt5.login(login=int(username), password=password, server=server)
    return authorized

def get_market_price(symbol, trade_type):
    """Fetch the latest bid/ask price for a given trade type."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"❌ Failed to fetch tick data for {symbol}")
        return None
    return tick.ask if trade_type == "BUY" else tick.bid

def place_order(order_type, symbol, volume, stop_loss, take_profit, comment, magic_number):
    """Place an order in MetaTrader 5."""
    price = get_market_price(symbol, order_type)
    if price is None:
        return None

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": stop_loss,
        "tp": take_profit,
        "deviation": 10,
        "magic": magic_number,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    return result

def order_exists(symbol, trade_type, magic_number):
    """Check if an order of a given type already exists."""
    positions = mt5.positions_get(symbol=symbol)
    if positions:
        for pos in positions:
            if pos.magic == magic_number:
                if (trade_type == "BUY" and pos.type == mt5.ORDER_TYPE_BUY) or \
                   (trade_type == "SELL" and pos.type == mt5.ORDER_TYPE_SELL):
                    return True  # Trade already exists
    return False  # No matching trade found