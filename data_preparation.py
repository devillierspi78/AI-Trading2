import MetaTrader5 as mt5
import pandas as pd
import talib

def prepare_market_data(symbol, lookback):
    data = get_multi_tf_data(symbol)
    
    # Merge different timeframes
    df = data["M5"]
    df = add_indicators(df)
    df = detect_engulfing(df)

    df.dropna(inplace=True)
    
    X, y, scaler = create_features_and_labels(df, lookback)
    return X, y, scaler

def get_multi_tf_data(symbol, timeframes=["M5", "H1", "D1"], n_bars=500):
    """Fetch market data from multiple timeframes."""
    all_data = {}

    timeframe_map = {
        "M5": mt5.TIMEFRAME_M5,
        "H1": mt5.TIMEFRAME_H1,
        "D1": mt5.TIMEFRAME_D1
    }

    for tf2 in timeframes:
        rates = mt5.copy_rates_from_pos(symbol, timeframe_map[tf2], 0, n_bars)

        if rates is not None:
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            all_data[tf2] = df

    return all_data

def add_indicators(df):
    """Add technical indicators to the dataset."""
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['close'])
    df['Upper_BB'], df['Middle_BB'], df['Lower_BB'] = talib.BBANDS(df['close'])
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    return df

def detect_engulfing(df):
    """Detect bullish & bearish engulfing candlestick patterns."""
    df['Bullish_Engulfing'] = (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1))
    df['Bearish_Engulfing'] = (df['close'] < df['open'].shift(1)) & (df['open'] > df['close'].shift(1))
    return df