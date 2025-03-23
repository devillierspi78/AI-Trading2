import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import requests



# 🔹 Market Status Check
def is_market_open():
    now = datetime.utcnow()
    return now.weekday() < 5  # Market is open Monday-Friday
 

#Modify calculate_sl_tp() to Use Bollinger Bands
def calculate_sl_tp2(symbol, base_atr_multiplier=2.0):
    """Dynamically adjusts SL/TP based on Bollinger Bands & ATR."""
    
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 100)  # 100 candles
    if rates is None:
        return None, None  # Fallback if data fetch fails

    df = pd.DataFrame(rates)

    # ATR Calculation
    df['high-low'] = df['high'] - df['low']
    df['high-close'] = np.abs(df['high'] - df['close'].shift(1))
    df['low-close'] = np.abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['high-low', 'high-close', 'low-close']].max(axis=1)
    atr = df['true_range'].rolling(window=14).mean().iloc[-1]

    # Bollinger Bands Calculation
    df['SMA'] = df['close'].rolling(window=20).mean()
    df['std_dev'] = df['close'].rolling(window=20).std()
    df['upper_band'] = df['SMA'] + (df['std_dev'] * 2)
    df['lower_band'] = df['SMA'] - (df['std_dev'] * 2)

    # Calculate Bollinger Band width (measures market volatility)
    bollinger_width = df['upper_band'].iloc[-1] - df['lower_band'].iloc[-1]
    
    # Adjust SL/TP dynamically
    dynamic_multiplier = 1 + (bollinger_width / df['SMA'].iloc[-1])  # Increase if volatility is high
    sl = atr * base_atr_multiplier * dynamic_multiplier
    tp = atr * base_atr_multiplier * 1.5 * dynamic_multiplier

    return sl, tp


#Get High impart news
def is_high_impact_news():
    """Check if a high-impact news event is near (e.g., NFP, FOMC)."""
    # Fetch news data from a reliable API (ForexFactory, Investing.com, etc.)
    news_events = fetch_forex_news()  # Replace with API function
    now = datetime.utcnow()

    for event in news_events:
        event_time = event["time"]  # Assuming API returns UTC timestamp
        if now - timedelta(minutes=30) < event_time < now + timedelta(minutes=30):
            return True  # High-impact news is within 30 minutes

    return False  # No major news nearby

#Fetch High-Impact News
def fetch_forex_news():
    """Fetch high-impact news from a forex economic calendar API."""
    try:
        response = requests.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json")
        if response.status_code != 200:
            print("❌ Failed to fetch news data.")
            return []

        news_data = response.json()
        high_impact_news = []

        for event in news_data:
            impact = event.get("impact", "").lower()
            if impact == "high":
                event_time = datetime.strptime(event["date"], "%Y-%m-%dT%H:%M:%S")
                if event_time > datetime.utcnow():  # Only consider future events
                    high_impact_news.append(event)

        return high_impact_news

    except Exception as e:
        print(f"❌ Error fetching news: {e}")
        return []