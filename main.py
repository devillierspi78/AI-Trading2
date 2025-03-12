import os
import sys
import time
import schedule
import MetaTrader5 as mt5
from tensorflow_probability.python.distributions import kullback_leibler

import mt5_interaction as mty
from data_preparation import prepare_market_data
from ai_models import train_and_forecast_bayesian_model, execute_ai_trade, run_backtesting
from utils import ensure_model_path_exists, is_market_open

# Set the environment variable
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Portfolio symbols for AI-Based Trading Strategies
TRADING_ASSETS = ["EURUSD", "XAUUSD", "USDJPY", "AUDUSD"]

# AI-Based Trading Parameters
LOOKBACK_WINDOW = 50
LEARNING_RATE = 0.001
EPOCHS = 10
BATCH_SIZE = 32
EXECUTION_SPEED = 0.1

BNN_MAGIC_NUMBER = 123456
QUANTUM_MAGIC_NUMBER = 654321

# Initialize MetaTrader 5
if not mt5.initialize_mt5(username='7083958', password='1234qwer!@#$QWER', server='FPMarketsSC-Demo', path='C:\\Program Files\\FP Markets MT5 Terminal\\terminal64.exe'):
    print("❌ MT5 initialization failed. Exiting...")
    mt5.shutdown()
    sys.exit()
else:
    print("MT5 initialization successful")

# Schedule the trading and backtesting tasks
schedule.every(0.1).seconds.do(execute_ai_trade)
schedule.every(1).minutes.do(train_and_forecast_bayesian_model)
schedule.every().day.at("00:00").do(run_backtesting)

if __name__ == "__main__":
    print("🚀 AI Trading Bot Running...")
    while True:
        schedule.run_pending()
        time.sleep(EXECUTION_SPEED)