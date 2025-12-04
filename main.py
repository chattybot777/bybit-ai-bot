# === Bybit AI Trading Bot: Single-File Production Version ===
# Architecture: Monolithic (All-in-One) for Deployment Stability
# Logic: Risk-First Sizing + Gradient Boosting (GB) + RSI + Q-Learning (RL)
# Platform: Render.com

import os
import sys
import time
import math
import logging
import threading
import json
import numpy as np
import pandas as pd
import random
from http.server import BaseHTTPRequestHandler, HTTPServer
from dotenv import load_dotenv
from pybit.unified_trading import HTTP
from pybit.unified_trading.exceptions import InvalidRequestError
from tenacity import retry, stop_after_attempt, wait_exponential
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler

# --- CONFIGURATION & ENV VARS ---
load_dotenv()

# API Credentials (with aggressive cleaning for Render environment issues)
def clean_env(key):
    val = os.getenv(key, "")
    if not val: return ""
    return val.strip().replace('\n','').replace('\r','').replace(' ','')

API_KEY = clean_env("BYBIT_API_KEY")
API_SECRET = clean_env("BYBIT_API_SECRET")
TESTNET = os.getenv("TESTNET", "true").lower() == "true"

# Trading Settings
SYMBOL_LIST = ["BTCUSDT", "ETHUSDT", "SOLUSDT"] # Start with top 3 for stability
TIMEFRAME = "15" # 15-minute candles
RISK_PER_TRADE = 0.005 # 0.5% equity risk per trade
MAX_LEVERAGE = 10
# MIN_LEVERAGE = 1
SL_ATR_MULTIPLIER = 2.0 # Stop Loss distance = 2x ATR
TP_ATR_MULTIPLIER = 3.0 # Take Profit distance = 3x ATR

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- GLOBAL STATE ---
bot_state = {
    "equity": 0.0,
    "active_symbols": SYMBOL_LIST,
    "positions": {},
    "q_table": {}, # Stores RL knowledge
    "last_update": 0
}

# --- HEALTH SERVER (Keep Render Happy) ---
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        elif self.path == "/status":
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            # Convert non-serializable types for JSON
            safe_state = json.loads(json.dumps(bot_state, default=str))
            self.wfile.write(json.dumps(safe_state).encode())

def start_server():
    port = int(os.getenv("PORT", 10000))
    server = HTTPServer(('0.0.0.0', port), HealthHandler)
    logger.info(f"Health server started on port {port}")
    server.serve_forever()

# --- REINFORCEMENT LEARNING ENGINE (The Brain) ---
def get_market_state(volatility):
    # Classify market state based on volatility
    if volatility < 0.005: return 'low_vol'
    if volatility < 0.015: return 'med_vol'
    return 'high_vol'

def choose_leverage(symbol, state, epsilon=0.1):
    # Initialize Q-table for symbol if missing
    if symbol not in bot_state['q_table']:
        bot_state['q_table'][symbol] = {
            'low_vol': [0.0] * MAX_LEVERAGE,
            'med_vol': [0.0] * MAX_LEVERAGE,
            'high_vol': [0.0] * MAX_LEVERAGE
        }
    
    # Epsilon-Greedy Strategy: Exploration vs Exploitation
    if random.random() < epsilon:
        # Explore: Random leverage
        lev = random.randint(1, MAX_LEVERAGE)
    else:
        # Exploit: Best known leverage for this state
        q_values = bot_state['q_table'][symbol][state]
        lev = int(np.argmax(q_values) + 1)
    
    return lev

def update_q_table(symbol, state, leverage, reward):
    # Q-Learning Update Rule
    # Q(s,a) = Q(s,a) + alpha * (reward - Q(s,a))
    alpha = 0.1 # Learning rate
    
    if symbol in bot_state['q_table']:
        current_q = bot_state['q_table'][symbol][state][leverage-1]
        new_q = current_q + alpha * (reward - current_q)
        bot_state['q_table'][symbol][state][leverage-1] = new_q
        logger.info(f"RL UPDATE: {symbol} State={state} Lev={leverage}x Reward={reward:.4f} NewQ={new_q:.4f}")

# --- DATA & ML ENGINE ---
def calculate_indicators(df):
    # RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR (14) for Volatility & Risk
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    # Features for ML
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    
    return df.dropna()

def train_and_predict(df):
    # Prepare features for Gradient Boosting
    # We predict the *next* close price based on current features
    df['target'] = df['close'].shift(-1)
    data = df.dropna()
    
    if len(data) < 50:
        return 0 # Not enough data
        
    features = ['rsi', 'volatility', 'returns', 'atr']
    X = data[features].values
    y = data['target'].values
    
    # Simple Train/Test split (last row is what we want to predict for)
    X_train = X[:-1]
    y_train = y[:-1]
    X_current = X[-1].reshape(1, -1)
    
    model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    prediction = model.predict(X_current)[0]
    return prediction

# --- TRADING EXECUTION ---
class TradingBot:
    def __init__(self):
        self.session = HTTP(testnet=TESTNET, api_key=API_KEY, api_secret=API_SECRET)
        self.check_credentials()
        self.previous_equity = 0.0

    def check_credentials(self):
        try:
            # Simple call to verify keys
            self.session.get_server_time()
            logger.info("Credentials verified. Connected to Bybit.")
        except Exception as e:
            logger.critical(f"CREDENTIAL ERROR: {e}")
            sys.exit(1)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch_data(self, symbol):
        # Fetch 200 candles of 15m data
        r = self.session.get_kline(category="linear", symbol=symbol, interval=TIMEFRAME, limit=200)
        if r['retCode'] != 0:
            raise Exception(f"API Error: {r['retMsg']}")
            
        data = r['result']['list']
        # Bybit returns [time, open, high, low, close, volume, turnover]
        # We need to reverse it (API returns newest first)
        df = pd.DataFrame(data, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'turnover'])
        df = df.iloc[::-1].reset_index(drop=True)
        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
        return df

    def get_balance(self):
        try:
            r = self.session.get_wallet_balance(accountType="UNIFIED")
            total = float(r['result']['list'][0]['totalEquity'])
            bot_state['equity'] = total
            return total
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0.0

    def set_leverage_safe(self, symbol, leverage):
        try:
            self.session.set_leverage(category="linear", symbol=symbol, buy_leverage=str(leverage), sell_leverage=str(leverage))
        except InvalidRequestError as e:
            # Ignore "leverage not modified" error
            if "110043" not in str(e) and "34036" not in str(e):
                logger.warning(f"Leverage set warning: {e}")

    def execute_trade(self, symbol, side, entry_price, sl_dist, leverage):
        # RISK MANAGEMENT: Calculate Size based on Risk Amount
        equity = self.get_balance()
        risk_amt = equity * RISK_PER_TRADE
        
        # Quantity = Risk $ / Distance to Stop Loss $
        if sl_dist == 0: return
        
        qty = risk_amt / sl_dist
        
        # Format to 3 decimals 
        qty_str = f"{qty:.3f}"
        
        logger.info(f"SIGNAL: {symbol} {side} | Equity: {equity} | Risk: {risk_amt} | Qty: {qty_str} | Lev: {leverage}x")
        
        try:
            self.set_leverage_safe(symbol, leverage)
            self.session.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=qty_str,
                stopLoss=f"{(entry_price - sl_dist):.2f}" if side == "Buy" else f"{(entry_price + sl_dist):.2f}",
                takeProfit=f"{(entry_price + sl_dist * 1.5):.2f}" if side == "Buy" else f"{(entry_price - sl_dist * 1.5):.2f}"
            )
            logger.info(f"ORDER PLACED: {symbol} {side} {qty_str}")
            return True
        except Exception as e:
            logger.error(f"Order failed: {e}")
            return False

    def run(self):
        logger.info("Starting Main Trading Loop...")
        self.previous_equity = self.get_balance()
        
        while True:
            # 1. Check Portfolio Performance (Reward Calculation)
            current_equity = self.get_balance()
            pnl_change = current_equity - self.previous_equity
            self.previous_equity = current_equity
            
            # Simple reward attribution: Positive PnL = +1, Negative = -1
            # In a real system, you'd attribute this to specific closed trades,
            # but for a portfolio bot, total equity change is the ultimate truth.
            reward = 1.0 if pnl_change > 0 else (-1.0 if pnl_change < 0 else 0.0)

            for symbol in SYMBOL_LIST:
                try:
                    df = self.fetch_data(symbol)
                    df = calculate_indicators(df)
                    
                    current_price = df['close'].iloc[-1]
                    predicted_price = train_and_predict(df)
                    atr = df['atr'].iloc[-1]
                    rsi = df['rsi'].iloc[-1]
                    vol = df['volatility'].iloc[-1]
                    
                    # RL Step 1: Determine Market State
                    market_state = get_market_state(vol)
                    
                    # RL Step 2: Choose Leverage based on experience in this state
                    leverage = choose_leverage(symbol, market_state)
                    
                    # RL Step 3: Update Q-Table from PREVIOUS actions results (using Portfolio PnL)
                    # We update the *current* state/leverage pair with the recent global reward
                    # This is a simplification for stability: "Did the bot make money recently? Yes -> This lev is good."
                    if pnl_change != 0:
                        update_q_table(symbol, market_state, leverage, reward)

                    # SIGNAL LOGIC
                    threshold = current_price * 0.002 # 0.2% predicted move required
                    
                    if predicted_price > (current_price + threshold) and rsi < 70:
                        self.execute_trade(symbol, "Buy", current_price, atr * SL_ATR_MULTIPLIER, leverage)
                        
                    elif predicted_price < (current_price - threshold) and rsi > 30:
                        self.execute_trade(symbol, "Sell", current_price, atr * SL_ATR_MULTIPLIER, leverage)
                        
                    else:
                        logger.info(f"{symbol}: No Signal (Pred: {predicted_price:.2f} | Cur: {current_price:.2f})")
                        
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    
            # Update state for Health Server
            bot_state['last_update'] = time.time()
            
            # Sleep 60 seconds (1 minute) to avoid rate limits
            time.sleep(60)

if __name__ == "__main__":
    # Start Health Server in Background
    t = threading.Thread(target=start_server, daemon=True)
    t.start()
    
    # Start Bot
    bot = TradingBot()
    bot.run()