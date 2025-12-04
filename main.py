# === Bybit AI Trading Bot: Single-File Production Version ===
# Architecture: Monolithic (All-in-One) for Deployment Stability
# Logic: Risk-First Sizing + Gradient Boosting (GB) + RSI + Q-Learning
# Fixes: Minimum Order Size Enforcement + Correct Imports
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
# FIX: Correct import path for pybit exceptions
from pybit.exceptions import InvalidRequestError
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

# Strategy Settings
SYMBOL_LIST = ["BTCUSDT", "ETHUSDT", "SOLUSDT"] 
TIMEFRAME = "15" # 15-minute candles
RISK_PER_TRADE = 0.005 # 0.5% equity risk per trade
MAX_LEVERAGE = 10
SL_ATR_MULTIPLIER = 2.0 
TP_ATR_MULTIPLIER = 3.0

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- GLOBAL STATE ---
bot_state = {
    "equity": 0.0,
    "active_symbols": SYMBOL_LIST,
    "positions": {},
    "q_table": {},
    "last_update": 0
}

# --- HEALTH SERVER ---
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200); self.end_headers(); self.wfile.write(b"OK")
        elif self.path == "/status":
            self.send_response(200); self.send_header('Content-type', 'application/json'); self.end_headers()
            self.wfile.write(json.dumps(json.loads(json.dumps(bot_state, default=str))).encode())

def start_server():
    port = int(os.getenv("PORT", 10000))
    HTTPServer(('0.0.0.0', port), HealthHandler).serve_forever()

# --- REINFORCEMENT LEARNING ---
def get_market_state(volatility):
    if volatility < 0.005: return 'low_vol'
    if volatility < 0.015: return 'med_vol'
    return 'high_vol'

def choose_leverage(symbol, state, epsilon=0.1):
    if symbol not in bot_state['q_table']:
        bot_state['q_table'][symbol] = {'low_vol': [0.0]*10, 'med_vol': [0.0]*10, 'high_vol': [0.0]*10}
    
    if random.random() < epsilon:
        return random.randint(1, MAX_LEVERAGE) # Explore
    else:
        return int(np.argmax(bot_state['q_table'][symbol][state]) + 1)

def update_q_table(symbol, state, leverage, reward):
    alpha = 0.1 # Learning rate
    if symbol in bot_state['q_table']:
        idx = max(0, min(leverage - 1, 9))
        current_q = bot_state['q_table'][symbol][state][idx]
        new_q = current_q + alpha * (reward - current_q)
        bot_state['q_table'][symbol][state][idx] = new_q

# --- DATA & ML ENGINE ---
def calculate_indicators(df):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    df['atr'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()
    
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    return df.dropna()

def train_and_predict(df):
    df['target'] = df['close'].shift(-1)
    current_features = df.iloc[-1][['rsi', 'volatility', 'returns', 'atr']].values.reshape(1, -1)
    train_data = df.dropna()
    
    if len(train_data) < 50: return 0 
    
    X = train_data[['rsi', 'volatility', 'returns', 'atr']].values
    y = train_data['target'].values
    
    model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X, y)
    
    return model.predict(current_features)[0]

# --- TRADING BOT ---
class TradingBot:
    def __init__(self):
        self.session = HTTP(testnet=TESTNET, api_key=API_KEY, api_secret=API_SECRET)
        self.symbol_info = {} # Stores min qty rules
        self.previous_equity = 0.0
        try:
            self.session.get_server_time()
            logger.info("Connected to Bybit.")
            self.load_instrument_info()
        except Exception as e:
            logger.critical(f"Auth Failed: {e}"); sys.exit(1)

    def load_instrument_info(self):
        # Fetch rules for all symbols (min qty, precision, etc)
        try:
            r = self.session.get_instruments_info(category="linear")
            for i in r['result']['list']:
                if i['symbol'] in SYMBOL_LIST:
                    self.symbol_info[i['symbol']] = {
                        'min_qty': float(i['lotSizeFilter']['minOrderQty']),
                        'qty_step': float(i['lotSizeFilter']['qtyStep']),
                    }
            logger.info(f"Loaded rules for {len(self.symbol_info)} symbols.")
        except Exception as e:
            logger.error(f"Failed to load instrument info: {e}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch_data(self, symbol):
        r = self.session.get_kline(category="linear", symbol=symbol, interval=TIMEFRAME, limit=200)
        if r['retCode'] != 0: raise Exception(r['retMsg'])
        df = pd.DataFrame(r['result']['list'], columns=['ts','open','high','low','close','vol','to'])
        df = df.iloc[::-1].reset_index(drop=True)
        return df[['open','high','low','close']].astype(float)

    def get_equity(self):
        r = self.session.get_wallet_balance(accountType="UNIFIED")
        return float(r['result']['list'][0]['totalEquity'])

    def execute(self, symbol, side, price, sl_dist, lev):
        equity = self.get_equity()
        bot_state['equity'] = equity
        risk_usd = equity * RISK_PER_TRADE
        if sl_dist <= 0: return
        
        # 1. Calculate ideal quantity based on risk
        raw_qty = risk_usd / sl_dist
        
        # 2. Enforce Minimum Order Quantity (The Fix)
        info = self.symbol_info.get(symbol)
        if info:
            min_qty = info['min_qty']
            step = info['qty_step']
            
            # If calculated size is too small, bump it to the minimum
            if raw_qty < min_qty:
                # logger.warning(f"{symbol}: Risk qty {raw_qty:.4f} too small. Bumping to min {min_qty}")
                raw_qty = min_qty
            
            # Round to correct step (e.g. 0.001)
            qty = math.floor(raw_qty / step) * step
            # Safe formatting for crypto
            decimals = str(step)[::-1].find('.')
            if decimals < 0: decimals = 0
            qty_str = f"{qty:.{decimals}f}"
        else:
            qty_str = f"{raw_qty:.3f}"

        logger.info(f"TRADE: {symbol} {side} | Lev: {lev}x | Qty: {qty_str} | Risk: ${risk_usd:.2f}")
        
        try:
            try: self.session.set_leverage(category="linear", symbol=symbol, buy_leverage=str(lev), sell_leverage=str(lev))
            except: pass
            
            sl_price = price - sl_dist if side == "Buy" else price + sl_dist
            tp_price = price + (sl_dist * TP_ATR_MULTIPLIER) if side == "Buy" else price - (sl_dist * TP_ATR_MULTIPLIER)
            
            self.session.place_order(
                category="linear", symbol=symbol, side=side, orderType="Market", qty=qty_str,
                stopLoss=f"{sl_price:.2f}", takeProfit=f"{tp_price:.2f}"
            )
        except Exception as e:
            logger.error(f"Execution Failed: {e}")

    def run(self):
        logger.info("Starting Main Trading Loop...")
        self.previous_equity = self.get_equity()
        while True:
            curr_eq = self.get_equity()
            reward = 1.0 if (curr_eq - self.previous_equity) > 0 else -1.0
            self.previous_equity = curr_eq
            
            for sym in SYMBOL_LIST:
                try:
                    df = self.fetch_data(sym)
                    df = calculate_indicators(df)
                    price = df['close'].iloc[-1]
                    pred = train_and_predict(df)
                    atr = df['atr'].iloc[-1]
                    vol = df['volatility'].iloc[-1]
                    rsi = df['rsi'].iloc[-1]
                    
                    state = get_market_state(vol)
                    lev = choose_leverage(sym, state)
                    if reward != 0: update_q_table(sym, state, lev, reward)
                    
                    if pred > (price * 1.002) and rsi < 70:
                        self.execute(sym, "Buy", price, atr * SL_ATR_MULTIPLIER, lev)
                    elif pred < (price * 0.998) and rsi > 30:
                        self.execute(sym, "Sell", price, atr * SL_ATR_MULTIPLIER, lev)
                    else:
                        logger.info(f"{sym}: Wait. Pred:{pred:.2f} Cur:{price:.2f}")
                except Exception as e:
                    logger.error(f"Loop Error {sym}: {e}")
            
            time.sleep(15)

if __name__ == "__main__":
    threading.Thread(target=start_server, daemon=True).start()
    TradingBot().run()