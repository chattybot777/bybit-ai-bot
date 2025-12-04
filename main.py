# === Bybit AI Trading Bot: Single-File Production Version ===
# Architecture: Monolithic (All-in-One) for Deployment Stability
# Logic: Risk-First Sizing + Gradient Boosting (Live Training) + RSI + Q-Learning
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
from pybit.exceptions import InvalidRequestError
from tenacity import retry, stop_after_attempt, wait_exponential
from sklearn.ensemble import GradientBoostingRegressor

# --- CONFIGURATION ---
load_dotenv()
API_KEY = os.getenv("BYBIT_API_KEY", "").strip()
API_SECRET = os.getenv("BYBIT_API_SECRET", "").strip()
TESTNET = os.getenv("TESTNET", "true").lower() == "true"
SYMBOL_LIST = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
TIMEFRAME = "15"
RISK_PER_TRADE = 0.005 # 0.5% Risk
MAX_LEVERAGE = 10

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- STATE ---
bot_state = {"equity": 0.0, "active_symbols": SYMBOL_LIST, "q_table": {}, "instrument_info": {}}

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

# --- AI & STRATEGY ---
def get_market_state(vol):
    if vol < 0.005: return 'low'
    if volatility < 0.015: return 'med'
    return 'high'

def choose_leverage(symbol, state):
    if symbol not in bot_state['q_table']:
        bot_state['q_table'][symbol] = {'low': [0.0]*10, 'med': [0.0]*10, 'high': [0.0]*10}
    
    # Simple Epsilon-Greedy
    if random.random() < 0.1: return random.randint(1, MAX_LEVERAGE)
    return int(np.argmax(bot_state['q_table'][symbol][state]) + 1)

def update_q(symbol, state, lev, reward):
    if symbol in bot_state['q_table']:
        idx = max(0, min(lev - 1, 9))
        curr = bot_state['q_table'][symbol][state][idx]
        bot_state['q_table'][symbol][state][idx] = curr + 0.1 * (reward - curr)

def prepare_data(df):
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low'] - df['close'].shift()).abs()
    df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    
    return df.dropna()

def predict_direction(df):
    # Predict next close using Gradient Boosting
    df['target'] = df['close'].shift(-1)
    data = df.dropna()
    if len(data) < 50: return 0
    
    feat_cols = ['rsi', 'volatility', 'returns', 'atr']
    model = GradientBoostingRegressor(n_estimators=50, max_depth=3)
    model.fit(data[feat_cols], data['target'])
    
    last_row = df.iloc[-1][feat_cols].values.reshape(1, -1)
    return model.predict(last_row)[0]

# --- EXECUTION ---
class TradingBot:
    def __init__(self):
        self.session = HTTP(testnet=TESTNET, api_key=API_KEY, api_secret=API_SECRET)
        self.prev_eq = 0.0
        try: 
            self.session.get_server_time()
            logger.info("Connected to Bybit.")
            self.load_instrument_info()
        except Exception as e: 
            logger.critical(f"Auth Error: {e}"); sys.exit(1)

    def load_instrument_info(self):
        try:
            # Dynamically fetch minimum order quantities to prevent 10001 errors
            r = self.session.get_instruments_info(category="linear")
            for i in r['result']['list']:
                if i['symbol'] in SYMBOL_LIST:
                    bot_state['instrument_info'][i['symbol']] = {
                        'min_qty': float(i['lotSizeFilter']['minOrderQty']),
                        'qty_step': float(i['lotSizeFilter']['qtyStep'])
                    }
            logger.info("Instrument Info Loaded.")
        except Exception as e:
            logger.error(f"Instrument Info Failed: {e}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_data(self, sym):
        r = self.session.get_kline(category="linear", symbol=sym, interval=TIMEFRAME, limit=200)
        df = pd.DataFrame(r['result']['list'], columns=['ts','open','high','low','close','v','t'])
        df = df.iloc[::-1].reset_index(drop=True)
        return df[['open','high','low','close']].astype(float)

    def run(self):
        # Initial Balance
        try:
            bal = self.session.get_wallet_balance(accountType="UNIFIED")
            self.prev_eq = float(bal['result']['list'][0]['totalEquity'])
        except: self.prev_eq = 1000.0

        while True:
            # Update Equity & Reward
            try:
                bal = self.session.get_wallet_balance(accountType="UNIFIED")
                curr_eq = float(bal['result']['list'][0]['totalEquity'])
                bot_state['equity'] = curr_eq
                reward = 1.0 if curr_eq > self.prev_eq else -1.0
                self.prev_eq = curr_eq
            except: curr_eq = self.prev_eq

            for sym in SYMBOL_LIST:
                try:
                    df = self.get_data(sym)
                    df = prepare_data(df)
                    price = df['close'].iloc[-1]
                    pred = predict_direction(df)
                    rsi = df['rsi'].iloc[-1]
                    vol = df['volatility'].iloc[-1]
                    atr = df['atr'].iloc[-1]
                    
                    state = get_market_state(vol)
                    lev = choose_leverage(sym, state)
                    update_q(sym, state, lev, reward)
                    
                    # LOGIC: Prediction + RSI Filter
                    signal = None
                    # Stronger signal requirement: 0.2% predicted move
                    if pred > price * 1.002 and rsi < 70: signal = "Buy"
                    elif pred < price * 0.998 and rsi > 30: signal = "Sell"
                    
                    if signal:
                        risk_amt = curr_eq * RISK_PER_TRADE
                        sl_dist = atr * 2.0
                        raw_qty = risk_amt / sl_dist
                        
                        # ENFORCE MINIMUMS
                        info = bot_state['instrument_info'].get(sym)
                        if info:
                            if raw_qty < info['min_qty']: raw_qty = info['min_qty']
                            # Round to step
                            step = info['qty_step']
                            qty = math.floor(raw_qty / step) * step
                            qty_str = f"{qty:.{str(step)[::-1].find('.')}f}"
                        else:
                            qty_str = f"{raw_qty:.3f}"
                        
                        try: self.session.set_leverage(category="linear", symbol=sym, buy_leverage=str(lev), sell_leverage=str(lev))
                        except: pass
                        
                        sl_p = price - sl_dist if signal == "Buy" else price + sl_dist
                        tp_p = price + (sl_dist * 3.0) if signal == "Buy" else price - (sl_dist * 3.0)
                        
                        self.session.place_order(
                            category="linear", symbol=sym, side=signal, orderType="Market", qty=qty_str,
                            stopLoss=f"{sl_p:.2f}", takeProfit=f"{tp_p:.2f}"
                        )
                        logger.info(f"{sym} {signal} | Qty: {qty_str} | Lev: {lev}x")
                    else:
                        logger.info(f"{sym}: No Signal. Pred: {pred:.2f} / Cur: {price:.2f}")
                        
                except Exception as e:
                    logger.error(f"Error {sym}: {e}")
            
            time.sleep(15)

if __name__ == "__main__":
    threading.Thread(target=start_server, daemon=True).start()
    TradingBot().run()