import ccxt
import pandas as pd
import numpy as np
import time
import os
import logging
import json
import csv
import threading
from datetime import datetime
from collections import deque
from flask import Flask, request, redirect, url_for

# --- Configuration ---
API_KEY = os.getenv('BYBIT_API_KEY')
API_SECRET = os.getenv('BYBIT_API_SECRET')
USE_TESTNET = os.getenv('TESTNET', 'False').lower() == 'true'
CATEGORY = 'linear' 
TIMEFRAME = '15m'
LIMIT = 200 

# --- CONCENTRATED ASSET LIST (Top 5) ---
SYMBOLS = [
    'BTCUSDT', 
    'ETHUSDT', 
    'SOLUSDT', 
    'XRPUSDT', 
    'DOGEUSDT'
]

# Q-Learning Constants
ALPHA = 0.1  
GAMMA = 0.9  
EPSILON = 0.05 
ROUND_TRIP_COST = 0.0015 
MAX_LEVERAGE = 5

# --- NEW: Maximum acceptable simulated loss per trade (2.5% safety net) ---
MAX_SIMULATED_LOSS = -0.025 

# --- Shared Memory ---
bot_memory = {
    'wins': 0, 
    'losses': 0, 
    'total_actions': 0, 
    'cumulative_pnl_percent': 0.0,
    'last_trade': "None yet",
    'status': "RUNNING", 
    'risk_per_trade': 0.02, 
    'uptime': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

log_buffer = deque(maxlen=50)

# --- Logging ---
class WebLogger(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_buffer.appendleft(f"[{timestamp}] {log_entry}")

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(WebLogger()) 

# --- WEB DASHBOARD ---
app = Flask(__name__)

@app.route('/')
def dashboard():
    state_color = "green" if bot_memory['status'] == "RUNNING" else "red"
    logs_html = "".join([f"<div style='border-bottom:1px solid #eee; padding:5px; font-family:monospace; font-size:12px;'>{log}</div>" for log in log_buffer])
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Gavin's AI Trader (Top 5)</title>
        <meta http-equiv="refresh" content="10">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{ font-family: -apple-system, system-ui, sans-serif; background: #f0f2f5; padding: 20px; margin: 0; }}
            .container {{ max-width: 800px; margin: auto; }}
            .card {{ background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 20px; }}
            .stat-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }}
            .stat-box {{ background: #f8f9fa; padding: 10px; border-radius: 8px; text-align: center; }}
            .value {{ font-size: 20px; font-weight: bold; color: #333; }}
            .label {{ color: #666; font-size: 12px; }}
            .log-box {{ height: 300px; overflow-y: auto; background: #fff; border: 1px solid #ddd; padding: 10px; border-radius: 8px; }}
            button {{ padding: 10px 20px; border-radius: 6px; border: none; cursor: pointer; font-weight: bold; margin-right: 5px; }}
            .btn-stop {{ background: #ff4757; color: white; }}
            .btn-start {{ background: #2ed573; color: white; }}
            .btn-update {{ background: #3742fa; color: white; }}
            input {{ padding: 8px; border-radius: 4px; border: 1px solid #ccc; width: 80px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h2>🤖 Top 5 Sniper</h2>
                    <span style="background:{state_color}; color:white; padding:5px 15px; border-radius:20px; font-weight:bold; font-size:12px;">
                        {bot_memory['status']}
                    </span>
                </div>
                
                <div class="stat-grid">
                    <div class="stat-box"><div class="value" style="color:green">{bot_memory['wins']}</div><div class="label">Wins</div></div>
                    <div class="stat-box"><div class="value" style="color:red">{bot_memory['losses']}</div><div class="label">Losses</div></div>
                    <div class="stat-box"><div class="value">{bot_memory['total_actions']}</div><div class="label">Total Trades</div></div>
                    <div class="stat-box"><div class="value">{bot_memory['cumulative_pnl_percent']*100:.2f}%</div><div class="label">Net PnL</div></div>
                </div>
                <p style="font-size:14px;"><strong>Last Action:</strong> {bot_memory['last_trade']}</p>
            </div>

            <div class="card">
                <h3>⚙️ Controls</h3>
                <form action="/control" method="post" style="margin-bottom:15px;">
                    <button name="action" value="start" class="btn-start">▶ Resume</button>
                    <button name="action" value="stop" class="btn-stop">⏸ Pause</button>
                </form>
                
                <form action="/settings" method="post">
                    <label style="font-size:14px;">Set Risk %: </label>
                    <input type="number" step="0.01" name="risk" value="{bot_memory['risk_per_trade']}">
                    <button type="submit" class="btn-update">Update</button>
                </form>
            </div>

            <div class="card">
                <h3>🧠 Bot Thought Stream</h3>
                <div class="log-box">
                    {logs_html}
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return html

@app.route('/control', methods=['POST'])
def control():
    action = request.form.get('action')
    if action == 'stop':
        bot_memory['status'] = "PAUSED"
        logger.warning("⏸ BOT PAUSED")
    elif action == 'start':
        bot_memory['status'] = "RUNNING"
        logger.info("▶ BOT RESUMED")
    return redirect(url_for('dashboard'))

@app.route('/settings', methods=['POST'])
def settings():
    try:
        new_risk = float(request.form.get('risk'))
        bot_memory['risk_per_trade'] = new_risk
        logger.info(f"⚙️ Risk set to {new_risk*100}%")
    except: pass
    return redirect(url_for('dashboard'))

def run_web_server():
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)

# --- TRADING LOGIC ---
def connect_exchange():
    try:
        exchange = ccxt.bybit({'apiKey': API_KEY, 'secret': API_SECRET})
        if USE_TESTNET: exchange.set_sandbox_mode(True)
        return exchange
    except: return None

def fetch_data(exchange, symbol):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=LIMIT)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except: return pd.DataFrame()

def calculate_indicators(df):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR'] = ranges.max(axis=1).rolling(14).mean()
    
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    return df

def get_state(df):
    if df.empty or len(df) < 50: return 'unknown'
    df = calculate_indicators(df)
    df['SMA_12'] = df['close'].rolling(12).mean()
    df['SMA_26'] = df['close'].rolling(26).mean()
    latest = df.iloc[-1]
    
    trend = 'uptrend' if latest['SMA_12'] > latest['SMA_26'] else 'downtrend'
    if abs(latest['SMA_12'] - latest['SMA_26']) < (latest['close'] * 0.001): trend = 'neutral'
    
    momentum = 'neutral'
    if latest['RSI'] > 70: momentum = 'overbought'
    elif latest['RSI'] < 30: momentum = 'oversold'
    
    avg_atr = df['ATR'].rolling(50).mean().iloc[-1]
    volatility = 'high_vol' if latest['ATR'] > avg_atr else 'low_vol'
    
    return f"{trend}_{momentum}_{volatility}"

q_table = {} 
def get_q_values(state):
    if state not in q_table: q_table[state] = [0.0, 0.0, 0.0]
    return q_table[state]

def choose_action(state):
    if np.random.uniform(0, 1) < EPSILON: return np.random.choice([0, 1, 2])
    return np.argmax(get_q_values(state))

def update_q_table(state, action, reward, next_state):
    try:
        curr = get_q_values(state)[action]
        max_next = np.max(get_q_values(next_state))
        q_table[state][action] = (1 - ALPHA) * curr + ALPHA * (reward + GAMMA * max_next)
    except: pass

def calculate_reward(entry, current, pos_type, symbol):
    raw = (current - entry) / entry if pos_type == 'long' else (entry - current) / entry
    
    # --- CRITICAL CHANGE: Enforce Max Simulated Loss ---
    # This prevents a single volatile trade from corrupting the entire Q-Table.
    if raw < MAX_SIMULATED_LOSS:
        net = MAX_SIMULATED_LOSS # Cap the loss at -2.5%
        res = "STOPPED_OUT"
    else:
        net = raw - ROUND_TRIP_COST
        res = "WIN" if net > 0 else "LOSS"
    
    bot_memory['last_trade'] = f"{symbol} {pos_type.upper()} {res} ({net*100:.2f}%)"
    
    try:
        with open('trades.csv', 'a', newline='') as f:
            csv.writer(f).writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), symbol, pos_type, res, f"{net*100:.4f}%"])
    except: pass
    
    return net

def execute_trade(exchange, symbol, action, atr, close_price, state):
    if action == 0: return
    
    balance = 1000 
    risk = balance * bot_memory['risk_per_trade']
    if pd.isna(atr) or atr == 0: atr = close_price * 0.01 
    size = min(risk / atr, balance / close_price * MAX_LEVERAGE)
    
    side = "BUY" if action == 1 else "SELL"
    logger.info(f"⚡ {side} {symbol} | State: {state} | Size: {size:.4f}")

def save_bot_state():
    try:
        data = { "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "stats": bot_memory, "q_table": q_table }
        with open('bot_state.json', 'w') as f: json.dump(data, f, indent=4)
    except: pass

# --- MAIN LOOP ---
def main():
    t = threading.Thread(target=run_web_server)
    t.daemon = True 
    t.start()

    exchange = connect_exchange()
    if not exchange: return

    prev_states = {s: 'unknown' for s in SYMBOLS}
    prev_prices = {s: 0.0 for s in SYMBOLS}
    last_actions = {s: 0 for s in SYMBOLS}

    logger.info(f"🚀 Top 5 Sniper Mode Active. Monitoring: {SYMBOLS}")

    while True:
        try:
            if bot_memory['status'] == "PAUSED":
                time.sleep(5) 
                continue

            trades_made = 0
            for symbol in SYMBOLS:
                df = fetch_data(exchange, symbol)
                if df.empty: continue
                
                curr_state = get_state(df)
                curr_price = df.iloc[-1]['close']
                curr_atr = df.iloc[-1]['ATR'] if 'ATR' in df else 0
                
                p_state, p_action = prev_states[symbol], last_actions[symbol]
                
                if p_state != 'unknown' and p_action != 0:
                    pos_type = 'long' if p_action == 1 else 'short'
                    reward = calculate_reward(prev_prices[symbol], curr_price, pos_type, symbol)
                    
                    bot_memory['total_actions'] += 1
                    bot_memory['cumulative_pnl_percent'] += reward
                    if reward > 0: bot_memory['wins'] += 1
                    else: bot_memory['losses'] += 1
                    
                    update_q_table(p_state, p_action, reward, curr_state)

                action = choose_action(curr_state)
                execute_trade(exchange, symbol, action, curr_atr, curr_price, curr_state)
                if action != 0: trades_made += 1
                
                prev_states[symbol] = curr_state
                prev_prices[symbol] = curr_price
                last_actions[symbol] = action
                
                time.sleep(0.5) 
            
            if trades_made == 0:
                logger.info(f"💤 Scan Complete. Market calm. {len(SYMBOLS)} pairs checked.")
            
            save_bot_state()
            time.sleep(50) 

        except KeyboardInterrupt: break
        except Exception as e:
            logger.error(f"System Error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()