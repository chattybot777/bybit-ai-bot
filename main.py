import ccxt
import pandas as pd
import numpy as np
import time
import os
import logging
import json
from datetime import datetime

# --- Configuration ---
API_KEY = os.getenv('BYBIT_API_KEY')
API_SECRET = os.getenv('BYBIT_API_SECRET')
USE_TESTNET = os.getenv('TESTNET', 'False').lower() == 'true'

CATEGORY = 'linear' 
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'] 
TIMEFRAME = '15m'
LIMIT = 200 # Increased to ensure enough data for RSI

# Q-Learning Parameters
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON = 0.1 # Exploration rate

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()] 
)
logger = logging.getLogger()

# --- Exchange Connection ---
def connect_exchange():
    try:
        exchange = ccxt.bybit({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,
            'options': { 'defaultType': 'future', 'adjustForTimeDifference': True }
        })
        
        if USE_TESTNET:
            exchange.set_sandbox_mode(True)
            logger.info("⚠️ RUNNING IN TESTNET MODE ⚠️")
        else:
            logger.info("RUNNING IN LIVE MODE")

        exchange.load_markets()
        return exchange
    except Exception as e:
        logger.error(f"Failed to connect to Bybit: {e}")
        return None

# --- Data Fetching ---
def fetch_data(exchange, symbol):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=LIMIT)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# --- INTELLIGENCE UPGRADE: RSI Calculation ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    
    # Use Exponential Moving Average for RSI (Standard)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- INTELLIGENCE UPGRADE: Advanced State ---
def get_state(df):
    if df.empty or len(df) < 50:
        return 'unknown'
    
    # 1. Trend (SMA)
    df['SMA_12'] = df['close'].rolling(window=12).mean()
    df['SMA_26'] = df['close'].rolling(window=26).mean()
    
    # 2. Momentum (RSI)
    df['RSI'] = calculate_rsi(df['close'])
    
    latest = df.iloc[-1]
    
    # Determine Trend
    if latest['SMA_12'] > latest['SMA_26']:
        trend = 'uptrend'
    elif latest['SMA_12'] < latest['SMA_26']:
        trend = 'downtrend'
    else:
        trend = 'neutral'
        
    # Determine Momentum
    if latest['RSI'] > 70:
        momentum = 'overbought'
    elif latest['RSI'] < 30:
        momentum = 'oversold'
    else:
        momentum = 'neutral'
        
    # Combine them into a complex state (e.g., "uptrend_overbought")
    return f"{trend}_{momentum}"

# --- Q-Learning Logic (Expanded Brain) ---

# We now generate states dynamically or define the common ones
# 3 Trends * 3 Momentums = 9 States
q_table = {} 

def get_q_values(state):
    # If state is new, initialize it with [0.0, 0.0, 0.0]
    if state not in q_table:
        q_table[state] = [0.0, 0.0, 0.0]
    return q_table[state]

def choose_action(state):
    if np.random.uniform(0, 1) < EPSILON:
        return np.random.choice([0, 1, 2]) # Explore
    else:
        return np.argmax(get_q_values(state)) # Exploit

def update_q_table(state, action, reward, next_state):
    try:
        current_q = get_q_values(state)[action]
        max_future_q = np.max(get_q_values(next_state))
        
        new_q = (1 - ALPHA) * current_q + ALPHA * (reward + GAMMA * max_future_q)
        q_table[state][action] = new_q
        
        # Only log significant updates to keep logs clean
        if abs(new_q) > 0.001:
            logger.info(f"Learned: {state} -> Action {action} = {new_q:.4f}")
            
    except Exception as e:
        logger.error(f"Error updating Q-table: {e}")

def calculate_reward(entry, current, pos_type):
    if pos_type == 'long': return (current - entry) / entry
    if pos_type == 'short': return (entry - current) / entry
    return 0

# --- Save State for Shell Diagnostics ---
def save_bot_state(stats):
    try:
        data = { 
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
            "stats": stats, 
            "q_table": q_table # Dumps the whole brain
        }
        with open('bot_state.json', 'w') as f: 
            json.dump(data, f, indent=4)
    except Exception as e: 
        logger.error(f"Save failed: {e}")

# --- Execution Logic ---
def execute_trade(exchange, symbol, action):
    if action == 1: logger.info(f"Signal: BUY {symbol}")
    elif action == 2: logger.info(f"Signal: SELL {symbol}")
    else: logger.info(f"Signal: HOLD {symbol}")

# --- Main Loop ---
def main():
    exchange = connect_exchange()
    if not exchange: return

    stats = {'wins': 0, 'losses': 0, 'total_actions': 0}
    prev_states = {s: 'unknown' for s in SYMBOLS}
    prev_prices = {s: 0.0 for s in SYMBOLS}
    last_actions = {s: 0 for s in SYMBOLS}

    logger.info("Bot upgraded with RSI Intelligence. Starting loop...")

    while True:
        try:
            for symbol in SYMBOLS:
                df = fetch_data(exchange, symbol)
                if df.empty: continue
                
                curr_state = get_state(df)
                curr_price = df.iloc[-1]['close']
                
                # Reward & Update
                p_state, p_action = prev_states[symbol], last_actions[symbol]
                
                if p_state != 'unknown':
                    reward = 0
                    if p_action == 1: reward = calculate_reward(prev_prices[symbol], curr_price, 'long')
                    elif p_action == 2: reward = calculate_reward(prev_prices[symbol], curr_price, 'short')
                    
                    if reward > 0: stats['wins'] += 1
                    elif reward < 0: stats['losses'] += 1
                    if p_action != 0: stats['total_actions'] += 1
                    
                    update_q_table(p_state, p_action, reward, curr_state)

                # Action & Execute
                action = choose_action(curr_state)
                execute_trade(exchange, symbol, action)
                
                prev_states[symbol] = curr_state
                prev_prices[symbol] = curr_price
                last_actions[symbol] = action
            
            # Save diagnostics
            save_bot_state(stats)
            time.sleep(60)

        except KeyboardInterrupt: break
        except Exception as e:
            logger.error(f"Loop Error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()