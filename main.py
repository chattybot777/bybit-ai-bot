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
LIMIT = 200 

# Q-Learning Parameters
ALPHA = 0.1  
GAMMA = 0.9  
EPSILON = 0.1 

# --- CRITICAL: REALITY CHECK ---
# We simulate a "Hurdle Rate".
# Bybit Fee (0.06%) * 2 (Entry/Exit) = 0.12%
# We set this to 0.0015 (0.15%) to account for slippage.
# The bot will ONLY be happy if it makes more than 0.15% profit.
ROUND_TRIP_COST = 0.0015 

# Risk Management
RISK_PER_TRADE = 0.02 # Increased slightly to 2% since we trade less often
MAX_LEVERAGE = 5

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
        logger.error(f"Failed to connect: {e}")
        return None

# --- Data Fetching ---
def fetch_data(exchange, symbol):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=LIMIT)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception:
        return pd.DataFrame()

# --- MATH INDICATORS ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean()

def calculate_obv(df):
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    return obv

# --- ADVANCED STATE ---
def get_state(df):
    if df.empty or len(df) < 50: return 'unknown'
    
    df['SMA_12'] = df['close'].rolling(12).mean()
    df['SMA_26'] = df['close'].rolling(26).mean()
    df['RSI'] = calculate_rsi(df['close'])
    df['ATR'] = calculate_atr(df)
    df['OBV'] = calculate_obv(df)
    df['OBV_SMA'] = df['OBV'].rolling(12).mean()

    latest = df.iloc[-1]
    
    trend = 'uptrend' if latest['SMA_12'] > latest['SMA_26'] else 'downtrend'
    if abs(latest['SMA_12'] - latest['SMA_26']) < (latest['close'] * 0.001): trend = 'neutral'

    momentum = 'neutral'
    if latest['RSI'] > 70: momentum = 'overbought'
    elif latest['RSI'] < 30: momentum = 'oversold'
    
    avg_atr = df['ATR'].rolling(50).mean().iloc[-1]
    volatility = 'high_vol' if latest['ATR'] > avg_atr else 'low_vol'
    
    return f"{trend}_{momentum}_{volatility}"

# --- Q-Learning Logic ---
q_table = {} 

def get_q_values(state):
    if state not in q_table: q_table[state] = [0.0, 0.0, 0.0]
    return q_table[state]

def choose_action(state):
    if np.random.uniform(0, 1) < EPSILON: return np.random.choice([0, 1, 2])
    return np.argmax(get_q_values(state))

def update_q_table(state, action, reward, next_state):
    try:
        current_q = get_q_values(state)[action]
        max_next_q = np.max(get_q_values(next_state))
        new_q = (1 - ALPHA) * current_q + ALPHA * (reward + GAMMA * max_next_q)
        q_table[state][action] = new_q
    except Exception: pass

# --- PROFIT-FOCUSED REWARD FUNCTION ---
def calculate_reward(entry, current, pos_type):
    # 1. Calculate Raw Gain
    raw_profit = 0
    if pos_type == 'long': 
        raw_profit = (current - entry) / entry
    elif pos_type == 'short': 
        raw_profit = (entry - current) / entry
    
    # 2. SUBTRACT "THE HOUSE CUT"
    # If the trade made 0.10%, but fees are 0.15%, the Result is -0.05% (NEGATIVE).
    # The bot learns that small trades are painful.
    net_reward = raw_profit - ROUND_TRIP_COST
    
    return net_reward

def save_bot_state(stats):
    try:
        data = { "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "stats": stats, "q_table": q_table }
        with open('bot_state.json', 'w') as f: json.dump(data, f, indent=4)
    except Exception: pass

# --- EXECUTION ---
def execute_trade(exchange, symbol, action, atr, close_price):
    if action == 0:
        logger.info(f"Signal: HOLD {symbol}")
        return

    # Dynamic Sizing (Risk Management)
    mock_balance = 1000 
    risk_amt = mock_balance * RISK_PER_TRADE
    if pd.isna(atr) or atr == 0: atr = close_price * 0.01 
    position_size = risk_amt / atr
    position_size = min(position_size, mock_balance / close_price * MAX_LEVERAGE)

    side = "BUY" if action == 1 else "SELL"
    # Log the fee awareness
    logger.info(f"Signal: {side} {symbol} | Size: {position_size:.4f} | Must beat 0.15% to win")

# --- Main Loop ---
def main():
    exchange = connect_exchange()
    if not exchange: return

    stats = {'wins': 0, 'losses': 0, 'total_actions': 0}
    prev_states = {s: 'unknown' for s in SYMBOLS}
    prev_prices = {s: 0.0 for s in SYMBOLS}
    last_actions = {s: 0 for s in SYMBOLS}

    logger.info("Bot Updated: Strict Fee Filtering Enabled.")

    while True:
        try:
            for symbol in SYMBOLS:
                df = fetch_data(exchange, symbol)
                if df.empty: continue
                
                curr_state = get_state(df)
                curr_price = df.iloc[-1]['close']
                curr_atr = df.iloc[-1]['ATR'] if 'ATR' in df else 0
                
                # Reward & Update
                p_state, p_action = prev_states[symbol], last_actions[symbol]
                if p_state != 'unknown':
                    reward = 0
                    if p_action == 1: reward = calculate_reward(prev_prices[symbol], curr_price, 'long')
                    elif p_action == 2: reward = calculate_reward(prev_prices[symbol], curr_price, 'short')
                    
                    # Update Stats (Only count as win if reward is POSITIVE after fees)
                    if reward > 0: stats['wins'] += 1
                    elif reward < 0: stats['losses'] += 1
                    if p_action != 0: stats['total_actions'] += 1
                    
                    update_q_table(p_state, p_action, reward, curr_state)

                # Action & Execute
                action = choose_action(curr_state)
                execute_trade(exchange, symbol, action, curr_atr, curr_price)
                
                prev_states[symbol] = curr_state
                prev_prices[symbol] = curr_price
                last_actions[symbol] = action
            
            save_bot_state(stats)
            time.sleep(60)

        except KeyboardInterrupt: break
        except Exception as e:
            logger.error(f"Loop Error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()