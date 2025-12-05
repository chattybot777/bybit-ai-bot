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

# Risk Management
RISK_PER_TRADE = 0.01 # Risk 1% of capital per trade
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

# --- MATH UPGRADES (Lean & Fast) ---

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(df, period=14):
    # Volatility Measure
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean()

def calculate_obv(df):
    # Volume Confirmation
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    return obv

# --- ADVANCED STATE ---
def get_state(df):
    if df.empty or len(df) < 50: return 'unknown'
    
    # 1. Trend (SMA)
    df['SMA_12'] = df['close'].rolling(12).mean()
    df['SMA_26'] = df['close'].rolling(26).mean()
    
    # 2. Momentum (RSI)
    df['RSI'] = calculate_rsi(df['close'])
    
    # 3. Volatility (ATR)
    df['ATR'] = calculate_atr(df)
    
    # 4. Volume (OBV Slope)
    df['OBV'] = calculate_obv(df)
    df['OBV_SMA'] = df['OBV'].rolling(12).mean() # Is volume trending up?

    latest = df.iloc[-1]
    
    # Trend
    trend = 'uptrend' if latest['SMA_12'] > latest['SMA_26'] else 'downtrend'
    if abs(latest['SMA_12'] - latest['SMA_26']) < (latest['close'] * 0.001): trend = 'neutral' # Filter weak trends

    # RSI
    momentum = 'neutral'
    if latest['RSI'] > 70: momentum = 'overbought'
    elif latest['RSI'] < 30: momentum = 'oversold'
    
    # Volatility State (Low/High)
    # Compare current ATR to average ATR of last 50 candles
    avg_atr = df['ATR'].rolling(50).mean().iloc[-1]
    volatility = 'high_vol' if latest['ATR'] > avg_atr else 'low_vol'
    
    # Volume Confirmation
    volume_conf = 'conf' if latest['OBV'] > latest['OBV_SMA'] else 'div' # Confirmed or Divergent

    # Complex State: "uptrend_overbought_highvol_conf"
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

def calculate_reward(entry, current, pos_type):
    if pos_type == 'long': return (current - entry) / entry
    if pos_type == 'short': return (entry - current) / entry
    return 0

def save_bot_state(stats):
    try:
        data = { "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "stats": stats, "q_table": q_table }
        with open('bot_state.json', 'w') as f: json.dump(data, f, indent=4)
    except Exception: pass

# --- EXECUTION (With Dynamic Sizing) ---
def execute_trade(exchange, symbol, action, atr, close_price):
    if action == 0:
        logger.info(f"Signal: HOLD {symbol}")
        return

    # Dynamic Position Sizing based on Volatility (ATR)
    # If Volatility is HIGH, Position Size is LOW
    # Formula: (Account Balance * Risk) / ATR
    # *Simplified for demo: We just calculate the raw quantity, assuming $1000 balance
    
    mock_balance = 1000 
    risk_amt = mock_balance * RISK_PER_TRADE
    
    # Safety check for zero/nan ATR
    if pd.isna(atr) or atr == 0: atr = close_price * 0.01 
    
    position_size = risk_amt / atr
    position_size = min(position_size, mock_balance / close_price * MAX_LEVERAGE) # Cap leverage

    side = "BUY" if action == 1 else "SELL"
    logger.info(f"Signal: {side} {symbol} | Size: {position_size:.4f} (Adj. for Volatility)")

# --- Main Loop ---
def main():
    exchange = connect_exchange()
    if not exchange: return

    stats = {'wins': 0, 'losses': 0, 'total_actions': 0}
    prev_states = {s: 'unknown' for s in SYMBOLS}
    prev_prices = {s: 0.0 for s in SYMBOLS}
    last_actions = {s: 0 for s in SYMBOLS}

    logger.info("Bot upgraded to Ultimate Edition (ATR + OBV + Risk Mgr).")

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