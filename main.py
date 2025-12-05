import ccxt
import pandas as pd
import numpy as np
import time
import os
import logging
from datetime import datetime

# --- Configuration ---
API_KEY = os.getenv('BYBIT_API_KEY')
API_SECRET = os.getenv('BYBIT_API_SECRET')

# SURGICAL FIX: Check Render Environment for TESTNET toggle
USE_TESTNET = os.getenv('TESTNET', 'False').lower() == 'true'

# Use 'linear' for USDT perpetuals on Bybit
CATEGORY = 'linear' 
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'] 
TIMEFRAME = '15m'
LIMIT = 100

# Q-Learning Parameters
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON = 0.1 # Exploration rate

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()] # Print to console/Render logs
)
logger = logging.getLogger()

# --- Exchange Connection ---
def connect_exchange():
    try:
        exchange = ccxt.bybit({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future', 
                'adjustForTimeDifference': True,
            }
        })
        
        # SURGICAL FIX: Switch to Testnet only if configured
        if USE_TESTNET:
            exchange.set_sandbox_mode(True)
            logger.info("⚠️ RUNNING IN TESTNET MODE ⚠️")
        else:
            logger.info("RUNNING IN LIVE MODE")

        # Check connection
        exchange.load_markets()
        logger.info("Connected to Bybit successfully.")
        return exchange
    except Exception as e:
        logger.error(f"Failed to connect to Bybit: {e}")
        return None

# --- Data Fetching ---
def fetch_data(exchange, symbol, timeframe='15m', limit=100):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

# --- Technical Indicators (The 'State') ---
def get_state(df):
    if df.empty or len(df) < 26:
        return 'unknown'
    
    # Simple Moving Average Crossover State
    df['SMA_12'] = df['close'].rolling(window=12).mean()
    df['SMA_26'] = df['close'].rolling(window=26).mean()
    
    latest = df.iloc[-1]
    
    if latest['SMA_12'] > latest['SMA_26']:
        return 'uptrend'
    elif latest['SMA_12'] < latest['SMA_26']:
        return 'downtrend'
    else:
        return 'neutral'

# --- Q-Learning Logic ---

# Initialize Q-table
# State: uptrend, downtrend, neutral
# Actions: 0 (Hold), 1 (Buy/Long), 2 (Sell/Short)
q_table = {
    'uptrend':   [0.0, 0.0, 0.0],
    'downtrend': [0.0, 0.0, 0.0],
    'neutral':   [0.0, 0.0, 0.0],
    'unknown':   [0.0, 0.0, 0.0]
}

def choose_action(state):
    # Epsilon-greedy strategy
    if np.random.uniform(0, 1) < EPSILON:
        return np.random.choice([0, 1, 2]) # Explore
    else:
        return np.argmax(q_table.get(state, [0,0,0])) # Exploit

def update_q_table(state, action, reward, next_state):
    """
    Updates the Q-table using the Bellman equation.
    """
    try:
        current_q = q_table[state][action]
        max_future_q = np.max(q_table[next_state])
        
        # Bellman Equation
        new_q = (1 - ALPHA) * current_q + ALPHA * (reward + GAMMA * max_future_q)
        
        q_table[state][action] = new_q
        logger.info(f"Updated Q-table for {state}, action {action}: {new_q:.4f}")
    except Exception as e:
        logger.error(f"Error in update_q_table: {e}")

def calculate_reward(entry_price, current_price, position_type):
    if position_type == 'long':
        return (current_price - entry_price) / entry_price
    elif position_type == 'short':
        return (entry_price - current_price) / entry_price
    return 0

# --- Execution Logic ---
def execute_trade(exchange, symbol, action):
    # This is a simplified execution for demonstration
    # 0 = Hold, 1 = Buy, 2 = Sell
    
    # In a real bot, you would check current positions first using private API
    # For now, we just log the intent
    if action == 1:
        logger.info(f"Signal: BUY {symbol}")
        # exchange.create_market_buy_order(symbol, amount) <-- Real execution
    elif action == 2:
        logger.info(f"Signal: SELL {symbol}")
        # exchange.create_market_sell_order(symbol, amount) <-- Real execution
    else:
        logger.info(f"Signal: HOLD {symbol}")

# --- Main Loop ---
def main():
    exchange = connect_exchange()
    if not exchange:
        return

    logger.info("Starting bot loop...")
    
    # Keep track of previous states to calculate rewards
    previous_states = {symbol: 'unknown' for symbol in SYMBOLS}
    previous_prices = {symbol: 0.0 for symbol in SYMBOLS}
    last_actions = {symbol: 0 for symbol in SYMBOLS}

    while True:
        try:
            for symbol in SYMBOLS:
                df = fetch_data(exchange, symbol, TIMEFRAME)
                if df.empty:
                    continue
                
                current_state = get_state(df)
                current_price = df.iloc[-1]['close']
                
                # 1. Calculate Reward from PREVIOUS action (if any)
                # (Simplified: assumes we entered at previous price)
                prev_state = previous_states[symbol]
                prev_action = last_actions[symbol]
                
                if prev_state != 'unknown':
                    reward = 0
                    if prev_action == 1: # Long
                        reward = calculate_reward(previous_prices[symbol], current_price, 'long')
                    elif prev_action == 2: # Short
                        reward = calculate_reward(previous_prices[symbol], current_price, 'short')
                    
                    # 2. Update Q-Table based on what happened
                    update_q_table(prev_state, prev_action, reward, current_state)

                # 3. Decide NEW action
                action = choose_action(current_state)
                
                # 4. Execute
                execute_trade(exchange, symbol, action)
                
                # 5. Store state for next iteration
                previous_states[symbol] = current_state
                previous_prices[symbol] = current_price
                last_actions[symbol] = action
                
            logger.info("Sleeping for 1 minute...")
            time.sleep(60) # Run every minute

        except KeyboardInterrupt:
            logger.info("Bot stopped by user.")
            break
        except Exception as e:
            logger.error(f"Global Loop Error: {e}")
            time.sleep(30) # Wait before retrying

if __name__ == "__main__":
    main()