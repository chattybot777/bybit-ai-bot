# === Bybit Bot: Clean Multi-Symbol (15m), Adaptive, Risk-Sized, Stable Globals ===
import os, sys, time, math, threading, logging, pickle, json, inspect
from typing import Tuple, Optional, Dict, List
import requests
import numpy as np # <-- np is used globally
import pandas as pd
import torch
import torch.nn as nn
from pybit.unified_trading import HTTP
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from tenacity import retry, stop_after_attempt, wait_exponential
from http.server import BaseHTTPRequestHandler, HTTPServer
import string # REQUIRED for aggressive environment variable cleanup


class SafeJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to reliably handle NumPy types (int64, float64) 
    by converting them to standard Python int/float for serialization.
    """
    def default(self, o):
        # Use the globally imported 'np' for checks
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return super().default(o)
        
# -------- Logging --------
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# -------- Env & Session --------
load_dotenv()
torch.set_num_threads(1)

def sanitize_env_var(value: str) -> str:
    """
    Aggressively strips whitespace, null bytes, and non-printable characters 
    to prevent header errors from environments like Render (the key fix).
    """
    if not value:
        return ""
    # 1. Strip all leading/trailing whitespace, including \n, \r, \t, etc.
    cleaned = value.strip()
    # 2. Remove all internal/trailing carriage returns and newlines that .strip() might miss
    cleaned = cleaned.replace('\n', '').replace('\r', '').replace('\x00', '')
    # 3. Filter out any non-printable ASCII characters (just for maximum safety)
    cleaned = ''.join(c for c in cleaned if c in string.printable)
    return cleaned.strip() # Final aggressive strip for good measure

# API Keys and Setup (REVISED, SANITIZED AUTHENTICATION BLOCK)
API_KEY = sanitize_env_var(os.getenv("BYBIT_API_KEY", ""))
API_SECRET = sanitize_env_var(os.getenv("BYBIT_API_SECRET") or os.getenv("API_SECRET") or "")

# --- Diagnostic Logging (Safety Check) ---
# Log the length of the sanitized key/secret to confirm cleanup
logging.info(f"Auth Check: API_KEY length={len(API_KEY)} (Sanitized)")
logging.info(f"Auth Check: API_SECRET length={len(API_SECRET)} (Sanitized)")
if not API_KEY or not API_SECRET:
    logging.critical("API credentials are empty after sanitization. Check Render environment variables!")

TESTNET = (os.getenv("BYBIT_TESTNET") or os.getenv("TESTNET") or "true").lower() in ("1","true","yes")
RECV_WINDOW = int(os.getenv("RECV_WINDOW", "60000"))

# Bot Controls & Risk Management
BOT_PAUSED = os.getenv("BOT_PAUSED", "false").lower() == "true"
LEVERAGE_MAX = int(os.getenv("LEVERAGE_MAX", "20"))
LEVERAGE_MIN = int(os.getenv("LEVERAGE_MIN", "10"))
MIN_NOTIONAL_USDT = float(os.getenv("MIN_NOTIONAL_USDT", "10"))
ENTRY_ALIGN_MODE = os.getenv("ENTRY_ALIGN_MODE","AND").upper()
BASE_POSITION_PCT = float(os.getenv("BASE_POSITION_PCT", "0.05")) # Used as a soft cap on notional exposure relative to max leverage
DRAWDOWN_STOP = float(os.getenv("DRAWDOWN_STOP", "0.20"))
RESET_PNL_WEEKLY = os.getenv("RESET_PNL_WEEKLY", "false").lower() == "true"
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.005")) # CRITICAL: Max equity loss (0.5%) per trade

# Strategy knobs
COOLDOWN_SEC = 15 * 60          # per-symbol cooldown
MAX_TRADES_PER_DAY = 10         # per symbol
TIME_STOP_HOURS = 4
BASE_TP_PCT = 0.015
BASE_SL_PCT = 0.010
MIN_GATE_STD = float(os.getenv("MIN_GATE_STD", "0.003"))   # lower gate => more trades
MIN_RISK = float(os.getenv("MIN_RISK", "4.0"))             # risk gate

# Data hardening defaults
VOL_FALLBACK = float(os.getenv("VOL_FALLBACK", "0.01"))     # ~1% 15m vol as a sane default
VOL_CAP      = float(os.getenv("VOL_CAP", "0.20"))          # cap 15m vol at 20%

# Communications
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Initialize Bybit Session
session = HTTP(testnet=TESTNET, api_key=API_KEY, api_secret=API_SECRET, recv_window=RECV_WINDOW)

# -------- Symbols --------
def get_top_symbols() -> List[str]:
    """Fetches top 10 USDT symbols by 24h turnover from Bybit."""
    try:
        r = session.get_tickers(category="linear")["result"]["list"]
        usdt = [t for t in r if t.get("symbol","").endswith("USDT")]
        cleaned = []
        for t in usdt:
            try:
                lp = float(t.get("lastPrice") or 0)
            except Exception:
                lp = 0.0
            if lp > 0:
                cleaned.append(t)
        cleaned.sort(key=lambda t: float(t.get("turnover24h", 0) or 0), reverse=True)
        return [t["symbol"] for t in cleaned[:10]] or [t["symbol"] for t in usdt[:10]]
    except Exception as e:
        logging.warning(f"Symbol fetch error: {e}")
        return ["BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","BNBUSDT","TRXUSDT","ADAUSDT","AVAXUSDT","LINKUSDT"]

_ENV_SYMBOLS_RAW = os.getenv("SYMBOLS", "").strip()
ENV_SYMBOLS_FIXED = bool(_ENV_SYMBOLS_RAW)
SYMBOLS: List[str] = [s.strip() for s in _ENV_SYMBOLS_RAW.split(",") if s.strip()] if ENV_SYMBOLS_FIXED else get_top_symbols()

# -------- Models --------
class LSTMModel(nn.Module):
    """Simple LSTM for time-series feature processing."""
    def __init__(self, input_size=8, hidden_size=50, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# Global model artifacts
model = LSTMModel()
scaler: Optional[MinMaxScaler] = None
gb_model: Optional[GradientBoostingRegressor] = None

def load_artifacts():
    """Loads pre-trained models and scaler from disk."""
    global model, scaler, gb_model
    # Safe torch.load with weights_only if supported
    if "weights_only" in inspect.signature(torch.load).parameters:
        state = torch.load("lstm.pth", map_location="cpu", weights_only=True)
    else:
        state = torch.load("lstm.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("gb.pkl", "rb") as f:
        gb_model = pickle.load(f)
    logging.info("Artifacts loaded: Shared models for multi-symbol")

# -------- Q-table (RL leverage) persistence --------
q_tables: Dict[str, Dict[str, List[float]]] = {}
def load_q_tables():
    """Loads Q-tables for Reinforcement Learning leverage selection."""
    global q_tables
    try:
        with open("q_tables.pkl", "rb") as f:
            q_tables = pickle.load(f)
        logging.info("Loaded Q-tables")
    except:
        q_tables = {}
        logging.info("Initialized empty Q-tables")
def save_q_tables():
    """Saves Q-tables to disk."""
    try:
        with open("q_tables.pkl", "wb") as f:
            pickle.dump(q_tables, f)
    except Exception as e:
        logging.warning(f"Q-table save error: {e}")

# -------- Indicators & Data --------
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates the Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_data(symbol: str) -> pd.DataFrame:
    """Fetches kline data and computes required indicators (RSI, MA50, Volatility)."""
    r = session.get_kline(category="linear", symbol=symbol, interval="15", limit=200)
    rows = r["result"]["list"]
    cols = ["start", "open", "high", "low", "close", "volume", "turnover"]
    df = pd.DataFrame(rows, columns=cols)
    for c in ["open","high","low","low","close","volume","turnover"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Strict price sanitization
    for c in ["open","high","low","close"]:
        df = df[df[c] > 0]

    df["time"] = pd.to_datetime(pd.to_numeric(df["start"], errors="coerce"), unit="ms")
    df = df.dropna(subset=["close","time"]).sort_values("time").reset_index(drop=True)

    df["ma50"] = df["close"].rolling(50, min_periods=50).mean()
    df["rsi"] = compute_rsi(df["close"])

    # Robust 15m volatility (standard deviation of price change)
    vol = df["close"].pct_change().rolling(20, min_periods=20).std()
    vol = vol.replace([np.inf, -np.inf], np.nan).fillna(VOL_FALLBACK).clip(lower=0, upper=VOL_CAP)
    df["volatility"] = vol

    df = df.dropna().reset_index(drop=True)
    return df

def _sanitize_feature_row(row: np.ndarray) -> Optional[np.ndarray]:
    """Ensures a feature row is valid before feeding to models."""
    r = row.copy()
    # basic checks for prices
    for i in range(4):
        if not np.isfinite(r[i]) or r[i] <= 0:
            return None
    # volume non-negative finite
    if not np.isfinite(r[4]) or r[4] < 0: r[4] = 0.0
    # ma50 positive/fallback to close
    if not np.isfinite(r[5]) or r[5] <= 0: r[5] = r[3]
    # rsi in [0,100], fallback 50
    if not np.isfinite(r[6]) or r[6] < 0 or r[6] > 100: r[6] = 50.0
    # vol finite, cap
    if not np.isfinite(r[7]) or r[7] < 0: r[7] = VOL_FALLBACK
    if r[7] > VOL_CAP: r[7] = VOL_CAP
    return r

def predict_price(df: pd.DataFrame) -> float:
    """Generates a consensus price prediction from LSTM and GB models."""
    assert scaler is not None and gb_model is not None
    feats = df[["open","high","low","close","volume","ma50","rsi","volatility"]].values.astype(np.float32)

    last = _sanitize_feature_row(feats[-1])
    if last is None:
        return float(df["close"].iloc[-1])

    feats[-1] = last
    # Lightweight repair: use last good row to fill bad rows
    bad_rows = ~np.isfinite(feats).all(axis=1)
    if bad_rows.any():
        good_idx = np.where(~bad_rows)[0]
        if len(good_idx) == 0:
            return float(df["close"].iloc[-1])
        last_good = feats[good_idx[-1]]
        feats[bad_rows] = last_good

    try:
        X_scaled = scaler.transform(feats)
    except Exception as e:
        logging.warning(f"Scaler.transform failed ({e}); using cur price as pred")
        return float(df["close"].iloc[-1])

    # LSTM Prediction (sequence-based)
    x_seq = torch.tensor(X_scaled.reshape(1, -1, 8), dtype=torch.float32)
    with torch.no_grad():
        lstm_pred = float(model(x_seq).item())
    
    # Gradient Boosting Prediction (point-in-time)
    try:
        gb_pred = float(gb_model.predict(feats[-1:].astype(np.float64))[0])
    except Exception as e:
        logging.warning(f"GB.predict failed ({e}); using LSTM only")
        gb_pred = lstm_pred

    pred = (lstm_pred + gb_pred) / 2.0
    cur = float(df["close"].iloc[-1])
    # Clamp prediction to prevent extreme, non-sensical values
    return max(cur * 0.9, min(cur * 1.1, pred))

# -------- Risk, Leverage, Adaptation (Core Performance Logic) --------
def get_state(vol: float) -> str:
    """Categorizes market volatility for RL Q-table state."""
    return "low" if vol < 0.01 else ("med" if vol < 0.03 else "high")

def adaptive_thresholds(vol: float, base: float) -> Tuple[float, float, float]:
    """Dynamically adjusts TP/SL and entry gates based on current volatility."""
    vol_factor = 1 + vol * 10
    tp_pct = base * vol_factor
    sl_pct = base * vol_factor * 0.67 # SL is tighter than TP (e.g., 1.5R ratio)
    gate_std = max(MIN_GATE_STD, base / vol_factor)
    return tp_pct, sl_pct, gate_std

def choose_leverage(symbol: str, state: str, risk_score: float, trade_count: int) -> int:
    """Selects leverage using Q-learning (exploitation/exploration) and risk score cap."""
    if symbol not in q_tables:
        q_tables[symbol] = {"low":[0.0]*LEVERAGE_MAX, "med":[0.0]*LEVERAGE_MAX, "high":[0.0]*LEVERAGE_MAX}
    q = q_tables[symbol][state]
    epsilon = max(0.05, 0.15 * (1 - trade_count / 100)) # Decreasing exploration over time
    
    # 1. Q-learning choice (Explore or Exploit)
    lev_q_choice = np.random.randint(1, LEVERAGE_MAX + 1) if (np.random.rand() < epsilon) else (np.argmax(q) + 1)
    
    # 2. Risk-based cap (lower leverage for higher perceived risk)
    # The '50' is an arbitrary scaling factor. Max risk score 9.99 -> min lev ~4x
    base_lev_cap = int(max(1, min(LEVERAGE_MAX, 50.0 / (risk_score + 1.0))))
    
    # Final leverage selection (min of Q-choice and Risk-cap, clamped by min/max)
    final_lev = min(int(lev_q_choice), base_lev_cap)
    
    return max(LEVERAGE_MIN, min(final_lev, LEVERAGE_MAX))

def assess_risk(df: pd.DataFrame, symbol: str, trade_count: int) -> Tuple[int, float, str, float, float, float]:
    """Bundles risk assessment and threshold calculation."""
    vol = float(df["volatility"].iloc[-1])

    # Sanitize volatility
    if not np.isfinite(vol) or vol <= 0:
        vol = VOL_FALLBACK
    if vol > VOL_CAP:
        vol = VOL_CAP

    state = get_state(vol)
    risk_score = max(0.0, min(vol * 100.0, 9.99))

    lev = choose_leverage(symbol, state, risk_score, trade_count)
    tp_pct, sl_pct, gate_std = adaptive_thresholds(vol, BASE_TP_PCT)
    return lev, risk_score, state, tp_pct, sl_pct, gate_std

def update_q(symbol: str, reward: float, state: str, action_lev: int):
    """Updates the Q-table based on the trade outcome (reward)."""
    if symbol not in q_tables:
        q_tables[symbol] = {"low":[0.0]*LEVERAGE_MAX, "med":[0.0]*LEVERAGE_MAX, "high":[0.0]*LEVERAGE_MAX}
    q = q_tables[symbol][state]
    idx = action_lev - 1
    if 0 <= idx < len(q):
        # Basic Q-learning update: Q = Q + alpha * (Reward - Q)
        alpha = 0.1
        q[idx] += alpha * (reward - q[idx])
        save_q_tables()

# -------- Utils --------
def send_telegram(msg: str):
    """Sends a message to the Telegram chat."""
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            requests.get(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                         params={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=15)
        except Exception as e:
            logging.warning(f"Telegram error: {e}")

def quantize(value: float, step: float, minimum: float) -> float:
    """Quantizes a value (like order quantity) to exchange requirements."""
    if step <= 0: return max(value, minimum)
    k = math.floor(value / step)
    return max(minimum, k * step)

def get_portfolio_summary() -> str:
    """Fetches a quick summary of current equity."""
    try:
        wb = session.get_wallet_balance(accountType="UNIFIED")['result']['list'][0]
        equity = float(wb.get('totalEquity', 0) or 0)
        return f"Portfolio: {equity:.2f} USDT | Symbols: {len(SYMBOLS)} active"
    except:
        return "Portfolio: N/A"

# /status helper
_status_cache = {"t": 0, "data": {}}
def get_status() -> dict:
    """Gathers real-time operational status for the health server."""
    global _status_cache
    now = time.time()
    if now - _status_cache["t"] < 60: # Cache for 60 seconds
        return _status_cache["data"]
    
    status = {"equity": None, "positions": [], "open_orders": {}, "signals": {}}
    
    # Equity
    try:
        wb = session.get_wallet_balance(accountType="UNIFIED")["result"]["list"][0]
        status["equity"] = float(wb.get("totalEquity", 0) or 0)
    except Exception as e:
        status["equity"] = f"error: {e}"
        
    for sym in SYMBOLS:
        # Positions
        try:
            pos_list = session.get_position(category="linear", symbol=sym)["result"]["list"]
            if pos_list:
                p = pos_list[0]
                size = float(p.get("size", 0) or 0)
                if abs(size) > 0:
                    status["positions"].append({
                        "symbol": sym, "side": p.get("side"), "size": size,
                        "avgPrice": float(p.get("avgPrice", 0) or 0),
                        "unrealisedPnl": float(p.get("unrealisedPnl", 0) or 0)
                    })
        except:
            pass
            
        # Open orders
        try:
            oo = session.get_open_orders(category="linear", symbol=sym)["result"]["list"]
            status["open_orders"][sym] = len(oo)
        except:
            status["open_orders"][sym] = 0
            
        # Live signal snapshot
        try:
            df = fetch_data(sym)
            cur = float(df['close'].iloc[-1])
            pred = predict_price(df) # Predict price returns a standard Python float
            lev, risk_score, state, tp, sl, gate = assess_risk(df, sym, total_trade_counts.get(sym, 0))
            status["signals"][sym] = {
                # Ensure all numerical values are explicitly converted/rounded to standard float/int
                "delta_pct": float(round((pred/cur - 1)*100, 3)),
                "risk_score": float(round(risk_score, 3)),
                "state": state,
                "tp_pct": float(round(tp*100, 3)),
                "sl_pct": float(round(sl*100, 3)),
                "gate_std_pct": float(round(gate*100, 3)),
                "lev": int(lev)
            }
        except Exception as e:
            status["signals"][sym] = {"error": str(e)}
            
    _status_cache = {"t": now, "data": status}
    return status

# -------- Health server --------
def start_health_server():
    """Starts a simple HTTP server for health checks and status display."""
    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", "/health"):
                self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
            elif self.path == "/status":
                try:
                    st = get_status()
                    self.send_response(200); self.send_header("Content-Type","application/json"); self.end_headers()
                    # Use the corrected SafeJSONEncoder here
                    self.wfile.write(json.dumps(st, indent=2, cls=SafeJSONEncoder).encode())
                except Exception as e:
                    self.send_response(500); self.end_headers(); self.wfile.write(f"Status error: {e}".encode())
            else:
                self.send_response(404); self.end_headers()
        def log_message(self, *a, **k): pass # Suppress default logging
        
    try:
        port = int(os.getenv("PORT", "10000"))
        logging.info(f"Starting health server on port {port}")
        HTTPServer(("0.0.0.0", port), HealthHandler).serve_forever()
    except Exception as e:
        logging.warning(f"Health server failed: {e}")

# -------- Trading actions --------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def execute_trade(symbol: str, action: str, qty: float, leverage: int,
                  entry_price: float, entry_time: pd.Timestamp,
                  delta_pct: float, risk: float):
    """Executes a market entry trade (long/short)."""
    # 1. Set leverage (must be done before order)
    leverage = int(max(LEVERAGE_MIN, min(leverage, LEVERAGE_MAX)))
    try:
        session.set_leverage(category="linear", symbol=symbol,
                             buy_leverage=str(leverage), sell_leverage=str(leverage))
    except Exception as e:
        if "110043" not in str(e): # Ignore 'no position' error if setting leverage
            logging.error(f"Leverage set error for {symbol}: {e}")
            
    # 2. Place market order
    side = "Buy" if action == "long" else "Sell"
    try:
        order = session.place_order(
            category="linear", symbol=symbol, side=side,
            order_type="Market", qty=str(qty),
            positionIdx=0, reduce_only=False
        )
    except TypeError:
        order = session.place_order( # Fallback for older client libraries
            category="linear", symbol=symbol, side=side,
            order_type="Market", qty=str(qty),
            positionIdx=0, reduceOnly=False
        )
        
    if order.get('retCode', 1) != 0:
        logging.error(f"Order failed for {symbol}: {order.get('retMsg')}")
        return None, None, None
        
    try:
        # Use average price from execution, or fallback to current price
        avg_entry = float(order['result'].get('avgPrice', entry_price))
    except Exception:
        avg_entry = entry_price
        
    logging.info(f"ENTRY {action} {symbol} qty={qty}, lev={leverage}: avg_price={avg_entry}")
    send_telegram(f"ENTRY {action.upper()} {symbol} Δ={delta_pct:.2f}% risk={risk:.2f} qty={qty} lev={leverage} @ {avg_entry:.2f} | {get_portfolio_summary()}")
    return side, order, avg_entry

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def close_position(symbol: str, open_side: str, qty: float,
                   entry_price: float, entry_time: pd.Timestamp,
                   reason: str, exit_price: float):
    """Closes an open position with a market order."""
    exit_side = "Sell" if open_side == "Buy" else "Buy"
    try:
        order = session.place_order(
            category="linear", symbol=symbol, side=exit_side,
            order_type="Market", qty=str(qty),
            positionIdx=0, reduce_only=True
        )
    except TypeError:
        order = session.place_order(
            category="linear", symbol=symbol, side=exit_side,
            order_type="Market", qty=str(qty),
            positionIdx=0, reduceOnly=True
        )
        
    if order.get('retCode', 1) != 0:
        logging.error(f"Close order failed for {symbol}: {order.get('retMsg')}")
        return False, 0.0
        
    try:
        avg_exit = float(order['result'].get('avgPrice', exit_price))
    except Exception:
        avg_exit = exit_price
        
    # Calculate P&L for Q-learning reward (simplified: no fees/funding)
    if open_side == "Buy":
        realized_pnl = qty * (avg_exit - entry_price)
    else: # Sell
        realized_pnl = qty * (entry_price - avg_exit)
        
    logging.info(f"CLOSE {open_side} {symbol} qty={qty} ({reason}): P&L {realized_pnl:.2f}")
    send_telegram(f"CLOSE {open_side} {symbol} reason={reason} PnL={realized_pnl:.2f} | {get_portfolio_summary()}")
    
    # Reward is normalized P&L percentage based on initial notional
    notional = qty * entry_price
    reward = realized_pnl / notional if notional > 0 else 0.0
    return True, reward

# -------- Globals (declared once, mutated in main loop) --------
last_trade_times: Dict[str, float] = {sym: 0.0 for sym in SYMBOLS}
trade_counts_today: Dict[str, int] = {sym: 0 for sym in SYMBOLS}
total_trade_counts: Dict[str, int] = {sym: 0 for sym in SYMBOLS}
last_day_bucket = 0
last_week_bucket = 0
initial_balance = 0.0
cumulative_pnl = 0.0
# Stores: (side, qty, entry_price, entry_time, leverage_used, state_at_entry)
open_positions: Dict[str, Tuple] = {} 
no_signal_minutes = 0

# -------- Boot diag --------
def boot_diag():
    """Performs startup diagnostics."""
    logging.info("=== BOOT DIAG START (Multi-Symbol) ===")
    logging.info(f"TESTNET={TESTNET} SYMBOLS={SYMBOLS} BOT_PAUSED={BOT_PAUSED} RISK_PER_TRADE={RISK_PER_TRADE*100:.2f}%")
    try:
        wb = session.get_wallet_balance(accountType="UNIFIED")['result']['list'][0]
        global initial_balance
        initial_balance = float(wb.get('totalEquity', 0) or 0)
        logging.info(f"Wallet Check: equity={initial_balance:.6f}")
    except Exception as e:
        logging.warning(f"Wallet Check: error: {e}")
    for sym in SYMBOLS[:2]:
        try:
            k = session.get_kline(category='linear', symbol=sym, interval='15', limit=2)['result']['list']
            logging.info(f"Kline Check {sym}: last_close={float(k[-1][4]):.2f}")
        except Exception as e:
            logging.warning(f"Kline Check {sym}: {e}")
    logging.info("=== BOOT DIAG END ===")

def adopt_open_positions():
    """Checks for and adopts any positions open on the exchange at startup."""
    try:
        for sym in SYMBOLS:
            try:
                pos_list = session.get_position(category="linear", symbol=sym)["result"]["list"]
            except Exception:
                pos_list = []
            if pos_list:
                p = pos_list[0]
                size = float(p.get("size", 0) or 0)
                if abs(size) > 0:
                    side = p.get("side")
                    avg = float(p.get("avgPrice", 0) or 0)
                    try:
                        lev = int(float(p.get("leverage", LEVERAGE_MIN)))
                    except Exception:
                        lev = LEVERAGE_MIN
                    lev = max(LEVERAGE_MIN, min(LEVERAGE_MAX, lev))
                    # Use a placeholder 'adopt' state if actual state is unknown
                    open_positions[sym] = (side, abs(size), avg, pd.Timestamp.now(tz="UTC"), lev, "adopt") 
                    logging.info(f"Adopted {sym}: side={side} size={size} avg={avg} lev={lev}")
    except Exception as e:
        logging.warning(f"Adopt positions error: {e}")

# -------- Main loop --------
def main():
    global last_day_bucket, last_week_bucket, cumulative_pnl, no_signal_minutes
    load_q_tables()
    
    # Get initial balance for risk calculation
    try:
        wb = session.get_wallet_balance(accountType="UNIFIED")["result"]["list"][0]
        balance = float(wb.get("totalEquity", 0) or 0)
    except Exception:
        balance = 1000.0
    global initial_balance
    initial_balance = balance

    while True:
        try:
            if BOT_PAUSED:
                time.sleep(60); continue

            # Weekly P&L reset
            week_bucket = int(time.time() // (7 * 86400))
            if RESET_PNL_WEEKLY and week_bucket != last_week_bucket:
                cumulative_pnl = 0.0
                last_week_bucket = week_bucket
                send_telegram(f"Weekly P&L Reset | {get_portfolio_summary()}")

            # Daily symbol refresh & trade counter reset
            day_bucket = int(time.time() // 86400)
            if day_bucket != last_day_bucket:
                if not ENV_SYMBOLS_FIXED:
                    new_syms = get_top_symbols()
                    for d, default in [(last_trade_times, 0.0),
                                       (trade_counts_today, 0),
                                       (total_trade_counts, 0)]:
                        for k in list(d.keys()):
                            if k not in new_syms: d.pop(k)
                        for k in new_syms: d.setdefault(k, default)
                    SYMBOLS[:] = new_syms[:]
                for s in SYMBOLS:
                    trade_counts_today[s] = 0
                last_day_bucket = day_bucket
                send_telegram(f"Daily Reset | Symbols: {SYMBOLS} | Cum P&L: {cumulative_pnl:.2f} | {get_portfolio_summary()}")

            # Per-symbol processing
            for symbol in SYMBOLS:
                # Cooldown and daily trade limit check
                if (time.time() - last_trade_times[symbol]) < COOLDOWN_SEC or trade_counts_today[symbol] >= MAX_TRADES_PER_DAY:
                    continue

                df = fetch_data(symbol)
                if df.empty:
                    logging.info(f"Skip {symbol}: no data after sanitize")
                    continue

                current_price = float(df["close"].iloc[-1])
                if current_price <= 0 or not np.isfinite(current_price):
                    logging.info(f"Skip {symbol}: invalid price {current_price}")
                    continue

                predicted = predict_price(df)
                leverage, risk_score, state, tp_pct, sl_pct, gate_std = assess_risk(df, symbol, total_trade_counts.get(symbol, 0))
                rsi = float(df["rsi"].iloc[-1]); ma50 = float(df["ma50"].iloc[-1])
                vol = float(df["volatility"].iloc[-1])
                
                # Update current balance for accurate sizing
                try:
                    wb = session.get_wallet_balance(accountType="UNIFIED")["result"]["list"][0]
                    balance = float(wb.get("totalEquity", 0) or 0)
                except Exception:
                    balance = initial_balance


                # Exits (Monitor for TP/SL/Time Stop)
                if symbol in open_positions:
                    open_side, qty, entry_price, entry_time, lev_used, state_at_entry = open_positions[symbol]
                    tp = entry_price * (1.0 + tp_pct if open_side == "Buy" else 1.0 - tp_pct)
                    sl = entry_price * (1.0 - sl_pct if open_side == "Buy" else 1.0 + sl_pct)
                    hours_open = (pd.Timestamp.now(tz='UTC') - entry_time).total_seconds() / 3600.0
                    reason = None
                    if (open_side == "Buy" and current_price >= tp) or (open_side == "Sell" and current_price <= tp):
                        reason = "TP"
                    elif (open_side == "Buy" and current_price <= sl) or (open_side == "Sell" and current_price >= sl):
                        reason = "SL"
                    elif hours_open >= TIME_STOP_HOURS:
                        reason = "Time Stop"
                        
                    if reason:
                        # Close position and update Q-table with reward
                        ok, reward = close_position(symbol, open_side, qty, entry_price, entry_time, reason, current_price)
                        if ok:
                            cumulative_pnl += reward * (qty * entry_price) # Add raw P&L
                            update_q(symbol, reward, state_at_entry, lev_used) # RL update
                            open_positions.pop(symbol)
                            last_trade_times[symbol] = time.time()
                        continue

                # Entries (Signal Generation)
                delta = (predicted / current_price) - 1.0
                delta_pct = delta * 100.0
                action = None
                
                # Signal logic
                if abs(delta) >= 0.01 and risk_score < MIN_RISK:
                    # High-confidence override (1% move predicted, low risk)
                    action = "long" if delta > 0 else "short"
                elif abs(delta) >= gate_std and risk_score < MIN_RISK:
                    # Standard gate with indicator alignment
                    align_and = (ENTRY_ALIGN_MODE == "AND")
                    
                    long_cond_ma = (current_price > ma50)
                    long_cond_rsi = (rsi > 50.0)
                    long_ok = (long_cond_ma and long_cond_rsi) if align_and else (long_cond_ma or long_cond_rsi)
                    
                    short_cond_ma = (current_price < ma50)
                    short_cond_rsi = (rsi < 50.0)
                    short_ok = (short_cond_ma and short_cond_rsi) if align_and else (short_cond_ma or short_cond_rsi)
                    
                    if long_ok and delta > 0: action = "long"
                    elif short_ok and delta < 0: action = "short"

                if action:
                    try:
                        info = session.get_instruments_info(category="linear", symbol=symbol)["result"]["list"][0]
                        min_qty = float(info["lotSizeFilter"]["minOrderQty"])
                        qty_step = float(info["lotSizeFilter"]["qtyStep"])
                    except Exception as e:
                        logging.error(f"Instruments info error {symbol}: {e}")
                        continue

                    
                    # --- CRITICAL FIX: SAFE, RISK-BASED SIZING LOGIC (Risk-Per-Trade) ---
                    # 1. Calculate the absolute risk amount ($)
                    risk_amount = balance * RISK_PER_TRADE

                    # 2. Calculate Stop Loss Distance in Price ($ per unit)
                    # sl_pct comes from adaptive_thresholds
                    stop_loss_dist_price = current_price * sl_pct
                    if stop_loss_dist_price <= 0:
                        logging.warning(f"Skip {symbol}: invalid SL distance {stop_loss_dist_price}")
                        continue

                    # 3. Calculate Quantity based purely on Risk: Qty = Risk_USDT / SL_Dist_USDT
                    raw_qty_risk = risk_amount / stop_loss_dist_price
                    
                    # 4. Apply Notional Exposure Cap (sanity check)
                    # Cap notional based on max leverage and soft margin percentage (BASE_POSITION_PCT)
                    max_notional_cap = balance * LEVERAGE_MAX * BASE_POSITION_PCT
                    raw_qty_cap = max_notional_cap / max(1e-9, current_price)

                    # 5. Final raw qty is the smaller of the two (safer)
                    raw_qty = min(raw_qty_risk, raw_qty_cap)
                    
                    # 6. Quantize to exchange requirements
                    qty = quantize(raw_qty, qty_step, min_qty)
                    # -------------------------------------------------------------------

                    # Final check before execution
                    notional_check = qty * current_price
                    if qty >= min_qty and notional_check >= MIN_NOTIONAL_USDT:
                        entry_time = pd.Timestamp.now(tz='UTC')
                        res = execute_trade(symbol, action, qty, leverage, current_price, entry_time, delta_pct, risk_score)
                        
                        if not res or res[0] is None: continue
                        
                        open_side, order, avg_entry = res
                        open_positions[symbol] = (open_side, qty, avg_entry, entry_time, leverage, state)
                        last_trade_times[symbol] = time.time()
                        trade_counts_today[symbol] = trade_counts_today.get(symbol, 0) + 1
                        total_trade_counts[symbol] = total_trade_counts.get(symbol, 0) + 1
                    else:
                        logging.info(
                            f"Skipped {symbol}: Notional {notional_check:.2f} < min {MIN_NOTIONAL_USDT} "
                            f"or Qty {qty:.6f} < min {min_qty:.6f}"
                        )
                else:
                    # Log when no signal is found
                    logging.info(
                        f"No entry {symbol}: Δ={delta_pct:.2f}% risk={risk_score:.2f} "
                        f"rsi={rsi:.2f} px={current_price:.2f} ma50={ma50:.2f} "
                        f"gate={(gate_std*100):.2f}% vol={(vol*100):.2f}%"
                    )
                    no_signal_minutes += 1
                    if no_signal_minutes >= 48 * 60:
                        send_telegram(f"Alive: 48h low signals | Cum P&L: {cumulative_pnl:.2f} | {get_portfolio_summary()}")
                        no_signal_minutes = 0

            # Global drawdown guard
            try:
                wb = session.get_wallet_balance(accountType="UNIFIED")["result"]["list"][0]
                balance = float(wb.get("totalEquity", 0) or 0)
            except Exception:
                balance = initial_balance
                
            if (initial_balance - balance) / max(1e-9, initial_balance) > DRAWDOWN_STOP:
                logging.warning("Global drawdown limit reached. Stopping.")
                send_telegram(f"Bot stop: drawdown limit reached. P&L: {cumulative_pnl:.2f}")
                break

        except Exception as e:
            logging.error(f"Main loop error: {e}")

        # Pacing: ~12s/loop for 10 symbols to ensure API rate limits are safe
        time.sleep(max(1.0, 120 / max(1, len(SYMBOLS))))

# -------- Entrypoint --------
if __name__ == "__main__":
    try:
        # Start the non-blocking health server thread
        threading.Thread(target=start_health_server, daemon=True).start()
    except Exception as e:
        logging.warning(f"Health server thread error: {e}")
    try:
        # Load models and Q-tables
        load_artifacts()
        boot_diag()
        adopt_open_positions()
    except Exception as e:
        logging.warning(f"Bot initialization error: {e}")
    # Start the main trading loop
    main()