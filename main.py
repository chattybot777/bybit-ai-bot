import os
import sys
import time
import math
import threading
import logging
import pickle
from typing import Tuple, Optional, Dict, List
import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pybit.unified_trading import HTTP
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from tenacity import retry, stop_after_attempt, wait_exponential

# -------- Logging to stdout --------
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------- Health server --------
from http.server import BaseHTTPRequestHandler, HTTPServer

def start_health_server():
    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", "/health"):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")
            else:
                self.send_response(404)
                self.end_headers()
        def log_message(self, *args, **kwargs):
            pass
    try:
        port = int(os.getenv("PORT", "10000"))
        HTTPServer(("0.0.0.0", port), HealthHandler).serve_forever()
    except Exception as e:
        logging.warning(f"health server failed: {e}")

# -------- Env & Bybit session --------
load_dotenv()
torch.set_num_threads(1)
API_KEY = os.getenv("BYBIT_API_KEY", "")
API_SECRET = os.getenv("BYBIT_API_SECRET") or os.getenv("API_SECRET") or ""
TESTNET = os.getenv("TESTNET", "true").lower() == "true"
BOT_PAUSED = os.getenv("BOT_PAUSED", "false").lower() == "true"
RECV_WINDOW = int(os.getenv("RECV_WINDOW", "60000"))
LEVERAGE_MAX = int(os.getenv("LEVERAGE_MAX", "20"))
BASE_POSITION_PCT = float(os.getenv("BASE_POSITION_PCT", "0.05"))
DRAWDOWN_STOP = float(os.getenv("DRAWDOWN_STOP", "0.20"))
RESET_PNL_WEEKLY = os.getenv("RESET_PNL_WEEKLY", "false").lower() == "true"
COOLDOWN_SEC = 15 * 60
MAX_TRADES_PER_DAY = 10
TIME_STOP_HOURS = 4
BASE_TP_PCT = 0.015
BASE_SL_PCT = 0.01
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
session = HTTP(testnet=TESTNET, api_key=API_KEY, api_secret=API_SECRET, recv_window=RECV_WINDOW)

# -------- Dynamic Symbol Selection --------
def get_top_symbols() -> List[str]:
    try:
        res = session.get_tickers(category="linear")["result"]["list"]
        usdt_pairs = [t["symbol"] for t in res if t["symbol"].endswith("USDT")]
        sorted_pairs = sorted(
            usdt_pairs,
            key=lambda x: float([t for t in res if t["symbol"] == x][0].get("turnover24h", 0)),
            reverse=True
        )
        return sorted_pairs[:10]
    except Exception as e:
        logging.warning(f"Symbol fetch error: {e}")
        return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "TRXUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT"]

SYMBOLS: List[str] = get_top_symbols()

# -------- Models --------
class LSTMModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=50, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = LSTMModel()
scaler: Optional[MinMaxScaler] = None
gb_model: Optional[GradientBoostingRegressor] = None

def load_artifacts():
    global model, scaler, gb_model
    state = (__import__("torch").load("lstm.pth", map_location="cpu", weights_only=True)  if "weights_only" in __import__("inspect").signature(__import__("torch").load).parameters  else __import__("torch").load("lstm.pth", map_location="cpu"))
    model.load_state_dict(state)
    model.eval()
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("gb.pkl", "rb") as f:
        gb_model = pickle.load(f)
    logging.info("Artifacts loaded: Shared models for multi-symbol")

# -------- Q-table persistence --------
q_tables: Dict[str, Dict[str, List[float]]] = {}
def load_q_tables():
    global q_tables
    try:
        with open("q_tables.pkl", "rb") as f:
            q_tables = pickle.load(f)
        logging.info("Loaded Q-tables")
    except:
        q_tables = {}
        logging.info("Initialized empty Q-tables")
def save_q_tables():
    try:
        with open("q_tables.pkl", "wb") as f:
            pickle.dump(q_tables, f)
        logging.info("Saved Q-tables")
    except Exception as e:
        logging.warning(f"Q-table save error: {e}")

# -------- Indicators & data --------
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_data(symbol: str) -> pd.DataFrame:
    res = session.get_kline(category="linear", symbol=symbol, interval="60", limit=200)
    rows = res["result"]["list"]
    cols = ["start", "open", "high", "low", "close", "volume", "turnover"]
    df = pd.DataFrame(rows, columns=cols)
    for c in ["open", "high", "low", "close", "volume", "turnover"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["time"] = pd.to_datetime(pd.to_numeric(df["start"], errors="coerce"), unit="ms")
    df = df.dropna(subset=["close"]).sort_values("time").reset_index(drop=True)
    df["ma50"] = df["close"].rolling(50, min_periods=50).mean()
    df["rsi"] = compute_rsi(df["close"])
    df["volatility"] = df["close"].pct_change().rolling(20, min_periods=20).std()
    df = df.dropna().reset_index(drop=True)
    return df

def predict_price(df: pd.DataFrame) -> float:
    assert scaler is not None and gb_model is not None
    feats = df[["open", "high", "low", "close", "volume", "ma50", "rsi", "volatility"]].values.astype(np.float32)
    X_scaled = scaler.transform(feats)
    x_seq = torch.tensor(X_scaled.reshape(1, -1, 8), dtype=torch.float32)
    with torch.no_grad():
        lstm_pred = float(model(x_seq).item())
    gb_pred = float(gb_model.predict(feats[-1:].astype(np.float64))[0])
    pred = (lstm_pred + gb_pred) / 2.0
    cur = float(df["close"].iloc[-1])
    pred = max(cur * 0.9, min(cur * 1.1, pred))
    return pred

# -------- Risk, Leverage, Adaptation --------
def get_state(vol: float) -> str:
    return "low" if vol < 0.01 else ("med" if vol < 0.03 else "high")

def adaptive_thresholds(vol: float, base: float) -> Tuple[float, float, float]:
    vol_factor = 1 + vol * 10
    tp_pct = base * vol_factor
    sl_pct = base * vol_factor * 0.67
    gate_std = max(0.05, base / vol_factor)
    return tp_pct, sl_pct, gate_std

def choose_leverage(symbol: str, state: str, risk_score: float, trade_count: int) -> int:
    if symbol not in q_tables:
        q_tables[symbol] = {"low": [0.0] * LEVERAGE_MAX, "med": [0.0] * LEVERAGE_MAX, "high": [0.0] * LEVERAGE_MAX}
    q = q_tables[symbol][state]
    epsilon = max(0.05, 0.15 * (1 - trade_count / 100))
    if np.random.rand() < epsilon:
        lev = np.random.randint(1, LEVERAGE_MAX + 1)
    else:
        lev = np.argmax(q) + 1
    base_lev = int(max(1, min(LEVERAGE_MAX, 50.0 / (risk_score + 1.0))))
    lev = min(lev, base_lev)
    return lev

def assess_risk(df: pd.DataFrame, symbol: str, trade_count: int = 0) -> Tuple[int, float, str, float, float, float]:
    vol = float(df["volatility"].iloc[-1])
    state = get_state(vol)
    risk_score = vol * 100.0
    lev = choose_leverage(symbol, state, risk_score, trade_count)
    tp_pct, sl_pct, gate_std = adaptive_thresholds(vol, BASE_TP_PCT)
    return lev, risk_score, state, tp_pct, sl_pct, gate_std

def update_q(symbol: str, reward: float, state: str, action_lev: int):
    if symbol not in q_tables:
        q_tables[symbol] = {"low": [0.0] * LEVERAGE_MAX, "med": [0.0] * LEVERAGE_MAX, "high": [0.0] * LEVERAGE_MAX}
    q = q_tables[symbol][state]
    idx = action_lev - 1
    if 0 <= idx < len(q):
        q[idx] += 0.1 * (reward - q[idx])
        save_q_tables()

# -------- Utils --------
def send_telegram(msg: str):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            requests.get(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                params={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
                timeout=15
            )
        except Exception as e:
            logging.warning(f"Telegram error: {e}")

def quantize(value: float, step: float, minimum: float) -> float:
    if step <= 0:
        return max(value, minimum)
    k = math.floor(value / step)
    q = max(minimum, k * step)
    return q

def get_portfolio_summary() -> str:
    try:
        wb = session.get_wallet_balance(accountType="UNIFIED")['result']['list'][0]
        equity = float(wb.get('totalEquity', 0))
        return f"Portfolio: {equity:.2f} USDT | Symbols: {len(SYMBOLS)} active"
    except:
        return "Portfolio: N/A"

# -------- Trading actions --------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def execute_trade(symbol: str, action: str, qty: float, leverage: int, entry_price: float, entry_time: pd.Timestamp, delta_pct: float, risk: float):
    try:
        session.set_leverage(category="linear", symbol=symbol, buy_leverage=str(leverage), sell_leverage=str(leverage))
    except Exception as e:
        if "110043" not in str(e):
            logging.error(f"Leverage set error for {symbol}: {e}")
    side = "Buy" if action == "long" else "Sell"
    order = session.place_order(
        category="linear",
        symbol=symbol,
        side=side,
        order_type="Market",
        qty=str(qty),
        positionIdx=0,
        reduce_only=False
    )
    if order['retCode'] != 0:
        logging.error(f"Order failed for {symbol}: {order['retMsg']}")
        return None, None, None
    try:
        avg_entry = float(order['result']['avgPrice'])
    except (KeyError, ValueError):
        avg_entry = entry_price
    logging.info(f"ENTRY {action} {symbol} qty={qty}, lev={leverage}: avg_price={avg_entry}")
    send_telegram(f"ENTRY {action.upper()} {symbol} Δ={delta_pct:.2f}% risk={risk_score:.2f} qty={qty} lev={leverage} @ {avg_entry:.2f} | {get_portfolio_summary()}")
    return side, order, avg_entry

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def close_position(symbol: str, open_side: str, qty: float, entry_price: float, entry_time: pd.Timestamp, reason: str, exit_price: float):
    exit_side = "Sell" if open_side == "Buy" else "Buy"
    order = session.place_order(
        category="linear",
        symbol=symbol,
        side=exit_side,
        order_type="Market",
        qty=str(qty),
        positionIdx=0,
        reduce_only=True
    )
    if order['retCode'] != 0:
        logging.error(f"Close order failed for {symbol}: {order['retMsg']}")
        return False, 0.0
    try:
        avg_exit = float(order['result']['avgPrice'])
    except (KeyError, ValueError):
        avg_exit = exit_price
    if TESTNET:
        realized_pnl = qty * (avg_exit - entry_price) if open_side == "Buy" else qty * (entry_price - avg_exit)
    else:
        try:
            pos_list = session.get_positions(category="linear", symbol=symbol)["result"]["list"]
            realized_pnl = float(pos_list[0].get("realisedPnl", 0) or 0)
        except Exception:
            realized_pnl = qty * (avg_exit - entry_price) if open_side == "Buy" else qty * (entry_price - avg_exit)
    logging.info(f"CLOSE {open_side} {symbol} qty={qty} ({reason}): P&L {realized_pnl:.2f}")
    send_telegram(f"CLOSE {open_side} {symbol} reason={reason} PnL={realized_pnl:.2f} | {get_portfolio_summary()}")
    notional = qty * entry_price
    reward = realized_pnl / notional if notional > 0 else 0.0
    return True, reward

# -------- Main loop --------
last_trade_times: Dict[str, float] = {sym: 0.0 for sym in SYMBOLS}
trade_counts_today: Dict[str, int] = {sym: 0 for sym in SYMBOLS}

last_day_bucket = 0
last_week_bucket = 0
initial_balance = 0.0
cumulative_pnl = 0.0
open_positions: Dict[str, Tuple] = {}
no_signal_minutes = 0

def boot_diag():
    logging.info("=== BOOT DIAG START (Multi-Symbol) ===")
    logging.info(f"TESTNET={TESTNET} SYMBOLS={SYMBOLS} BOT_PAUSED={BOT_PAUSED}")
    try:
        wb = session.get_wallet_balance(accountType="UNIFIED")['result']['list'][0]
        global initial_balance
        initial_balance = float(wb.get('totalEquity', 0))
        logging.info(f"wallet_ok equity={initial_balance:.6f}")
        if initial_balance < 2232:
            logging.warning(f"Low equity: Need ~2232 USDT for BTCUSDT trades")
    except Exception as e:
        logging.warning(f"wallet_error: {e}")
    for sym in SYMBOLS[:2]:
        try:
            k = session.get_kline(category='linear', symbol=sym, interval='60', limit=2)['result']['list']
            logging.info(f"kline_ok {sym} last_close={float(k[-1][4]):.2f}")
        except Exception as e:
            logging.warning(f"kline_error {sym}: {e}")
    logging.info("=== BOOT DIAG END ===")

def main():
    global last_day_bucket, last_week_bucket, cumulative_pnl, no_signal_minutes, SYMBOLS
    load_q_tables()
    try:
        wb = session.get_wallet_balance(accountType="UNIFIED")["result"]["list"][0]
        balance = float(wb.get("totalEquity", 0))
    except Exception:
        balance = 1000.0
    global initial_balance
    initial_balance = balance
    while True:
        try:
            if BOT_PAUSED:
                time.sleep(60)
                continue
            # Weekly P&L reset (Monday UTC)
            week_bucket = int(time.time() // (7 * 86400))
            if RESET_PNL_WEEKLY and week_bucket != last_week_bucket:
                cumulative_pnl = 0.0
                last_week_bucket = week_bucket
                send_telegram(f"Weekly P&L Reset | {get_portfolio_summary()}")
            # Daily symbol refresh and counters
            day_bucket = int(time.time() // 86400)
            if day_bucket != last_day_bucket:
                SYMBOLS = get_top_symbols()
                last_trade_times = {sym: 0.0 for sym in SYMBOLS}
                for s in SYMBOLS:
                    trade_counts_today[s] = 0
                for s in SYMBOLS:
                    trade_counts_today[s] = 0
                last_day_bucket = day_bucket
                send_telegram(f"Daily Reset | Symbols: {SYMBOLS} | Cumulative P&L: {cumulative_pnl:.2f} USDT | {get_portfolio_summary()}")
            # Sequential per-symbol processing
            for symbol in SYMBOLS:
                # safe defaults to avoid NameError before assess_risk
                risk_score = 999.0
                state = 'unknown'
                tp_pct = BASE_TP_PCT
                sl_pct = BASE_SL_PCT
                gate_std = BASE_TP_PCT

                if (time.time() - last_trade_times[symbol]) < COOLDOWN_SEC or trade_counts_today[symbol] >= MAX_TRADES_PER_DAY:
                    continue
                df = fetch_data(symbol)
                current_price = float(df["close"].iloc[-1])
                predicted = predict_price(df)

                rsi = float(df["rsi"].iloc[-1])
                ma50 = float(df["ma50"].iloc[-1])
                # Exits
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
                        ok, reward = close_position(symbol, open_side, qty, entry_price, entry_time, reason, current_price)
                        if ok:
# removed stray global
                            cumulative_pnl += reward * (qty * entry_price)
                            open_positions.pop(symbol)
                            last_trade_times[symbol] = time.time()
                            trade_counts_today[symbol] += 1

                            update_q(symbol, reward, state_at_entry, lev_used)
                        continue
                # Entries
                delta_pct = (predicted / current_price - 1.0) * 100.0
                action = None
                if abs(delta_pct) >= 1.0 and risk_score < 5:
                    action = "long" if delta_pct > 0 else "short"
                elif abs(delta_pct) >= gate_std and risk_score < 5:
                    if delta_pct > 0 and (rsi > 50.0 and current_price > ma50):
                        action = "long"
                    elif delta_pct < 0 and (rsi < 50.0 and current_price < ma50):
                        action = "short"
                if action:
                    try:
                        info = session.get_instruments_info(category="linear", symbol=symbol)["result"]["list"][0]
                        min_qty = float(info["lotSizeFilter"]["minOrderQty"])
                        qty_step = float(info["lotSizeFilter"]["qtyStep"])
                    except Exception as e:
                        logging.error(f"Instruments info error {symbol}: {e}")
                        continue
                    try:
                        wb = session.get_wallet_balance(accountType="UNIFIED")["result"]["list"][0]
                        balance = float(wb.get("totalEquity", 0))
                    except Exception:
                        balance = initial_balance
                    vol = float(df["volatility"].iloc[-1])
                    vol_weight = max(0.1, 1 / (1 + vol * 100))
                    growth_factor = 1 + (cumulative_pnl / max(1, initial_balance))
                    position_pct = min(BASE_POSITION_PCT * growth_factor * vol_weight, 0.10)
                    raw_qty = (balance * position_pct) / max(1e-9, current_price)
                    qty = quantize(raw_qty, qty_step, min_qty)
                    if qty >= min_qty:
                        entry_time = pd.Timestamp.now(tz='UTC')
                        res = execute_trade(symbol, action, qty, leverage, current_price, entry_time, delta_pct, risk_score)
                        if res[0] is None:
                            continue
                        open_side, order, avg_entry = res
                        open_positions[symbol] = (open_side, qty, avg_entry, entry_time, leverage, state)
                        last_trade_times[symbol] = time.time()
                    else:
                        logging.info(f"Skipped {symbol}: qty {qty} < min {min_qty}")
                else:
                    logging.info(f"No entry {symbol}: Δ={delta_pct:.2f}% risk={risk_score:.2f} rsi={rsi:.2f} px={current_price:.2f} ma50={ma50:.2f} gate={gate_std:.2f}")
                    no_signal_minutes += 1
                    if no_signal_minutes >= 48 * 60:
                        send_telegram(f"Alive: 48h low signals | Cumulative P&L: {cumulative_pnl:.2f} | {get_portfolio_summary()}")
                        no_signal_minutes = 0
            try:
                wb = session.get_wallet_balance(accountType="UNIFIED")["result"]["list"][0]
                balance = float(wb.get("totalEquity", 0))
            except Exception:
                balance = initial_balance
            if (initial_balance - balance) / max(1e-9, initial_balance) > DRAWDOWN_STOP:
                logging.warning("Global drawdown limit reached. Stopping.")
                send_telegram(f"Bot stop: drawdown limit reached. P&L: {cumulative_pnl:.2f}")
                break
        except Exception as e:
            logging.error(f"Main loop error: {e}")
        time.sleep(max(1.0, 120 / max(1, len(SYMBOLS))))

if __name__ == "__main__":
    try:
        threading.Thread(target=start_health_server, daemon=True).start()
    except Exception as e:
        logging.warning(f"health thread err: {e}")
    try:
        load_artifacts()
        boot_diag()
    except Exception as e:
        logging.warning(f"boot diag err: {e}")
    main()