# === Bybit Bot: stable main.py (stdout logging, /health, no ws.ping, Grok strategy) ===
import os
import sys
import time
import math
import threading
import logging
import pickle
from typing import Tuple, Optional
import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pybit.unified_trading import HTTP, WebSocket
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from tenacity import retry, stop_after_attempt, wait_exponential

# -------- Logging to stdout (for Render Logs tab) --------
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------- Health server so Render keeps service up --------
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
        def log_message(self, *args, **kwargs):  # silence default access logs
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
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
TESTNET = os.getenv("TESTNET", "true").lower() == "true"
BOT_PAUSED = os.getenv("BOT_PAUSED", "false").lower() == "true"  # true=paused
RECV_WINDOW = int(os.getenv("RECV_WINDOW", "60000"))
LEVERAGE_MAX = int(os.getenv("LEVERAGE_MAX", "20"))  # 20x cap on testnet
POSITION_SIZE_PCT = float(os.getenv("POSITION_SIZE_PCT", "0.05"))  # 5% notional
DRAWDOWN_STOP = float(os.getenv("DRAWDOWN_STOP", "0.20"))  # 20% stop
COOLDOWN_SEC = 15 * 60
MAX_TRADES_PER_DAY = 10
TIME_STOP_HOURS = 4
TP_PCT = 0.015
SL_PCT = 0.01
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
session = HTTP(testnet=TESTNET, api_key=API_KEY, api_secret=API_SECRET, recv_window=RECV_WINDOW)

# Optional WS (not required for this bot)
try:
    ws = WebSocket(testnet=TESTNET, channel_type="linear")
except Exception as e:
    ws = None
    logging.warning(f"WebSocket init failed (non-fatal): {e}")

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
    # Torch 2.2 compatible: no weights_only, map to CPU
    state = torch.load("lstm.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("gb.pkl", "rb") as f:
        gb_model = pickle.load(f)
    logging.info("Artifacts loaded: lstm.pth, scaler.pkl, gb.pkl")

# -------- Indicators & data --------
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_data() -> pd.DataFrame:
    res = session.get_kline(category="linear", symbol=SYMBOL, interval="60", limit=200)
    rows = res["result"]["list"]
    # V5 returns: [start, open, high, low, close, volume, turnover] as strings
    cols = ["start", "open", "high", "low", "close", "volume", "turnover"]
    df = pd.DataFrame(rows, columns=cols)
    for c in ["open", "high", "low", "close", "volume", "turnover"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["time"] = pd.to_datetime(pd.to_numeric(df["start"], errors="coerce"), unit="ms")
    df = df.dropna(subset=["close"]).sort_values("time").reset_index(drop=True)
    # Indicators
    df["ma50"] = df["close"].rolling(50, min_periods=50).mean()
    df["rsi"] = compute_rsi(df["close"])
    df["volatility"] = df["close"].pct_change().rolling(20, min_periods=20).std()
    df = df.dropna().reset_index(drop=True)
    return df

def predict_price(df: pd.DataFrame) -> float:
    assert scaler is not None and gb_model is not None
    feats = df[["open", "high", "low", "close", "volume", "ma50", "rsi", "volatility"]].values.astype(np.float32)
    X_scaled = scaler.transform(feats)  # (n, 8)
    # simple last-window feed to LSTM head
    x_seq = torch.tensor(X_scaled[-1:].reshape(1, -1, 8), dtype=torch.float32)  # (1, 1, 8)
    with torch.no_grad():
        lstm_pred = float(model(x_seq).item())
    gb_pred = float(gb_model.predict(feats[-1:].astype(np.float64))[0])
    pred = (lstm_pred + gb_pred) / 2.0
    # Clamp to ±10% around current
    cur = float(df["close"].iloc[-1])
    pred = max(cur * 0.9, min(cur * 1.1, pred))
    return float(pred)

# -------- Risk & leverage --------
q_table = {}

def get_state(vol: float) -> str:
    return "low" if vol < 0.01 else ("med" if vol < 0.03 else "high")

def choose_leverage(state: str, risk_score: float, epsilon: float = 0.0) -> int:
    # Greedy adaptive: min(LEVERAGE_MAX, max(1, 50/(risk+1)))
    lev = int(max(1, min(LEVERAGE_MAX, 50.0 / (risk_score + 1.0))))
    return lev

def assess_risk(df: pd.DataFrame) -> Tuple[int, float, str]:
    vol = float(df["volatility"].iloc[-1])
    state = get_state(vol)
    risk_score = vol * 100.0
    lev = choose_leverage(state, risk_score)
    return lev, risk_score, state

def update_q(reward: float, state: str, action_lev: int):
    if state not in q_table:
        q_table[state] = [0.0] * LEVERAGE_MAX
    idx = max(1, min(action_lev, LEVERAGE_MAX)) - 1
    q_table[state][idx] += 0.1 * (reward - q_table[state][idx])

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

# -------- Trading actions --------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def execute_trade(action: str, qty: float, leverage: int, entry_price: float, entry_time: pd.Timestamp, delta_pct: float, risk: float):
    try:
        session.set_leverage(category="linear", symbol=SYMBOL, buy_leverage=str(leverage), sell_leverage=str(leverage))
    except Exception as e:
        if "110043" not in str(e):
            logging.error(f"Leverage set error: {e}")
    side = "Buy" if action == "long" else "Sell"
    order = session.place_order(
        category="linear",
        symbol=SYMBOL,
        side=side,
        order_type="Market",
        qty=str(qty),
        positionIdx=0,
        reduce_only=False
    )
    if order['retCode'] != 0:
        logging.error(f"Order failed: {order['retMsg']}")
        return None, None, None
    try:
        avg_entry = float(order['result']['avgPrice'])
    except (KeyError, ValueError):
        avg_entry = entry_price
    logging.info(f"ENTRY {action} qty={qty}, lev={leverage}: {order}, avg_price={avg_entry}")
    send_telegram(f"ENTRY {action.upper()} {SYMBOL} Δ={delta_pct:.2f}% risk={risk:.2f} qty={qty} lev={leverage} @ {avg_entry:.2f}")
    return side, order, avg_entry

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def close_position(open_side: str, qty: float, entry_price: float, entry_time: pd.Timestamp, reason: str, exit_price: float):
    exit_side = "Sell" if open_side == "Buy" else "Buy"
    order = session.place_order(
        category="linear",
        symbol=SYMBOL,
        side=exit_side,
        order_type="Market",
        qty=str(qty),
        positionIdx=0,
        reduce_only=True
    )
    if order['retCode'] != 0:
        logging.error(f"Close order failed: {order['retMsg']}")
        return False, 0.0
    try:
        avg_exit = float(order['result']['avgPrice'])
    except (KeyError, ValueError):
        avg_exit = exit_price
    if open_side == "Buy":
        realized_pnl = qty * (avg_exit - entry_price)
    else:
        realized_pnl = qty * (entry_price - avg_exit)
    logging.info(f"CLOSE {open_side} qty={qty} ({reason}): {order}, P&L: {realized_pnl:.2f}")
    send_telegram(f"CLOSE {open_side} {SYMBOL} qty={qty} reason={reason} entry={entry_price:.2f} PnL={realized_pnl:.2f}")
    notional = qty * entry_price
    reward = realized_pnl / notional if notional > 0 else 0.0
    return True, reward

# -------- Main loop --------
last_trade_time = 0.0
trade_count_today = 0
last_day_bucket = 0

def boot_diag():
    try:
        logging.info("=== BOOT DIAG START ===")
        logging.info(f"TESTNET={TESTNET} SYMBOL={SYMBOL} BOT_PAUSED={BOT_PAUSED}")
        try:
            df = fetch_data()
            cur = float(df['close'].iloc[-1])
            pred = float(predict_price(df))
            lev, risk, state = assess_risk(df)
            logging.info(f"data_ok rows={len(df)} last={cur:.2f} pred={pred:.2f} Δ%={(pred/cur-1)*100:.2f} lev={lev} risk={risk:.2f} state={state}")
        except Exception as e:
            logging.warning(f"data_error: {e}")
        try:
            wb = session.get_wallet_balance(accountType="UNIFIED")['result']['list'][0]
            logging.info(f"wallet_ok equity={float(wb.get('totalEquity',0)):.6f}")
        except Exception as e:
            logging.warning(f"wallet_error: {e}")
        try:
            k = session.get_kline(category='linear', symbol=SYMBOL, interval='60', limit=2)['result']['list']
            logging.info(f"kline_ok last_close={float(k[-1][4])}")
        except Exception as e:
            logging.warning(f"kline_error: {e}")
        logging.info("=== BOOT DIAG END ===")
    except Exception as e:
        logging.warning(f"boot_diag error: {e}")

def main():
    global last_trade_time, trade_count_today, last_day_bucket
    try:
        equity_raw = session.get_wallet_balance(accountType="UNIFIED")["result"]["list"][0]["totalEquity"]
        balance = float(equity_raw)
    except Exception:
        balance = 0.0
    initial_balance = balance
    open_position = None  # (open_side, qty, entry_price, entry_time, lev_used, state_at_entry)
    no_signal_minutes = 0
    while True:
        try:
            if BOT_PAUSED:
                time.sleep(60)
                continue
            # Reset daily counter
            day_bucket = int(time.time() // 86400)
            if day_bucket != last_day_bucket:
                trade_count_today = 0
                last_day_bucket = day_bucket
            df = fetch_data()
            current_price = float(df["close"].iloc[-1])
            predicted = float(predict_price(df))
            leverage, risk, state = assess_risk(df)
            rsi = float(df["rsi"].iloc[-1])
            ma50 = float(df["ma50"].iloc[-1])
            # Cooldown / trade cap
            if (time.time() - last_trade_time) < COOLDOWN_SEC or trade_count_today >= MAX_TRADES_PER_DAY:
                time.sleep(60)
                continue
            # Exits first
            if open_position:
                open_side, qty, entry_price, entry_time, lev_used, state_at_entry = open_position
                tp = entry_price * (1.0 + TP_PCT if open_side == "Buy" else 1.0 - TP_PCT)
                sl = entry_price * (1.0 - SL_PCT if open_side == "Buy" else 1.0 + SL_PCT)
                hours_open = (pd.Timestamp.now(tz='UTC') - entry_time).total_seconds() / 3600.0
                reason = None
                if (open_side == "Buy" and current_price >= tp) or (open_side == "Sell" and current_price <= tp):
                    reason = "TP"
                elif (open_side == "Buy" and current_price <= sl) or (open_side == "Sell" and current_price >= sl):
                    reason = "SL"
                elif hours_open >= TIME_STOP_HOURS:
                    reason = "Time Stop"
                if reason:
                    exit_price = current_price  # approx
                    ok, reward = close_position(open_side, qty, entry_price, entry_time, reason, exit_price)
                    if ok:
                        open_position = None
                        last_trade_time = time.time()
                        trade_count_today += 1
                        update_q(reward, state_at_entry, lev_used)
                    time.sleep(60)
                    continue
            # Entries
            delta_pct = (predicted / current_price - 1.0) * 100.0
            action = None
            # Override: ignore RSI/MA
            if abs(delta_pct) >= 1.0 and risk < 5:
                action = "long" if delta_pct > 0 else "short"
            # Standard: require alignment with delta direction
            elif abs(delta_pct) >= 0.5 and risk < 5:
                if delta_pct > 0 and (rsi > 50.0 and current_price > ma50):
                    action = "long"
                elif delta_pct < 0 and (rsi < 50.0 and current_price < ma50):
                    action = "short"
            if action:
                try:
                    info = session.get_instruments_info(category="linear", symbol=SYMBOL)["result"]["list"][0]
                    min_qty = float(info["lotSizeFilter"]["minOrderQty"])
                    qty_step = float(info["lotSizeFilter"]["qtyStep"])
                except Exception as e:
                    logging.error(f"Instruments info error: {e}")
                    time.sleep(60)
                    continue
                try:
                    balance_raw = session.get_wallet_balance(accountType="UNIFIED")["result"]["list"][0]["totalEquity"]
                    balance = float(balance_raw)
                except Exception:
                    balance = 0.0
                raw_qty = (balance * POSITION_SIZE_PCT) / max(1e-9, current_price)
                qty = quantize(raw_qty, qty_step, min_qty)
                if qty >= min_qty:
                    entry_time = pd.Timestamp.now(tz='UTC')
                    res = execute_trade(
                        action=action,
                        qty=qty,
                        leverage=leverage,
                        entry_price=current_price,
                        entry_time=entry_time,
                        delta_pct=delta_pct,
                        risk=risk
                    )
                    if res[0] is None:
                        time.sleep(60)
                        continue
                    open_side, order, avg_entry = res
                    open_position = (open_side, qty, avg_entry, entry_time, leverage, state)
                    last_trade_time = time.time()
                else:
                    logging.info(f"Skipped: qty {qty} < min {min_qty}")
            else:
                logging.info(f"No entry: Δ={delta_pct:.2f}% risk={risk:.2f} rsi={rsi:.2f} px={current_price:.2f} ma50={ma50:.2f}")
                no_signal_minutes += 1
                if no_signal_minutes >= 48 * 60:
                    send_telegram(f"Alive: 48h no trades. Last Δ={delta_pct:.2f}% risk={risk:.2f}")
                    no_signal_minutes = 0
            try:
                equity_raw = session.get_wallet_balance(accountType="UNIFIED")["result"]["list"][0]["totalEquity"]
                balance = float(equity_raw)
            except Exception:
                balance = initial_balance  # fallback
            if (initial_balance - balance) / max(1e-9, initial_balance) > DRAWDOWN_STOP:
                logging.warning("Drawdown limit reached. Stopping.")
                send_telegram("Bot stop: drawdown limit reached.")
                break
        except Exception as e:
            logging.error(f"Main loop error: {e}")
        time.sleep(60)

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