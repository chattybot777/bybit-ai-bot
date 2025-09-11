# === Bybit Bot: Clean Multi-Symbol (15m), Adaptive, /status, stable globals ===
import os, sys, time, math, threading, logging, pickle, json, inspect
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
from http.server import BaseHTTPRequestHandler, HTTPServer

# -------- Logging --------
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# -------- Env & Session --------
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

# Strategy knobs
COOLDOWN_SEC = 15 * 60          # per-symbol cooldown
MAX_TRADES_PER_DAY = 10         # per symbol
TIME_STOP_HOURS = 4
BASE_TP_PCT = 0.015
BASE_SL_PCT = 0.010
MIN_GATE_STD = float(os.getenv("MIN_GATE_STD", "0.003"))   # lower gate => more trades
MIN_RISK = float(os.getenv("MIN_RISK", "4.0"))             # risk gate

# FIX: safe defaults for data hardening (no env needed; change only if you want)
VOL_FALLBACK = float(os.getenv("VOL_FALLBACK", "0.01"))     # ~1% 15m vol as a sane default
VOL_CAP      = float(os.getenv("VOL_CAP", "0.20"))          # cap 15m vol at 20%

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

session = HTTP(testnet=TESTNET, api_key=API_KEY, api_secret=API_SECRET, recv_window=RECV_WINDOW)

# -------- Symbols --------
def get_top_symbols() -> List[str]:
    try:
        r = session.get_tickers(category="linear")["result"]["list"]
        usdt = [t for t in r if t.get("symbol","").endswith("USDT")]
        # FIX: drop obviously bogus tickers (no price)
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
    # Safe torch.load if supported
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
    except Exception as e:
        logging.warning(f"Q-table save error: {e}")

# -------- Indicators & Data --------
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_data(symbol: str) -> pd.DataFrame:
    r = session.get_kline(category="linear", symbol=symbol, interval="15", limit=200)
    rows = r["result"]["list"]
    cols = ["start", "open", "high", "low", "close", "volume", "turnover"]
    df = pd.DataFrame(rows, columns=cols)
    for c in ["open","high","low","close","volume","turnover"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # FIX: strictly drop non-positive prices (protect indicators)
    for c in ["open","high","low","close"]:
        df = df[df[c] > 0]

    df["time"] = pd.to_datetime(pd.to_numeric(df["start"], errors="coerce"), unit="ms")
    df = df.dropna(subset=["close","time"]).sort_values("time").reset_index(drop=True)

    df["ma50"] = df["close"].rolling(50, min_periods=50).mean()
    df["rsi"] = compute_rsi(df["close"])

    # FIX: robust 15m vol; cap & fallback
    vol = df["close"].pct_change().rolling(20, min_periods=20).std()
    # sanitize series
    vol = vol.replace([np.inf, -np.inf], np.nan).fillna(VOL_FALLBACK).clip(lower=0, upper=VOL_CAP)
    df["volatility"] = vol

    df = df.dropna().reset_index(drop=True)
    return df

def _sanitize_feature_row(row: np.ndarray) -> np.ndarray:
    # row shape (8,) = [open,high,low,close,volume,ma50,rsi,vol]
    r = row.copy()
    # prices must be positive
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
    assert scaler is not None and gb_model is not None
    feats = df[["open","high","low","close","volume","ma50","rsi","volatility"]].values.astype(np.float32)

    # FIX: sanitize the last row and sequence before scaling
    last = _sanitize_feature_row(feats[-1])
    if last is None:
        # no edge: return current price => delta≈0
        return float(df["close"].iloc[-1])

    feats[-1] = last
    # If any row is broken, just replace with previous valid (simple ffill on array tail)
    bad_rows = ~np.isfinite(feats).all(axis=1)
    if bad_rows.any():
        # very lightweight repair: use last good row to fill
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

    x_seq = torch.tensor(X_scaled.reshape(1, -1, 8), dtype=torch.float32)  # full sequence
    with torch.no_grad():
        lstm_pred = float(model(x_seq).item())
    try:
        gb_pred = float(gb_model.predict(feats[-1:].astype(np.float64))[0])
    except Exception as e:
        logging.warning(f"GB.predict failed ({e}); using LSTM only")
        gb_pred = lstm_pred

    pred = (lstm_pred + gb_pred) / 2.0
    cur = float(df["close"].iloc[-1])
    # clamp ±10% to avoid runaway
    return max(cur * 0.9, min(cur * 1.1, pred))

# -------- Risk, Leverage, Adaptation --------
def get_state(vol: float) -> str:
    return "low" if vol < 0.01 else ("med" if vol < 0.03 else "high")

def adaptive_thresholds(vol: float, base: float) -> Tuple[float, float, float]:
    vol_factor = 1 + vol * 10
    tp_pct = base * vol_factor
    sl_pct = base * vol_factor * 0.67
    gate_std = max(MIN_GATE_STD, base / vol_factor)
    return tp_pct, sl_pct, gate_std

def choose_leverage(symbol: str, state: str, risk_score: float, trade_count: int) -> int:
    if symbol not in q_tables:
        q_tables[symbol] = {"low":[0.0]*LEVERAGE_MAX, "med":[0.0]*LEVERAGE_MAX, "high":[0.0]*LEVERAGE_MAX}
    q = q_tables[symbol][state]
    epsilon = max(0.05, 0.15 * (1 - trade_count / 100))
    lev = np.random.randint(1, LEVERAGE_MAX + 1) if (np.random.rand() < epsilon) else (np.argmax(q) + 1)
    base_lev = int(max(1, min(LEVERAGE_MAX, 50.0 / (risk_score + 1.0))))
    return min(lev, base_lev)

def assess_risk(df: pd.DataFrame, symbol: str, trade_count: int) -> Tuple[int, float, str, float, float, float]:
    vol = float(df["volatility"].iloc[-1])

    # FIX: guard vol and risk_score; never let bogus 999 slip through
    if not np.isfinite(vol) or vol <= 0:
        logging.warning(f"{symbol}: invalid vol={vol}; using fallback {VOL_FALLBACK}")
        vol = VOL_FALLBACK
    if vol > VOL_CAP:
        logging.warning(f"{symbol}: capping vol {vol} -> {VOL_CAP}")
        vol = VOL_CAP

    state = get_state(vol)
    risk_score = max(0.0, min(vol * 100.0, 9.99))  # cap to single-digit; avoids blocking on junk

    lev = choose_leverage(symbol, state, risk_score, trade_count)
    tp_pct, sl_pct, gate_std = adaptive_thresholds(vol, BASE_TP_PCT)
    return lev, risk_score, state, tp_pct, sl_pct, gate_std

def update_q(symbol: str, reward: float, state: str, action_lev: int):
    if symbol not in q_tables:
        q_tables[symbol] = {"low":[0.0]*LEVERAGE_MAX, "med":[0.0]*LEVERAGE_MAX, "high":[0.0]*LEVERAGE_MAX}
    q = q_tables[symbol][state]
    idx = action_lev - 1
    if 0 <= idx < len(q):
        q[idx] += 0.1 * (reward - q[idx])
        save_q_tables()

# -------- Utils --------
def send_telegram(msg: str):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            requests.get(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                         params={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=15)
        except Exception as e:
            logging.warning(f"Telegram error: {e}")

def quantize(value: float, step: float, minimum: float) -> float:
    if step <= 0: return max(value, minimum)
    k = math.floor(value / step)
    return max(minimum, k * step)

def get_portfolio_summary() -> str:
    try:
        wb = session.get_wallet_balance(accountType="UNIFIED")['result']['list'][0]
        equity = float(wb.get('totalEquity', 0) or 0)
        return f"Portfolio: {equity:.2f} USDT | Symbols: {len(SYMBOLS)} active"
    except:
        return "Portfolio: N/A"

# /status helper
_status_cache = {"t": 0, "data": {}}
def get_status() -> dict:
    global _status_cache
    now = time.time()
    if now - _status_cache["t"] < 60:
        return _status_cache["data"]
    status = {"equity": None, "positions": [], "open_orders": {}, "signals": {}}
    try:
        wb = session.get_wallet_balance(accountType="UNIFIED")["result"]["list"][0]
        status["equity"] = float(wb.get("totalEquity", 0) or 0)
    except Exception as e:
        status["equity"] = f"error: {e}"
    for sym in SYMBOLS:
        # positions
        try:
            pos_list = session.get_position(category="linear", symbol=sym)["result"]["list"]
            if pos_list:
                p = pos_list[0]
                size = float(p.get("size", 0) or 0)
                if abs(size) > 0:
                    status["positions"].append({
                        "symbol": sym,
                        "side": p.get("side"),
                        "size": size,
                        "avgPrice": float(p.get("avgPrice", 0) or 0),
                        "unrealisedPnl": float(p.get("unrealisedPnl", 0) or 0)
                    })
        except:
            pass
        # open orders
        try:
            oo = session.get_open_orders(category="linear", symbol=sym)["result"]["list"]
            status["open_orders"][sym] = len(oo)
        except:
            status["open_orders"][sym] = 0
        # live signal snapshot (best-effort)
        try:
            df = fetch_data(sym)
            cur = float(df['close'].iloc[-1])
            pred = float(predict_price(df))
            lev, risk_score, state, tp, sl, gate = assess_risk(df, sym, total_trade_counts.get(sym, 0))
            status["signals"][sym] = {
                "delta_pct": round((pred/cur - 1)*100, 3),
                "risk_score": round(risk_score, 3),
                "state": state,
                "tp_pct": round(tp*100, 3),
                "sl_pct": round(sl*100, 3),
                "gate_std_pct": round(gate*100, 3),
                "lev": lev
            }
        except Exception as e:
            status["signals"][sym] = {"error": str(e)}
    _status_cache = {"t": now, "data": status}
    return status

# -------- Health server --------
def start_health_server():
    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", "/health"):
                self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
            elif self.path == "/status":
                try:
                    st = get_status()
                    self.send_response(200); self.send_header("Content-Type","application/json"); self.end_headers()
                    self.wfile.write(json.dumps(st, indent=2).encode())
                except Exception as e:
                    self.send_response(500); self.end_headers(); self.wfile.write(f"Status error: {e}".encode())
            else:
                self.send_response(404); self.end_headers()
        def log_message(self, *a, **k): pass
    try:
        port = int(os.getenv("PORT", "10000"))
        HTTPServer(("0.0.0.0", port), HealthHandler).serve_forever()
    except Exception as e:
        logging.warning(f"health server failed: {e}")

# -------- Trading actions --------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def execute_trade(symbol: str, action: str, qty: float, leverage: int,
                  entry_price: float, entry_time: pd.Timestamp,
                  delta_pct: float, risk: float):
    try:
        session.set_leverage(category="linear", symbol=symbol,
                             buy_leverage=str(leverage), sell_leverage=str(leverage))
    except Exception as e:
        if "110043" not in str(e):
            logging.error(f"Leverage set error for {symbol}: {e}")
    side = "Buy" if action == "long" else "Sell"
    try:
        order = session.place_order(
            category="linear", symbol=symbol, side=side,
            order_type="Market", qty=str(qty),
            positionIdx=0, reduce_only=False
        )
    except TypeError:
        # fallback for clients that expect camelCase
        order = session.place_order(
            category="linear", symbol=symbol, side=side,
            order_type="Market", qty=str(qty),
            positionIdx=0, reduceOnly=False
        )
    if order.get('retCode', 1) != 0:
        logging.error(f"Order failed for {symbol}: {order.get('retMsg')}")
        return None, None, None
    try:
        avg_entry = float(order['result']['avgPrice'])
    except Exception:
        avg_entry = entry_price
    logging.info(f"ENTRY {action} {symbol} qty={qty}, lev={leverage}: avg_price={avg_entry}")
    send_telegram(f"ENTRY {action.upper()} {symbol} Δ={delta_pct:.2f}% risk={risk:.2f} qty={qty} lev={leverage} @ {avg_entry:.2f} | {get_portfolio_summary()}")
    return side, order, avg_entry

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def close_position(symbol: str, open_side: str, qty: float,
                   entry_price: float, entry_time: pd.Timestamp,
                   reason: str, exit_price: float):
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
        avg_exit = float(order['result']['avgPrice'])
    except Exception:
        avg_exit = exit_price
    # Testnet: fee-less approx
    realized_pnl = qty * (avg_exit - entry_price) if open_side == "Buy" else qty * (entry_price - avg_exit)
    logging.info(f"CLOSE {open_side} {symbol} qty={qty} ({reason}): P&L {realized_pnl:.2f}")
    send_telegram(f"CLOSE {open_side} {symbol} reason={reason} PnL={realized_pnl:.2f} | {get_portfolio_summary()}")
    notional = qty * entry_price
    reward = realized_pnl / notional if notional > 0 else 0.0
    return True, reward

# -------- Globals (declare once, then mutate in place) --------
last_trade_times: Dict[str, float] = {sym: 0.0 for sym in SYMBOLS}
trade_counts_today: Dict[str, int] = {sym: 0 for sym in SYMBOLS}
total_trade_counts: Dict[str, int] = {sym: 0 for sym in SYMBOLS}
last_day_bucket = 0
last_week_bucket = 0
initial_balance = 0.0
cumulative_pnl = 0.0
open_positions: Dict[str, Tuple] = {}
no_signal_minutes = 0

# -------- Boot diag --------
def boot_diag():
    logging.info("=== BOOT DIAG START (Multi-Symbol) ===")
    logging.info(f"TESTNET={TESTNET} SYMBOLS={SYMBOLS} BOT_PAUSED={BOT_PAUSED}")
    try:
        wb = session.get_wallet_balance(accountType="UNIFIED")['result']['list'][0]
        global initial_balance
        initial_balance = float(wb.get('totalEquity', 0) or 0)
        logging.info(f"wallet_ok equity={initial_balance:.6f}")
        if initial_balance < 2232:
            logging.warning("Low equity: Need ~2232 USDT for BTCUSDT trades")
    except Exception as e:
        logging.warning(f"wallet_error: {e}")
    for sym in SYMBOLS[:2]:
        try:
            k = session.get_kline(category='linear', symbol=sym, interval='15', limit=2)['result']['list']
            logging.info(f"kline_ok {sym} last_close={float(k[-1][4]):.2f}")
        except Exception as e:
            logging.warning(f"kline_error {sym}: {e}")
    logging.info("=== BOOT DIAG END ===")

# -------- Main loop --------
def main():
    global last_day_bucket, last_week_bucket, cumulative_pnl, no_signal_minutes
    load_q_tables()
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

            # Weekly reset (Monday UTC bucket)
            week_bucket = int(time.time() // (7 * 86400))
            if RESET_PNL_WEEKLY and week_bucket != last_week_bucket:
                cumulative_pnl = 0.0
                last_week_bucket = week_bucket
                send_telegram(f"Weekly P&L Reset | {get_portfolio_summary()}")

            # Daily refresh
            day_bucket = int(time.time() // 86400)
            if day_bucket != last_day_bucket:
                if not ENV_SYMBOLS_FIXED:
                    new_syms = get_top_symbols()
                    # mutate symbol-dependent dicts in place
                    for d, default in [(last_trade_times, 0.0),
                                       (trade_counts_today, 0),
                                       (total_trade_counts, 0)]:
                        # remove gone
                        for k in list(d.keys()):
                            if k not in new_syms:
                                d.pop(k)
                        # add new
                        for k in new_syms:
                            d.setdefault(k, default)
                    # replace SYMBOLS last (read-only list)
                    SYMBOLS[:] = new_syms[:]
                # reset per-day counters in place
                for s in SYMBOLS:
                    trade_counts_today[s] = 0
                last_day_bucket = day_bucket
                send_telegram(f"Daily Reset | Symbols: {SYMBOLS} | Cum P&L: {cumulative_pnl:.2f} | {get_portfolio_summary()}")

            # Per-symbol pass (sequential, rate-safe)
            for symbol in SYMBOLS:
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
                            cumulative_pnl += reward * (qty * entry_price)
                            open_positions.pop(symbol)
                            last_trade_times[symbol] = time.time()
                        continue

                # Entries
                delta = (predicted / current_price) - 1.0
                delta_pct = delta * 100.0
                action = None
                # High-confidence override
                if abs(delta) >= 0.01 and risk_score < MIN_RISK:
                    action = "long" if delta > 0 else "short"
                # Standard gate with OR alignment
                elif abs(delta) >= gate_std and risk_score < MIN_RISK:
                    long_ok = (rsi > 50.0 or current_price > ma50) and delta > 0
                    short_ok = (rsi < 50.0 or current_price < ma50) and delta < 0
                    if long_ok: action = "long"
                    elif short_ok: action = "short"

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
                        balance = float(wb.get("totalEquity", 0) or 0)
                    except Exception:
                        balance = initial_balance

                    vol_now = max(0.0, min(vol, VOL_CAP))
                    vol_weight = max(0.1, 1 / (1 + vol_now * 100))
                    growth_factor = 1 + (cumulative_pnl / max(1, initial_balance))
                    position_pct = min(BASE_POSITION_PCT * growth_factor * vol_weight, 0.10)

                    effective_notional = balance * position_pct * max(1, leverage)
                    raw_qty = effective_notional / max(1e-9, current_price)
                    qty = quantize(raw_qty, qty_step, min_qty)

                    if qty >= min_qty:
                        entry_time = pd.Timestamp.now(tz='UTC')
                        res = execute_trade(symbol, action, qty, leverage, current_price, entry_time, delta_pct, risk_score)
                        if not res or res[0] is None:
                            continue
                        open_side, order, avg_entry = res
                        open_positions[symbol] = (open_side, qty, avg_entry, entry_time, leverage, state)
                        last_trade_times[symbol] = time.time()
                        trade_counts_today[symbol] = trade_counts_today.get(symbol, 0) + 1
                        total_trade_counts[symbol] = total_trade_counts.get(symbol, 0) + 1
                    else:
                        logging.info(f"Skipped {symbol}: qty {qty} < min {min_qty}")
                else:
                    # FIX: clearer log shows gate in % and current vol%
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

        # pacing: ~12s/loop for 10 symbols, rate-safe
        time.sleep(max(1.0, 120 / max(1, len(SYMBOLS))))

# -------- Entrypoint --------
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
