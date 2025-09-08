# main.py — Grok Bybit Bot (Torch 2.2 compatible, NaN-safe, override entries, exits, adaptive lev)
import os, time, math, pickle, logging, requests, threading
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from pybit.unified_trading import HTTP, WebSocket

# ---------- Logging ----------
import sys
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')s - %(levelname)s - %(message)s')s - %(levelname)s - %(message)s"
)

load_dotenv()
torch.set_num_threads(1)  # keep CPU small

# ---------- Env / Config ----------
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET") or os.getenv("API_SECRET")
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
TESTNET = os.getenv("TESTNET", "true").lower() == "true"
RECV_WINDOW = int(os.getenv("RECV_WINDOW", 60000))

# BOT_PAUSED=true -> pause; false -> run
BOT_PAUSED = os.getenv("BOT_PAUSED", "false").lower() == "true"

LEVERAGE_MAX = 20          # testnet cap
POSITION_SIZE_PCT = 0.05   # 5% equity
DRAWDOWN_STOP = 0.20       # 20% equity drawdown
COOLDOWN_SEC = 15 * 60
MAX_TRADES_PER_DAY = 10
TIME_STOP_HOURS = 4
TP_PCT = 0.015
SL_PCT = 0.01

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

session = HTTP(testnet=TESTNET, api_key=API_KEY, api_secret=API_SECRET, recv_window=RECV_WINDOW)
try:
    try:
    ws=None
try:
    try:
    ws = WebSocket(testnet=TESTNET, channel_type='linear')
except Exception as e:
    ws = None
    logging.warning(f"WebSocket init failed: {e}")
except Exception as e:
    logging.warning(f'WebSocket init failed: {e}')
except Exception as e:
    ws = None
    logging.warning(f"WebSocket init failed: {e}")
except Exception as e:
    ws = None
    logging.warning(f"WebSocket init failed: {e}")

# ---------- Model ----------
class LSTMModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=50, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = LSTMModel()
try:
    # Torch 2.2: no weights_only kw
    state = torch.load("lstm.pth", map_location="cpu")
    model.load_state_dict(state)
    scaler: MinMaxScaler = pickle.load(open("scaler.pkl", "rb"))
    gb_model: GradientBoostingRegressor = pickle.load(open("gb.pkl", "rb"))
except Exception as e:
    logging.error(f"Model load error: {e}")
    raise

# ---------- RL (leverage reward feedback) ----------
q_table = {}
last_trade_time = 0
trade_count_today = 0
last_day = 0

def get_state(volatility):
    if volatility < 0.01: return "low"
    if volatility < 0.03: return "med"
    return "high"

def choose_leverage(state, risk_score, epsilon=0.1):
    if np.random.rand() < epsilon:
        return np.random.randint(1, LEVERAGE_MAX + 1)
    if state not in q_table:
        q_table[state] = [0] * LEVERAGE_MAX
    return min(LEVERAGE_MAX, max(1, int(50 / (risk_score + 1))))

def update_q(reward, state, action):
    if state not in q_table:
        q_table[state] = [0] * LEVERAGE_MAX
    idx = max(1, min(action, LEVERAGE_MAX)) - 1
    q_table[state][idx] += 0.1 * (reward - q_table[state][idx])

# ---------- Utils ----------
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

def quantize(value, step, minimum):
    k = math.floor(value / step) if step > 0 else value
    q = k * step if step > 0 else value
    if q < minimum: q = minimum
    return float(f"{q:.10f}")

def compute_rsi(series: pd.Series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_data():
    kl = session.get_kline(category="linear", symbol=SYMBOL, interval="60", limit=200)["result"]["list"]
    df = pd.DataFrame(kl, columns=["time","open","high","low","close","volume","turnover"])
    # ensure numeric then to datetime (avoids FutureWarning) + float32 cast
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.astype({c: "float32" for c in ["open","high","low","close","volume","turnover"]})
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df.set_index("time", inplace=True)
    df.sort_index(inplace=True)
    df["ma50"] = df["close"].rolling(50).mean()
    df["rsi"] = compute_rsi(df["close"])
    df["volatility"] = df["close"].pct_change().rolling(20).std()
    return df.dropna()

def predict_price(df: pd.DataFrame) -> float:
    df = df.dropna().copy()
    if len(df) == 0:
        return float("nan")
    features = df[["open","high","low","close","volume","ma50","rsi","volatility"]].values.astype("float32")
    scaled = scaler.transform(features)
    X = scaled.reshape(1, -1, 8)
    with torch.no_grad():
        lstm_pred = float(model(torch.tensor(X)).item())
    gb_pred = float(gb_model.predict(features[-1].reshape(1, -1))[0])
    pred = (lstm_pred + gb_pred) / 2.0
    cur = float(df["close"].iloc[-1])
    return max(min(pred, cur * 1.1), cur * 0.9)  # clamp ±10%

def assess_risk(df: pd.DataFrame):
    vol = float(df["volatility"].iloc[-1])
    state = get_state(vol)
    risk_score = vol * 100.0
    lev = choose_leverage(state, risk_score)
    return lev, risk_score, state

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def execute_trade(action: str, qty: float, leverage: int, entry_price: float, entry_time, delta_pct: float, risk: float):
    try:
        session.set_leverage(category="linear", symbol=SYMBOL, buy_leverage=leverage, sell_leverage=leverage)
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
    logging.info(f"ENTRY {action} qty={qty}, lev={leverage}: {order}")
    send_telegram(f"ENTRY {action.upper()} {SYMBOL} Δ={delta_pct:.2f}% risk={risk:.2f} qty={qty} lev={leverage} @ {entry_price:.2f}")
    return side, order

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def close_position(open_side: str, qty: float, entry_price: float, entry_time, reason: str):
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
    pos_list = session.get_position(category="linear", symbol=SYMBOL)["result"]["list"]
    realized_pnl = 0.0
    try:
        if pos_list:
            realized_pnl = float(pos_list[0].get("realisedPnl", 0) or 0)
    except Exception:
        pass
    logging.info(f"CLOSE {open_side} qty={qty} ({reason}): {order}, P&L: {realized_pnl}")
    send_telegram(f"CLOSE {open_side} {SYMBOL} qty={qty} reason={reason} entry={entry_price:.2f} PnL={realized_pnl:.2f}")
    notional = max(1e-9, qty * entry_price)
    reward = realized_pnl / notional
    return True, reward

def main():
    global last_trade_time, trade_count_today, last_day
    equity_raw = session.get_wallet_balance(accountType="UNIFIED")["result"]["list"][0]["totalEquity"]
    balance = float(equity_raw)
    initial_balance = balance
    open_position = None  # (open_side, qty, entry_price, entry_time, leverage, state)
    no_signal_minutes = 0

    while True:
        try:
            if BOT_PAUSED:
                time.sleep(60); continue

            current_day = int(time.time() // 86400)
            if current_day != last_day:
                trade_count_today = 0
                last_day = current_day

            df = fetch_data()
            current_price = float(df["close"].iloc[-1])
            predicted = float(predict_price(df))
            leverage, risk, state = assess_risk(df)
            rsi = float(df["rsi"].iloc[-1])
            ma50 = float(df["ma50"].iloc[-1])

            if (time.time() - last_trade_time) < COOLDOWN_SEC or trade_count_today >= MAX_TRADES_PER_DAY:
                time.sleep(60); continue

            # Exits first
            if open_position:
                open_side, qty, entry_price, entry_time, lev_used, state_at_entry = open_position
                tp = entry_price * (1.0 + TP_PCT if open_side == "Buy" else 1.0 - TP_PCT)
                sl = entry_price * (1.0 - SL_PCT if open_side == "Buy" else 1.0 + SL_PCT)
                hours_open = (pd.Timestamp.utcnow() - entry_time).total_seconds() / 3600.0
                reason = None
                if (open_side == "Buy" and current_price >= tp) or (open_side == "Sell" and current_price <= tp):
                    reason = "TP"
                elif (open_side == "Buy" and current_price <= sl) or (open_side == "Sell" and current_price >= sl):
                    reason = "SL"
                elif hours_open >= TIME_STOP_HOURS:
                    reason = "Time Stop"
                if reason:
                    ok, reward = close_position(open_side, qty, entry_price, entry_time, reason)
                    if ok:
                        open_position = None
                        last_trade_time = time.time()
                        trade_count_today += 1
                        update_q(reward, state_at_entry, lev_used)
                    time.sleep(60); continue

            # Entries
            delta_pct = (predicted / current_price - 1.0) * 100.0
            action = None
            # Override: ignore RSI/MA
            if abs(delta_pct) >= 1.0 and risk < 5:
                action = "long" if delta_pct > 0 else "short"
            # Standard: require alignment
            elif abs(delta_pct) >= 0.5 and risk < 5:
                long_ok = (rsi > 50.0) and (current_price > ma50)
                short_ok = (rsi < 50.0) and (current_price < ma50)
                if long_ok:  action = "long"
                if short_ok: action = "short"

            if action:
                info = session.get_instruments_info(category="linear", symbol=SYMBOL)["result"]["list"][0]
                min_qty = float(info["lotSizeFilter"]["minOrderQty"])
                qty_step = float(info["lotSizeFilter"]["qtyStep"])

                balance = float(session.get_wallet_balance(accountType="UNIFIED")["result"]["list"][0]["totalEquity"])
                raw_qty = (balance * POSITION_SIZE_PCT) / max(1e-9, current_price)
                qty = quantize(raw_qty, qty_step, min_qty)

                if qty >= min_qty:
                    entry_time = pd.Timestamp.utcnow()
                    open_side, _ = execute_trade(
                        action=action,
                        qty=qty,
                        leverage=leverage,
                        entry_price=current_price,
                        entry_time=entry_time,
                        delta_pct=delta_pct,
                        risk=risk
                    )
                    open_position = (open_side, qty, current_price, entry_time, leverage, state)
                    last_trade_time = time.time()
                else:
                    logging.info(f"Skipped: qty {qty} < min {min_qty}")
            else:
                logging.info(f"No entry: Δ={delta_pct:.2f}% risk={risk:.2f} rsi={rsi:.2f} px={current_price:.2f} ma50={ma50:.2f}")
                no_signal_minutes += 1
                if no_signal_minutes >= 48 * 60:
                    send_telegram(f"Alive: 48h no trades. Last Δ={delta_pct:.2f}% risk={risk:.2f}")
                    no_signal_minutes = 0

            balance = float(session.get_wallet_balance(accountType="UNIFIED")["result"]["list"][0]["totalEquity"])
            if (initial_balance - balance) / max(1e-9, initial_balance) > DRAWDOWN_STOP:
                logging.warning("Drawdown limit reached. Stopping.")
                send_telegram("Bot stop: drawdown limit reached.")
                break

        except Exception as e:
            logging.error(f"Main loop error: {e}")

        time.sleep(60)

if __name__ == "__main__":
    threading.Thread(target=start_health_server, daemon=True).start()
    boot_diag()

    try:
        threading.Thread(target=start_health_server, daemon=True).start()
    except Exception as e:
        logging.warning(f"health thread err: {e}")
    try:
        boot_diag()
    except Exception:
        pass
    main()


from http.server import BaseHTTPRequestHandler, HTTPServer
def start_health_server():
    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/health", "/"):
                self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
            else:
                self.send_response(404); self.end_headers()
        def log_message(self, *args, **kwargs): pass
    try:
        port = int(os.getenv("PORT", "10000"))
        HTTPServer(("0.0.0.0", port), HealthHandler).serve_forever()
    except Exception as e:
        logging.warning(f"health server failed: {e}")


def boot_diag():
    try:
        logging.info("=== BOOT DIAG START ===")
        logging.info(f"TESTNET={os.getenv('TESTNET')} SYMBOL={os.getenv('SYMBOL')} BOT_PAUSED={os.getenv('BOT_PAUSED')}")
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
            k = session.get_kline(category='linear', symbol=os.getenv('SYMBOL','BTCUSDT'), interval='60', limit=2)['result']['list']
            logging.info(f"kline_ok last_close={float(k[-1][4])}")
        except Exception as e:
            logging.warning(f"kline_error: {e}")
        logging.info("=== BOOT DIAG END ===")
    except Exception as e:
        logging.warning(f"boot_diag error: {e}")
