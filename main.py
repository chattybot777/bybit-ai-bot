import os, time, json, logging, threading, pickle, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tenacity import retry, stop_after_attempt, wait_exponential
from pybit.unified_trading import HTTP

# ---------- Perf: keep CPU/memory small ----------
torch.set_num_threads(int(os.getenv("PYTORCH_NUM_THREADS", "1")))

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ---------- Env ----------
API_KEY     = os.getenv("BYBIT_API_KEY")
API_SECRET  = os.getenv("BYBIT_API_SECRET")
SYMBOL      = os.getenv("SYMBOL", "BTCUSDT")
TESTNET     = os.getenv("TESTNET", "true").lower() == "true"
RECV_WINDOW = int(os.getenv("RECV_WINDOW", "60000"))
BOT_PAUSED  = os.getenv("BOT_PAUSED", "false").lower() == "true"  # true => pause entries

POSITION_SIZE_PCT = float(os.getenv("POSITION_SIZE_PCT", "0.05"))  # 5%
DRAWDOWN_STOP     = float(os.getenv("DRAWDOWN_STOP", "0.20"))      # 20%
DELTA_STD         = 0.005  # 0.5%
DELTA_OVERRIDE    = 0.010  # 1.0%
VOL_LIMIT         = 5.0    # risk gate: volatility*100 < 5

# Telegram (optional)
TG_TOKEN   = os.getenv("TELEGRAM_TOKEN")
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ---------- Bybit Session ----------
session = HTTP(
    testnet=TESTNET,
    api_key=API_KEY,
    api_secret=API_SECRET,
    recv_window=RECV_WINDOW
)

# ---------- Model ----------
class LSTMModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=50, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)
    def forward(self, x):
        y, _ = self.lstm(x)
        return self.fc(y[:, -1, :])

models_ready = True
model = LSTMModel()
try:
    # safe load
    state = torch.load("lstm.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    logging.info("Loaded lstm.pth")
except Exception as e:
    models_ready = False
    logging.error(f"Load LSTM failed: {e}")

try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    logging.info("Loaded scaler.pkl")
except Exception as e:
    models_ready = False
    scaler = None
    logging.error(f"Load scaler failed: {e}")

try:
    with open("gb.pkl", "rb") as f:
        gb_model = pickle.load(f)
    logging.info("Loaded gb.pkl")
except Exception as e:
    models_ready = False
    gb_model = None
    logging.error(f"Load GB failed: {e}")

# ---------- Helpers ----------
def rsi(series, period=14):
    d = series.diff()
    gain = d.clip(lower=0).rolling(period).mean()
    loss = (-d.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100/(1+rs))

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=8))
def fetch_df():
    r = session.get_kline(category="linear", symbol=SYMBOL, interval="60", limit=200)
    kl = r["result"]["list"]
    df = pd.DataFrame(kl, columns=["ts","open","high","low","close","volume","turnover"])
    for c in ["open","high","low","close","volume","turnover"]:
        df[c] = df[c].astype("float32")
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.sort_values("ts").set_index("ts")
    df["ma50"] = df["close"].rolling(50).mean()
    df["rsi"]  = rsi(df["close"])
    df["volatility"] = df["close"].pct_change().rolling(20).std()
    return df.dropna()

def clamp_pred(pred, cur):
    lo = cur * 0.90
    hi = cur * 1.10
    return max(lo, min(hi, pred))

def choose_leverage(risk_score):
    # Grok rule, cap 20x on testnet
    lev = max(1, min(50, int(50 / (risk_score + 1))))
    return min(20, lev) if TESTNET else lev

def notify(msg):
    if not TG_TOKEN or not TG_CHAT_ID:
        return
    import requests
    try:
        requests.get(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            params={"chat_id": TG_CHAT_ID, "text": msg},
            timeout=10
        )
    except Exception as e:
        logging.warning(f"Telegram send failed: {e}")

# ---------- Trading Ops ----------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def set_leverage(lev):
    try:
        session.set_leverage(category="linear", symbol=SYMBOL,
                             buy_leverage=str(lev), sell_leverage=str(lev))
        logging.info(f"set_leverage OK → {lev}x")
    except Exception as e:
        if "110043" in str(e):  # not modified
            logging.info(f"set_leverage unchanged ({lev}x)")
        else:
            raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def place_entry(side, qty):
    return session.place_order(category="linear", symbol=SYMBOL, side=side,
                               order_type="Market", qty=str(qty),
                               position_idx=0, reduce_only=False)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def place_exit(current_side, qty):
    opp = "Sell" if current_side == "Buy" else "Buy"
    return session.place_order(category="linear", symbol=SYMBOL, side=opp,
                               order_type="Market", qty=str(qty),
                               position_idx=0, reduce_only=True)

def min_step_info():
    info = session.get_instruments_info(category="linear", symbol=SYMBOL)["result"]["list"][0]
    lot = info["lotSizeFilter"]
    return float(lot["minOrderQty"]), float(lot["qtyStep"])

def round_step(qty, step):
    return math.floor(qty / step) * step

# ---------- Main Loop ----------
def main():
    initial_equity = float(session.get_wallet_balance(accountType="UNIFIED")["result"]["list"][0]["totalEquity"])
    last_trade_ts = 0.0
    daily_count = 0
    last_day = int(time.time() // 86400)
    open_pos = None  # dict: {side, qty, entry, t0}

    logging.info(f"Boot | symbol={SYMBOL} | testnet={TESTNET} | models_ready={models_ready}")

    while True:
        try:
            # Reset day counter
            day = int(time.time() // 86400)
            if day != last_day:
                daily_count = 0
                last_day = day

            df = fetch_df()
            cur  = float(df["close"].iloc[-1])
            ma50 = float(df["ma50"].iloc[-1])
            r    = float(df["rsi"].iloc[-1])
            vol  = float(df["volatility"].iloc[-1])
            risk = vol * 100.0

            if not models_ready:
                logging.info("Models not loaded. Skipping entries.")
                time.sleep(60); continue

            # LSTM + GB ensemble
            feats = df[["open","high","low","close","volume","ma50","rsi","volatility"]].values
            # LSTM expects sequence; use recent 50 steps if available
            seq = feats[-50:] if len(feats) >= 50 else feats
            Xl = scaler.transform(seq)
            Xl = torch.tensor(Xl.reshape(1, len(seq), 8), dtype=torch.float32)
            with torch.no_grad():
                lstm_pred = float(model(Xl).item())

            Xg = scaler.transform([feats[-1]])
            gb_pred = float(gb_model.predict(Xg)[0])

            pred = clamp_pred((lstm_pred + gb_pred)/2.0, cur)
            delta = (pred / cur) - 1.0
            delta_pct = delta * 100.0

            # Cooldown & daily limit
            if time.time() - last_trade_ts < 900 or daily_count >= 10:
                time.sleep(60); continue

            # Exit logic if we have a tracked open position
            if open_pos:
                side = open_pos["side"]   # "Buy" or "Sell"
                qty  = open_pos["qty"]
                ent  = open_pos["entry"]
                t0   = open_pos["t0"]

                tp = ent * (1.015 if side == "Buy" else 0.985)
                sl = ent * (0.99  if side == "Buy" else 1.01)
                hours_open = (time.time() - t0) / 3600.0

                hit_tp = (cur >= tp) if side == "Buy" else (cur <= tp)
                hit_sl = (cur <= sl) if side == "Buy" else (cur >= sl)
                hit_time = hours_open >= 4.0

                if hit_tp or hit_sl or hit_time:
                    reason = "TP" if hit_tp else ("SL" if hit_sl else "Time")
                    res = place_exit(side, qty)
                    logging.info(f"EXIT {reason} → {res}")
                    notify(f"EXIT {reason} {SYMBOL} qty={qty} price≈{cur:.2f}")
                    open_pos = None
                    last_trade_ts = time.time()
                    daily_count += 1
                    time.sleep(60); continue

            if BOT_PAUSED:
                time.sleep(60); continue

            # Entry logic (Grok)
            action = None
            if abs(delta) >= DELTA_OVERRIDE and risk < VOL_LIMIT:
                action = "Buy" if delta > 0 else "Sell"
            elif abs(delta) >= DELTA_STD and risk < VOL_LIMIT:
                long_ok  = (delta > 0) and (r < 50) and (cur > ma50)
                short_ok = (delta < 0) and (r > 50) and (cur < ma50)
                if long_ok:  action = "Buy"
                if short_ok: action = "Sell"

            if not action:
                logging.info(f"No entry: delta={delta_pct:.2f}%, risk={risk:.2f}, rsi={r:.1f}, cur={cur:.2f}, ma50={ma50:.2f}")
                time.sleep(60); continue

            lev = choose_leverage(risk)
            try:
                set_leverage(lev)
            except Exception as e:
                logging.error(f"set_leverage error: {e}")

            # Qty sizing (5% equity)
            equity = float(session.get_wallet_balance(accountType="UNIFIED")["result"]["list"][0]["totalEquity"])
            min_qty, step = min_step_info()
            raw_qty = (equity * POSITION_SIZE_PCT) / cur
            qty = max(min_qty, round_step(raw_qty, step))

            if qty < min_qty:
                logging.info(f"Skip: qty {qty} < min {min_qty}")
                time.sleep(60); continue

            res = place_entry(action, qty)
            logging.info(f"ENTRY {action} lev={lev} qty={qty} → {res}")
            notify(f"ENTRY {action} {SYMBOL} Δ={delta_pct:.2f}% risk={risk:.2f} lev={lev} qty={qty} @≈{cur:.2f}")

            open_pos = {"side": action, "qty": qty, "entry": cur, "t0": time.time()}
            last_trade_ts = time.time()
            daily_count  += 1

            # Drawdown guard
            drop = (initial_equity - equity) / max(initial_equity, 1e-9)
            if drop > DRAWDOWN_STOP:
                logging.warning("Drawdown limit reached. Halting entries.")
                notify("Drawdown limit reached — halting.")
                break

        except Exception as e:
            logging.error(f"Main loop error: {e}")

        time.sleep(60)

if __name__ == "__main__":
    main()
