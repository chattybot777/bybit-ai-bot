import os, time, math, logging, pickle
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from pybit.unified_trading import HTTP
from tenacity import retry, stop_after_attempt, wait_exponential

# ---------- Config / Env ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
TESTNET = os.getenv("TESTNET","true").lower()=="true"
SYMBOL = os.getenv("SYMBOL","BTCUSDT")
RECV_WINDOW = int(os.getenv("RECV_WINDOW","60000"))
DRAWDOWN_STOP = float(os.getenv("DRAWDOWN_STOP","0.20"))   # 20%
POSITION_SIZE_PCT = float(os.getenv("POSITION_SIZE_PCT","0.05"))  # 5%
LEVERAGE_DEFAULT = os.getenv("LEVERAGE_DEFAULT","1")
COOLDOWN_SEC = 15*60
MAX_TRADES_PER_DAY = 10

# ---------- Session ----------
session = HTTP(testnet=TESTNET,
               api_key=os.getenv("BYBIT_API_KEY"),
               api_secret=os.getenv("BYBIT_API_SECRET"),
               recv_window=RECV_WINDOW)

# ---------- Helpers ----------
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    # TR = max(H-L, |H-PrevC|, |L-PrevC|)
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def now_utc():
    return datetime.now(timezone.utc)

def day_key(ts: datetime):
    return ts.strftime("%Y-%m-%d")

def api_call(label, fn, **kw):
    try:
        r = fn(**kw)
        rc = r.get("retCode", 0); rm = r.get("retMsg","")
        if rc == 0:
            logging.info(f"{label}: OK")
        elif rc == 110043:
            # leverage not modified => treat as no-op OK
            logging.info(f"{label}: no-op (leverage not modified)")
        else:
            logging.error(f"{label}: retCode={rc} retMsg={rm}. Request → {fn.__name__} {kw}")
        return r
    except Exception as e:
        logging.error(f"{label}: EXC {type(e).__name__}: {e}")
        return {"retCode": 99999, "retMsg": str(e)}

# ---------- Instruments (min qty / step) ----------
def load_instr_filters():
    r = api_call("get_instruments_info", session.get_instruments_info,
                 category="linear", symbol=SYMBOL)
    lot = (((r or {}).get("result") or {}).get("list") or [{}])[0].get("lotSizeFilter", {})
    min_qty = float(lot.get("minOrderQty", "0.001"))
    qty_step = float(lot.get("qtyStep", "0.001"))
    return min_qty, qty_step

MIN_QTY, QTY_STEP = load_instr_filters()

# ---------- Models ----------
class LSTMModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=50, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:,-1,:])

model = LSTMModel(input_size=8, hidden_size=50, num_layers=2)
scaler = MinMaxScaler()
gb_model = None

def try_load_models():
    ok = True
    try:
        if os.path.exists("lstm.pth"):
            state = torch.load("lstm.pth", map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            logging.info("Loaded lstm.pth")
        else:
            logging.warning("lstm.pth missing — trading will SKIP until models provided")
            ok = False
    except Exception as e:
        logging.error(f"Load LSTM failed: {e}"); ok = False
    try:
        if os.path.exists("scaler.pkl"):
            with open("scaler.pkl","rb") as f: 
                s = pickle.load(f)
                # ensure it’s a MinMaxScaler
                if isinstance(s, MinMaxScaler):
                    global scaler
                    scaler = s
                    logging.info("Loaded scaler.pkl")
        else:
            logging.warning("scaler.pkl missing — trading will SKIP until models provided")
            ok = False
    except Exception as e:
        logging.error(f"Load scaler failed: {e}"); ok = False
    try:
        if os.path.exists("gb.pkl"):
            with open("gb.pkl","rb") as f:
                global gb_model
                gb_model = pickle.load(f)
            logging.info("Loaded gb.pkl")
        else:
            logging.warning("gb.pkl missing — ensemble will be disabled")
    except Exception as e:
        logging.error(f"Load GB failed: {e}")
    return ok

MODELS_READY = try_load_models()

# ---------- Data ----------
def fetch_ohlc(limit=200):
    r = api_call("get_kline", session.get_kline,
                 category="linear", symbol=SYMBOL, interval="60", limit=limit)
    rows = ((r.get("result") or {}).get("list") or [])
    if not rows: return None
    # v5 returns list of lists; order oldest->newest not guaranteed; normalize
    df = pd.DataFrame(rows, columns=["start","open","high","low","close","volume","turnover"])
    for c in ["open","high","low","close","volume","turnover"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["start"] = pd.to_datetime(pd.to_numeric(df["start"], errors="coerce"), unit="ms", utc=True)
    df = df.dropna().sort_values("start").reset_index(drop=True)
    # features
    df["ma50"] = df["close"].rolling(50).mean()
    df["rsi"] = compute_rsi(df["close"], 14)
    df["volatility"] = df["close"].pct_change().rolling(20).std()
    df["atr14"] = compute_atr(df[["high","low","close"]].rename(columns=str), 14)
    return df.dropna()

def latest_price(df): return float(df["close"].iloc[-1])

# ---------- Prediction ----------
def predict_next_close(df):
    # features: OHLCV + ma50 + rsi + volatility  (8)
    feats = df[["open","high","low","close","volume","ma50","rsi","volatility"]].values.astype("float32")
    # scale and LSTM over last sequence window
    seq = 32
    if feats.shape[0] < seq: return None
    scaled = scaler.transform(feats) if hasattr(scaler, "min_") else feats
    X = torch.tensor(scaled[-seq:].reshape(1, seq, 8), dtype=torch.float32)
    with torch.no_grad():
        lstm_pred = model(X).item()
    # Ensemble with GB on the last feature row (unscaled raw works if GB trained that way)
    if gb_model is not None:
        gb_pred = gb_model.predict(feats[-1:].astype("float32"))[0]
        return (lstm_pred + float(gb_pred)) / 2.0
    return lstm_pred

# ---------- Position / Orders ----------
def get_equity():
    r = api_call("get_wallet_balance", session.get_wallet_balance, accountType="UNIFIED")
    try:
        lst = (r["result"]["list"][0]["totalEquity"])
        return float(lst)
    except Exception:
        return 0.0

def get_position():
    r = api_call("get_positions", session.get_positions, category="linear", symbol=SYMBOL)
    arr = (r.get("result") or {}).get("list") or []
    if not arr: return None
    p = arr[0]
    size = float(p.get("size","0") or 0)
    if size == 0: return None
    side = p.get("side","").capitalize()   # "Buy" or "Sell"
    avg_price = float(p.get("avgPrice","0") or 0)
    created = p.get("createdTime") or p.get("updatedTime")
    opened_at = None
    try:
        if created: opened_at = datetime.fromtimestamp(int(created)/1000, tz=timezone.utc)
    except Exception: 
        opened_at = None
    return {"size": size, "side": side, "avg_price": avg_price, "opened_at": opened_at}

def round_to_step(q):
    if QTY_STEP <= 0: return q
    return math.floor(q / QTY_STEP) * QTY_STEP

def compute_leverage(risk_score):
    # Grok: leverage = min(50, max(1, 50/(risk+1)))
    lev = int(max(1, min(50, round(50 / (risk_score + 1)))))
    return str(lev)

def set_leverage(lev_str):
    api_call("set_leverage", session.set_leverage,
             category="linear", symbol=SYMBOL,
             buy_leverage=lev_str, sell_leverage=lev_str)

def place_market(side, qty, reduce_only=False):
    if qty <= 0:
        return {"retCode": 10001, "retMsg": "qty<=0"}
    if qty < MIN_QTY:
        logging.warning(f"qty {qty} < MIN_QTY {MIN_QTY}; skipping")
        return {"retCode": 10001, "retMsg": "qty<min"}
    q_str = f"{qty:.10f}".rstrip("0").rstrip(".")
    return api_call("place_order", session.place_order,
                    category="linear", symbol=SYMBOL,
                    side=side, order_type="Market",
                    qty=q_str, positionIdx=0, reduce_only=reduce_only)

# ---------- Loop state ----------
initial_equity = get_equity()
last_trade_time = datetime.fromtimestamp(0, tz=timezone.utc)
trades_today = 0
day_marker = day_key(now_utc())

# ---------- Main loop ----------
def main():
    global last_trade_time, trades_today, day_marker
    logging.info(f"Booting bot | symbol={SYMBOL} | testnet={TESTNET} | models_ready={MODELS_READY}")

    while True:
        try:
            # reset daily counter
            now = now_utc()
            if day_key(now) != day_marker:
                day_marker = day_key(now); trades_today = 0

            df = fetch_ohlc(limit=200)
            if df is None or df.empty:
                time.sleep(60); continue

            px = latest_price(df)
            risk_score = float(df["volatility"].iloc[-1] * 100.0)
            ma50 = float(df["ma50"].iloc[-1])
            rsi = float(df["rsi"].iloc[-1])
            atr14 = float(df["atr14"].iloc[-1])

            pos = get_position()  # check exits if any
            if pos:
                # --- Exit rules (Grok) ---
                if pos["side"] == "Buy":
                    tp = pos["avg_price"] * 1.015
                    sl = max(pos["avg_price"] * 0.99, pos["avg_price"] - 2.0*atr14)  # ATR-based preference
                    hit_tp = px >= tp
                    hit_sl = px <= sl
                else:  # Sell (short)
                    tp = pos["avg_price"] * 0.985
                    sl = min(pos["avg_price"] * 1.01, pos["avg_price"] + 2.0*atr14)
                    hit_tp = px <= tp
                    hit_sl = px >= sl

                time_stop = False
                if pos.get("opened_at"):
                    time_stop = (now - pos["opened_at"]) >= timedelta(hours=4)

                # Optional adaptive exit: if prediction flips hard, exit early
                pred_ok = MODELS_READY
                will_exit_on_flip = False
                if pred_ok:
                    pred = predict_next_close(df)
                    if pred is not None:
                        if pos["side"] == "Buy" and (pred < px): will_exit_on_flip = True
                        if pos["side"] == "Sell" and (pred > px): will_exit_on_flip = True

                if hit_tp or hit_sl or time_stop or will_exit_on_flip:
                    close_side = "Sell" if pos["side"] == "Buy" else "Buy"
                    logging.info(f"Exit condition met | side={pos['side']} | TP={hit_tp} SL={hit_sl} TIME={time_stop} FLIP={will_exit_on_flip}")
                    # Close full position (reduce_only)
                    place_market(close_side, qty=pos["size"], reduce_only=True)
                    last_trade_time = now
                    trades_today += 1
                    time.sleep(60)
                    continue  # next loop

            # --- Entry rules (Grok) ---
            if not MODELS_READY:
                logging.info("Models not loaded (lstm.pth/scaler.pkl missing). Skipping entries.")
                time.sleep(60); continue

            # Frequency constraints
            if (now - last_trade_time).total_seconds() < COOLDOWN_SEC:
                logging.info("Cooldown active; skipping.")
                time.sleep(60); continue
            if trades_today >= MAX_TRADES_PER_DAY:
                logging.info("Max trades reached for today; skipping.")
                time.sleep(60); continue

            pred = predict_next_close(df)
            if pred is None:
                time.sleep(60); continue

            long_gate  = (pred > px*1.005) and (risk_score < 5) and (rsi < 50) and (px > ma50)
            short_gate = (pred < px*0.995) and (risk_score < 5) and (rsi > 50) and (px < ma50)

            action = None
            if long_gate:  action = "long"
            elif short_gate: action = "short"

            if action:
                # Sizing
                equity = get_equity()
                if equity <= 0:
                    logging.error("No equity; cannot size.")
                    time.sleep(60); continue
                raw_qty = (equity * POSITION_SIZE_PCT) / px
                qty = round_to_step(raw_qty)
                if qty < MIN_QTY:
                    logging.warning(f"qty {qty} < min {MIN_QTY}; skip entry.")
                    time.sleep(60); continue

                # Adaptive leverage per trade
                lev = compute_leverage(risk_score)
                set_leverage(lev)

                side = "Buy" if action=="long" else "Sell"
                r = place_market(side, qty=qty, reduce_only=False)
                if r.get("retCode")==0:
                    last_trade_time = now
                    trades_today += 1
            else:
                logging.info("No entry signal.")
        except Exception as e:
            logging.error(f"Loop error: {e}")
        time.sleep(60)

if __name__ == "__main__":
    # Single leverage set on boot (baseline), per Grok we also set per-trade adaptively
    set_leverage(LEVERAGE_DEFAULT)
    main()
