# main.py — Bybit AI Trading Bot (Grok plan, safe-boot guards)
import os, time, logging, numpy as np, pandas as pd, torch, torch.nn as nn
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.exceptions import NotFittedError
from pybit.unified_trading import HTTP, WebSocket

# ---------- config & logging ----------
load_dotenv()
import logging, math, os

def api_call(label, fn, **kw):
    try:
        r = fn(**kw)
        rc = r.get("retCode", 0); rm = r.get("retMsg","")
        if rc != 0:
            logging.error(f"{label}: retCode={rc} retMsg={rm} kw={kw}")
        else:
            logging.info(f"{label}: OK")
        return r
    except Exception as e:
        logging.error(f"{label}: EXC {type(e).__name__}: {e} kw={kw}")
        return {"retCode": 99999, "retMsg": str(e)}

MIN_QTY = float(os.getenv("MIN_QTY_DEFAULT","0"))
QTY_STEP = float(os.getenv("QTY_STEP_DEFAULT","0"))

def get_symbol_filters(session, symbol):
    r = api_call("get_instruments_info", session.get_instruments_info,
                 category="linear", symbol=symbol)
    try:
        info = r["result"]["list"][0]["lotSizeFilter"]
        return float(info.get("minOrderQty", 0.001)), float(info.get("qtyStep", 0.001))
    except Exception as e:
        logging.warning(f"lotSizeFilter parse failed: {e}; using 0.001 defaults")
        return 0.001, 0.001

def round_to_step(qty, step):
    if step <= 0: return qty
    return (qty // step) * step if step >= 1 else int(qty/step)*step

API_KEY    = os.getenv("BYBIT_API_KEY","")
API_SECRET = os.getenv("BYBIT_API_SECRET","")
SYMBOL     = os.getenv("SYMBOL","BTCUSDT")
TESTNET    = os.getenv("TESTNET","true").lower() == "true"
BOT_PAUSED = os.getenv("BOT_PAUSED","false").lower() == "true"

LEVERAGE_MAX        = 50       # upper bound (risk agent chooses within)
POSITION_SIZE_PCT   = 0.05     # 5% of equity per trade
DRAWDOWN_STOP       = 0.20     # stop if equity down >20%

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logging.info(f"Booting bot | symbol={SYMBOL} | testnet={TESTNET} | paused={BOT_PAUSED}")

# ---------- bybit clients ----------
RECV_WINDOW = int(os.getenv('RECV_WINDOW','60000'))
session = HTTP(testnet=TESTNET, api_key=API_KEY, api_secret=API_SECRET, recv_window=RECV_WINDOW)
ws      = WebSocket(testnet=TESTNET, channel_type="linear")

# ---------- tiny LSTM + ensemble ----------
class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=50, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out,_ = self.lstm(x)
        return self.fc(out[:, -1, :])

model    = LSTMModel()
scaler   = MinMaxScaler()
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
_gb_is_fitted = False  # until you add a training script

# Q-table (simple RL leverage)
q_table = {}
def state_from_vol(vol):
    return 'low' if vol<0.01 else ('med' if vol<0.03 else 'high')
def choose_leverage(state, eps=0.1):
    if np.random.rand()<eps: return np.random.randint(1, LEVERAGE_MAX+1)
    if state not in q_table: q_table[state]=[0]*LEVERAGE_MAX
    return int(np.argmax(q_table[state])+1)
def update_q(reward, state, action):
    if state not in q_table: q_table[state]=[0]*LEVERAGE_MAX
    q_table[state][action-1] += 0.1*(reward - q_table[state][action-1])

# ---------- helpers ----------
def compute_rsi(close, period=14):
    delta = close.diff()
    gain  = (delta.where(delta>0, 0)).rolling(period).mean()
    loss  = (-delta.where(delta<0,0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100/(1+rs))

def fetch_klines():
    # Bybit expects interval string like "60" for 1h
    res = api_call("get_kline", session.get_kline, category="linear", symbol=SYMBOL, interval="60", limit=200)
    kl  = res["result"]["list"]
    df  = pd.DataFrame(kl, columns=['time','open','high','low','close','volume','turnover']).astype(float)
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df = df.sort_values('time').set_index('time')
    df['ma50'] = df['close'].rolling(50).mean()
    df['rsi']  = compute_rsi(df['close'])
    df['volatility'] = df['close'].pct_change().rolling(20).std()
    return df.dropna()

def pred_price(df):
    # LSTM (untrained) + GB (guarded). Train later per Grok plan.
    feats   = df[['open','high','low','close','volume']].values
    scaled  = scaler.fit_transform(feats).reshape(1,-1,5)
    x_t     = torch.tensor(scaled, dtype=torch.float32)
    with torch.no_grad():
        lstm_pred = float(model(x_t).item())
    gb_pred = 0.0
    if _gb_is_fitted:
        try:
            gb_pred = float(gb_model.predict(feats[-1].reshape(1,-1))[0])
        except NotFittedError:
            gb_pred = 0.0
    return (lstm_pred + gb_pred)/2.0

def set_leverage(leverage:int):
    session.set_leverage(category="linear", symbol=SYMBOL,
                         buy_leverage=str(leverage), sell_leverage=str(leverage))

def place(side:str, qty:float):
    return session.place_order(category="linear", symbol=SYMBOL,
                               side=("Buy" if side=='long' else "Sell"),
                               order_type="Market", qty=str(qty), reduce_only=False)

def wallet_equity():
    res = session.get_wallet_balance(accountType="UNIFIED")
    try:
        return float(res['result']['list'][0]['totalEquity'])
    except Exception:
        for acct in res.get('result',{}).get('list',[]):
            if 'totalEquity' in acct:
                return float(acct['totalEquity'])
        return 0.0

def close_all_positions():
    try:
        pos = session.get_positions(category="linear", symbol=SYMBOL)["result"]["list"]
        for p in pos:
            side = "Sell" if p["side"]=="Buy" else "Buy"
            qty  = float(p.get("size", 0))
            if qty>0:
                session.place_order(category="linear", symbol=SYMBOL,
                                    side=side, order_type="Market", qty=str(qty), reduce_only=True)
        logging.info("Requested close of all positions.")
    except Exception as e:
        logging.error(f"close_all_positions error: {e}")

# ---------- main loop ----------

def bootstrap_bybit():
    try:
        r = api_call("switch_position_mode", session.switch_position_mode,
                     category="linear", symbol=SYMBOL, mode=0)  # 0 = One-Way
        # Log the response explicitly too
        logging.info(f"switch_position_mode raw: {r}")
    except Exception as e:
        logging.warning(f"switch_position_mode failed: {e}")

import math

MIN_QTY = 0.0
QTY_STEP = 0.0

def get_symbol_filters(session, symbol):
    r = api_call("get_instruments_info", session.get_instruments_info,
                 category="linear", symbol=symbol)
    try:
        info = r["result"]["list"][0]["lotSizeFilter"]
        min_qty = float(info.get("minOrderQty", 0.0))
        qty_step = float(info.get("qtyStep", 0.0))
        return min_qty, qty_step
    except Exception as e:
        logging.warning(f"lotSizeFilter parse failed: {e}; using 0.001 defaults")
        return 0.001, 0.001

def round_to_step(qty, step):
    if step <= 0:
        return qty
    return math.floor(qty / step) * step

def main():
    bootstrap_bybit()
    if not API_KEY or not API_SECRET:
        logging.error("Missing BYBIT_API_KEY / BYBIT_API_SECRET. Set them in .env.")
        return
    eq0 = wallet_equity()
    eq  = eq0
    logging.info("Bot loop started.")
    while True:
        try:
            if os.getenv("BOT_PAUSED","false").lower()=="true":
                logging.info("BOT_PAUSED=true — skipping entries (management only).")
                time.sleep(30); continue

            df   = fetch_klines()
            pr   = pred_price(df)
            cur  = float(df['close'].iloc[-1])
            vol  = float(df['volatility'].iloc[-1])
            st   = state_from_vol(vol)
            lev  = choose_leverage(st)

            # entry logic (per Grok thresholds)
            action = None
            if pr > cur*1.005 and vol < 0.05: action = 'long'
            elif pr < cur*0.995:              action = 'short'

            if action:
                set_leverage(lev)
                qty = max((eq*POSITION_SIZE_PCT)/cur, 0.0001)  # exchange precision varies
                place(action, qty)
                logging.info(f"Executed {action} lev={lev} qty={qty:.6f} price={cur}")
                update_q(1 if action=='long' else -1, st, lev)

            eq = wallet_equity()
            if eq0 > 0:
                dd = (eq0 - eq)/eq0
                if dd > DRAWDOWN_STOP:
                    logging.warning(f"Drawdown {dd:.2%} > {DRAWDOWN_STOP:.0%}. Pausing & closing.")
                    close_all_positions()
                    break

            time.sleep(60)
        except Exception as e:
            logging.error(f"Loop error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()


def execute_trade(action, qty, leverage):
    # Guard qty vs filters
    qty = round_to_step(qty, QTY_STEP)
    if qty <= 0 or (MIN_QTY > 0 and qty < MIN_QTY):
        logging.warning(f"qty {qty} < MIN_QTY {MIN_QTY}; skipping")
        return False

    side = "Buy" if action == "long" else "Sell"

    r = api_call("place_order", session.place_order,
                 category="linear",
                 symbol=SYMBOL,
                 side=side,
                 order_type="Market",
                 qty=str(qty),
                 position_idx=0,         # required in one-way (Grok)
                 reduce_only=False)
    ok = r.get("retCode", 99999) == 0
    if not ok:
        logging.error(f"place_order failed: {r}")
    return ok
