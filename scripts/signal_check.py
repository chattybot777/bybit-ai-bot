import os, json, pickle, numpy as np, pandas as pd, torch, torch.nn as nn
from pybit.unified_trading import HTTP

# ---- config ----
SYM     = os.getenv("SYMBOL","BTCUSDT")
TESTNET = os.getenv("TESTNET","true").lower() == "true"
RECV    = int(os.getenv("RECV_WINDOW","60000"))
DELTA_STD = 0.5   # % gate for standard entries
DELTA_OVR = 1.0   # % gate for override (ignore RSI/MA if risk ok)
RISK_LIM  = 5.0   # risk_score < 5 required

def rsi(series, period=14):
    d = series.diff()
    gain = d.clip(lower=0).rolling(period).mean()
    loss = (-d.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100/(1+rs))

class LSTMModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=50, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out,_ = self.lstm(x)
        return self.fc(out[:, -1, :])

def load_artifacts():
    for f in ("lstm.pth","scaler.pkl","gb.pkl"):
        if not os.path.exists(f):
            raise FileNotFoundError(f"{f} is missing. Pull latest models or train first.")
    model = LSTMModel()
    state = torch.load("lstm.pth", map_location="cpu")
    model.load_state_dict(state)
    with open("scaler.pkl","rb") as f: scaler = pickle.load(f)
    with open("gb.pkl","rb") as f: gb = pickle.load(f)
    model.eval()
    return model, scaler, gb

def fetch_df():
    s = HTTP(testnet=TESTNET,
             api_key=os.getenv("BYBIT_API_KEY"),
             api_secret=os.getenv("BYBIT_API_SECRET"),
             recv_window=RECV)
    r = s.get_kline(category="linear", symbol=SYM, interval="60", limit=200)
    rows = r["result"]["list"]
    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume","turnover"])
    for col in ["open","high","low","close","volume","turnover"]:
        df[col] = df[col].astype(float)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.sort_values("ts").set_index("ts")
    df["ma50"]       = df["close"].rolling(50).mean()
    df["rsi"]        = rsi(df["close"])
    df["volatility"] = df["close"].pct_change().rolling(20).std()
    df = df.dropna().copy()
    return df

def predict_price(df, model, scaler, gb):
    feats = df[["open","high","low","close","volume","ma50","rsi","volatility"]].values
    scaled = scaler.transform(feats)
    X_lstm = torch.tensor(scaled.reshape(1, -1, 8), dtype=torch.float32)
    with torch.no_grad():
        lstm_pred = float(model(X_lstm).item())
    gb_pred = float(gb.predict(feats[-1].reshape(1, -1))[0])
    current = float(df["close"].iloc[-1])
    blended = (lstm_pred + gb_pred) / 2.0
    low, high = current * 0.9, current * 1.1
    blended = max(min(blended, high), low)  # clamp ±10%
    return current, blended

def main():
    model, scaler, gb = load_artifacts()
    df = fetch_df()
    current, pred = predict_price(df, model, scaler, gb)

    delta_pct = (pred / current - 1.0) * 100.0
    vol       = float(df["volatility"].iloc[-1])
    risk      = vol * 100.0
    rsi_now   = float(df["rsi"].iloc[-1])
    ma50_now  = float(df["ma50"].iloc[-1])
    trend_up  = bool(current > ma50_now)

    passes_override = bool((abs(delta_pct) >= DELTA_OVR) and (risk < RISK_LIM))
    standard_long   = bool((delta_pct >=  DELTA_STD) and (risk < RISK_LIM) and (rsi_now < 50.0) and  trend_up)
    standard_short  = bool((delta_pct <= -DELTA_STD) and (risk < RISK_LIM) and (rsi_now > 50.0) and (not trend_up))

    would_long  = bool((passes_override and delta_pct > 0) or standard_long)
    would_short = bool((passes_override and delta_pct < 0) or standard_short)

    out = {
        "symbol": SYM,
        "current_price": round(current, 2),
        "predicted_price": round(pred, 2),
        "delta_pct": round(delta_pct, 3),
        "risk_score": round(risk, 2),
        "rsi": round(rsi_now, 2),
        "price_vs_ma50": "above" if trend_up else "below",
        "passes_override_1pct_and_risk_lt_5": passes_override,
        "standard_long_gate": standard_long,
        "standard_short_gate": standard_short,
        "would_long": would_long,
        "would_short": would_short
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
