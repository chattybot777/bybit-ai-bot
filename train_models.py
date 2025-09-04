import os, pickle, math
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import ccxt

# Reuse model def from main spec
class LSTMModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=50, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out,_ = self.lstm(x)
        return self.fc(out[:,-1,:])

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def prepare_df():
    ex = ccxt.bybit({'enableRateLimit': True})
    ohlcv = ex.fetch_ohlcv('BTC/USDT:USDT', timeframe='1h', limit=17520)  # ~2y
    df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
    df['ma50'] = df['close'].rolling(50).mean()
    df['rsi'] = compute_rsi(df['close'],14)
    df['volatility'] = df['close'].pct_change().rolling(20).std()
    df.dropna(inplace=True)
    return df

def make_sequences(feats, targets, seq=32):
    X, y = [], []
    for i in range(seq, len(feats)):
        X.append(feats[i-seq:i])
        y.append(targets[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def main():
    df = prepare_df()
    feats = df[['open','high','low','close','volume','ma50','rsi','volatility']].values.astype('float32')
    targs = df['close'].shift(-1).values.astype('float32')
    feats = feats[:-1]; targs = targs[:-1]  # align

    scaler = MinMaxScaler()
    feats_scaled = scaler.fit_transform(feats)

    seq = 32
    X, y = make_sequences(feats_scaled, targs, seq)
    n = int(0.8*len(X))
    Xtr, Xva = X[:n], X[n:]
    ytr, yva = y[:n], y[n:]

    ds = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    model = LSTMModel(input_size=8, hidden_size=50, num_layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()

    for epoch in range(50):
        model.train()
        tot = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            out = model(xb).squeeze(-1)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            tot += float(loss.item())
        if (epoch+1) % 10 == 0:
            print(f"epoch {epoch+1}: train_loss={tot/len(dl):.6f}")

    torch.save(model.state_dict(), "lstm.pth")
    with open("scaler.pkl","wb") as f:
        pickle.dump(scaler, f)

    # GB on raw feats (last row prediction uses same shape)
    from sklearn.ensemble import GradientBoostingRegressor
    gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3)
    gb.fit(feats, targs)
    with open("gb.pkl","wb") as f:
        pickle.dump(gb, f)

    print("Saved: lstm.pth, scaler.pkl, gb.pkl")

if __name__ == "__main__":
    main()
