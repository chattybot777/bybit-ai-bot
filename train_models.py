import pickle, time
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
import ccxt

def rsi(series, period=14):
    d = series.diff()
    gain = d.clip(lower=0).rolling(period).mean()
    loss = (-d.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100/(1+rs))

class LSTMModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=50, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)
    def forward(self, x):
        y, _ = self.lstm(x)
        return self.fc(y[:, -1, :])

exc = ccxt.bybit({'enableRateLimit': True})
sym = 'BTC/USDT:USDT'
tf  = '1h'
limit = 17520  # ~2y
ohlcv = exc.fetch_ohlcv(sym, timeframe=tf, limit=limit)
df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
df['ma50'] = df['close'].rolling(50).mean()
df['rsi']  = rsi(df['close'])
df['volatility'] = df['close'].pct_change().rolling(20).std()
df = df.dropna().reset_index(drop=True)

X = df[['open','high','low','close','volume','ma50','rsi','volatility']].values.astype(np.float32)
y = df['close'].shift(-1).dropna().values.astype(np.float32)
X = X[:-1]

scaler = MinMaxScaler()
Xs = scaler.fit_transform(X)

# LSTM train quick (small epochs for speed)
seq_len = 50
def make_seq(data, target, seq=50):
    Xs, ys = [], []
    for i in range(len(data)-seq):
        Xs.append(data[i:i+seq])
        ys.append(target[i+seq-1])
    return np.stack(Xs), np.array(ys)

Xl, yl = make_seq(Xs, y, seq_len)
ds = TensorDataset(torch.tensor(Xl, dtype=torch.float32),
                   torch.tensor(yl, dtype=torch.float32))
dl = DataLoader(ds, batch_size=64, shuffle=True)

model = LSTMModel(input_size=8)
opt = optim.Adam(model.parameters(), lr=1e-3)
lossf = nn.MSELoss()

for epoch in range(8):  # short train; cron will refine weekly
    for xb, yb in dl:
        opt.zero_grad()
        pred = model(xb).squeeze()
        loss = lossf(pred, yb)
        loss.backward()
        opt.step()

torch.save(model.state_dict(), 'lstm.pth')
with open('scaler.pkl','wb') as f: pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)

gb = GradientBoostingRegressor(random_state=42)
gb.fit(Xs, y)
with open('gb.pkl','wb') as f: pickle.dump(gb, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Wrote lstm.pth, scaler.pkl, gb.pkl")
