import joblib, ccxt, pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor

ex = ccxt.bybit({'enableRateLimit': True})
ohlcv = ex.fetch_ohlcv('BTC/USDT:USDT', timeframe='1h', limit=3000)
df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])

X = df[['open','high','low','close','volume']].astype('float32').values
y = df['close'].shift(-1).dropna().astype('float32').values
X = X[:-1]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

gb = GradientBoostingRegressor(random_state=42)
gb.fit(X_scaled, y)

joblib.dump(scaler, 'scaler.pkl')
joblib.dump(gb, 'gb.pkl')
print("Wrote scaler.pkl and gb.pkl with sklearn 1.5.1")
