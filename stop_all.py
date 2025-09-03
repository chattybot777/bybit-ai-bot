import os
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

load_dotenv()
TESTNET = os.getenv("TESTNET","true").lower()=="true"
API_KEY = os.getenv("BYBIT_API_KEY","")
API_SECRET = os.getenv("BYBIT_API_SECRET","")
SYMBOL = os.getenv("SYMBOL","BTCUSDT")

session = HTTP(testnet=TESTNET, api_key=API_KEY, api_secret=API_SECRET)

def close_all():
    try:
        pos = session.get_positions(category="linear", symbol=SYMBOL)["result"]["list"]
        for p in pos:
            side = "Sell" if p["side"]=="Buy" else "Buy"
            qty  = float(p.get("size", 0))
            if qty>0:
                session.place_order(category="linear", symbol=SYMBOL,
                                    side=side, order_type="Market", qty=qty, reduce_only=True)
        print("Close requests sent.")
    except Exception as e:
        print("Error closing positions:", e)

if __name__ == "__main__":
    close_all()
    print("Tip: set BOT_PAUSED=true in env to block new entries.")
