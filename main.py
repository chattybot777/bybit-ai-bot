import os
import time
import logging
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

# Setup simple logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
load_dotenv()

def main():
    print("--- STARTING SIMPLE BOT DIAGNOSTIC ---")
    
    # Get keys from environment
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    testnet = True # Force testnet for safety

    if not api_key or not api_secret:
        print("CRITICAL: API Key or Secret is MISSING from environment variables.")
        return

    print(f"API Key found (Length: {len(api_key)})")
    
    # Connect to Bybit
    try:
        session = HTTP(testnet=testnet, api_key=api_key, api_secret=api_secret)
        print("Network connection initialized...")
        
        # Test Call: Get Wallet Balance
        resp = session.get_wallet_balance(accountType="UNIFIED")
        
        if resp['retCode'] == 0:
            equity = resp['result']['list'][0]['totalEquity']
            print(f"SUCCESS! Connection Verified. Total Equity: {equity} USDT")
        else:
            print(f"API ERROR: {resp['retMsg']}")
            
    except Exception as e:
        print(f"CRITICAL EXCEPTION: {e}")

    print("--- DIAGNOSTIC COMPLETE. SLEEPING... ---")
    while True:
        time.sleep(60)

if __name__ == "__main__":
    main()