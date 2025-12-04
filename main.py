# This is the entry point for the trading bot service deployed on Render.
# Its purpose is solely to initialize the core trading logic and handle critical startup errors.
#
# Your 800+ lines of trading logic MUST be located in a separate file named 
# 'online_trading_inversions.py' in the same directory.
#
# NOTE: This file is intentionally short. It is a runner, not the core logic.

# FIX: Changed the import from the original absolute path (e.g., from package.file import Class) 
# to a relative path to resolve the "ModuleNotFoundError" seen in the deployment logs.
try:
    # Attempt to import your main trading class from the local file
    from online_trading_inversions import OnlineTradingInversions
except ImportError as e:
    # If this fails, the core trading logic file is either missing or misspelled.
    print(f"CRITICAL ERROR: Failed to import the core bot logic. Check that 'online_trading_inversions.py' is present. Details: {e}")
    # Exit gracefully if the core logic is not found.
    exit(1)


if __name__ == '__main__':
    # Initialize the core trading application.
    try:
        print("Initializing the bot (reading environment variables and loading models)...")
        
        # Instantiate the bot class. This class (from online_trading_inversions.py) 
        # is responsible for reading credentials (API_KEY, API_SECRET) 
        # from the Render environment variables and performing all trading actions.
        bot = OnlineTradingInversions()
        
        # Start the main bot process (e.g., streaming data, running the model, placing orders)
        print("SUCCESS: Bot initialized. Starting main execution loop...")
        bot.run_bot() 
        
    except Exception as e:
        # Catch any unexpected critical errors during startup or execution.
        print(f"CRITICAL BOT FAILURE: The bot encountered a severe error and has stopped. Details: {e}")