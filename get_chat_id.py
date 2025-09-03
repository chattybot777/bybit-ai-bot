import os, json, sys, requests
from dotenv import load_dotenv

# Load .env from the current working directory explicitly
load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))

# Allow override via CLI arg: python get_chat_id.py 123:ABC...
TOKEN = sys.argv[1] if len(sys.argv) > 1 else os.getenv("TELEGRAM_TOKEN", "")

if not TOKEN:
    print("ERROR: TELEGRAM_TOKEN missing. Put it in .env or pass it as an argument.")
    raise SystemExit(1)

url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
r = requests.get(url, timeout=15)
data = r.json()
print("Full response:", json.dumps(data, indent=2))

for item in data.get("result", []):
    msg = item.get("message") or item.get("channel_post") or {}
    chat = msg.get("chat", {})
    if "id" in chat:
        print("Your chat_id is:", chat["id"])
        break
else:
    print("No chat found. In Telegram, open your bot, tap Start, send 'hi', then run this again.")
