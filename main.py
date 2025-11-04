import asyncio
import os
import time
from typing import Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from bot_fib_scoring import BotController, ALLOWED_COINS

# Import routers
from routes.config_routes import router as config_router
from routes.accounts_routes import router as accounts_router
from routes.bot_routes import router as bot_router
from routes.dashboard_routes import router as dashboard_router
from routes.history_routes import router as history_router

# -----------------------
# Load environment
# -----------------------
load_dotenv()

APP_PORT = int(os.getenv("PORT", "8000"))
LIVE_MODE = os.getenv("LIVE_MODE", "False").lower() in ("1", "true", "yes")

app = FastAPI(
    title="MGX Crypto Trading Bot Backend",
    description="Backend API for MGX Crypto Trading Bot",
    version="1.0.0"
)

# -----------------------
# CORS
# -----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://aitrader.mgx.world"],  # Replace with frontend URL when stable
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -----------------------
# Root Endpoint
# -----------------------
@app.get("/")
def root():
    return {"message": "MGX Trading Bot Backend is running!"}
    
# -----------------------
# Bot Controller and Price Manager
# -----------------------
BC = BotController()
PRICE_POLL_INTERVAL = float(os.getenv("PRICE_POLL_INTERVAL", "3.0"))
price_cache: Dict[str, float] = {}

# WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        try:
            self.active.remove(ws)
        except Exception:
            pass

    async def broadcast(self, message: dict):
        for ws in list(self.active):
            try:
                await ws.send_json(message)
            except Exception:
                self.disconnect(ws)

ws_manager = ConnectionManager()

# -----------------------
# Background Price Loop (optional)
# -----------------------
async def price_loop():
    """Background task to fetch & broadcast prices."""
    import requests
    while True:
        try:
            for sym in ALLOWED_COINS:
                try:
                    r = requests.get(
                        "https://api.bybit.com/v5/market/tickers",
                        params={"category": "spot", "symbol": sym},
                        timeout=10
                    )
                    j = r.json()
                    if j.get("retCode") == 0 and j.get("result", {}).get("list"):
                        price = float(j["result"]["list"][0]["lastPrice"])
                        price_cache[sym] = price
                        await ws_manager.broadcast({
                            "type": "price",
                            "symbol": sym,
                            "price": price,
                            "timestamp": int(time.time())
                        })
                except Exception as e:
                    print(f"Price fetch error for {sym}: {e}")
            await asyncio.sleep(PRICE_POLL_INTERVAL)
        except Exception as e:
            print(f"Critical price loop error: {e}")
            await asyncio.sleep(5)

# -----------------------
# Include Routers
# -----------------------
app.include_router(config_router, prefix="/config", tags=["Config"])
app.include_router(accounts_router, prefix="/accounts", tags=["Accounts"])
app.include_router(bot_router, prefix="/bot", tags=["Bot"])
app.include_router(dashboard_router, prefix="/dashboard", tags=["Dashboard"])
app.include_router(history_router, prefix="/history", tags=["History"])

# -----------------------
# Startup Event
# -----------------------
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(price_loop())
    print("✅ MGX Trading Bot Backend started successfully")

# -----------------------
# Test write endpoint
# -----------------------
@app.get("/api/test-write")
def test_write():
    """Check if the backend can write to the filesystem (for app/accounts.json)."""
    try:
        test_file = "accounts_test.json"
        with open(test_file, "w") as f:
            f.write("test ok")
        return {"success": True, "message": f"Write successful: {os.path.abspath(test_file)}"}
    except Exception as e:
        return {"success": False, "error": str(e)}
