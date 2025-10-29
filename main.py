import asyncio
import json
import os
import signal
import time
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from bot_fib_scoring import BotController, ACCOUNTS_FILE, TRADES_FILE, ALLOWED_COINS, TRADE_SETTINGS

load_dotenv()

APP_PORT = int(os.getenv("PORT", "8000"))
LIVE_MODE = os.getenv("LIVE_MODE", "False").lower() in ("1", "true", "yes")

app = FastAPI(
    title="MGX Crypto Trading Bot Backend",
    description="Backend API for MGX Crypto Trading Bot",
    version="1.0.0"
)

# CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your specific frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BC = BotController()
PRICE_POLL_INTERVAL = float(os.getenv("PRICE_POLL_INTERVAL", "3.0"))


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


class BotCommand(BaseModel):
    action: str


class PlaceOrderRequest(BaseModel):
    account_id: str
    symbol: str
    side: str
    notional_usd: Optional[float] = None
    qty: Optional[float] = None


class AccountRequest(BaseModel):
    name: str
    api_key: str
    api_secret: str
    monitoring: bool = False


def load_json(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return []


def save_json(path, data):
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving JSON to {path}: {e}")


def get_bybit_price(symbol: str):
    try:
        r = requests.get(
            "https://api.bybit.com/v5/market/tickers", 
            params={"category": "spot", "symbol": symbol},
            timeout=10
        )
        r.raise_for_status()
        j = r.json()
        if j.get("retCode") == 0 and j.get("result", {}).get("list"):
            return float(j["result"]["list"][0]["lastPrice"])
        return None
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
        return None


price_cache: Dict[str, float] = {}


async def price_loop():
    """Background task to continuously fetch and broadcast price updates"""
    while True:
        try:
            for sym in ALLOWED_COINS:
                try:
                    price = get_bybit_price(sym)
                    if price:
                        price_cache[sym] = price
                        await ws_manager.broadcast({
                            "type": "price", 
                            "symbol": sym, 
                            "price": price,
                            "timestamp": int(time.time())
                        })
                except Exception as e:
                    print(f"Price loop error for {sym}: {e}")
            await asyncio.sleep(PRICE_POLL_INTERVAL)
        except Exception as e:
            print(f"Price loop critical error: {e}")
            await asyncio.sleep(5)  # Wait before retrying


@app.post("/bot")
def control_bot(cmd: BotCommand):
    """Control bot start/stop/status"""
    try:
        if cmd.action == "start":
            if BC.is_running():
                return {"message": "Bot is already running", "status": "running"}
            BC.start()
            return {"message": "Bot started successfully", "status": "running"}
        elif cmd.action == "stop":
            BC.stop()
            return {"message": "Bot stopped successfully", "status": "stopped"}
        elif cmd.action == "status":
            return {
                "running": BC.is_running(),
                "status": "running" if BC.is_running() else "stopped"
            }
        else:
            raise HTTPException(status_code=400, detail="Unknown action. Use 'start', 'stop', or 'status'")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bot control error: {str(e)}")


@app.get("/status")
def get_status():
    """Get overall system status"""
    try:
        return {
            "running": BC.is_running(),
            "dry_run": TRADE_SETTINGS.get("dry_run", True),
            "live_mode": LIVE_MODE,
            "price_cache_size": len(price_cache),
            "allowed_coins": len(ALLOWED_COINS),
            "timestamp": int(time.time())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status error: {str(e)}")


@app.get("/accounts")
def get_accounts():
    """Get all trading accounts"""
    try:
        return load_json(ACCOUNTS_FILE)
    except Exception as e:
        raise HTTPExcept
# Import routers
from routes.config_routes import router as config_router
from routes.accounts_routes import router as accounts_router
from routes.bot_routes import router as bot_router
from routes.dashboard_routes import router as dashboard_router
from routes.history_routes import router as history_router

# Register routers
app.include_router(config_router, prefix="/api/config", tags=["Config"])
app.include_router(accounts_router, prefix="/api/accounts", tags=["Accounts"])
app.include_router(bot_router, prefix="/api/bot", tags=["Bot"])
app.include_router(dashboard_router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(history_router, prefix="/api/history", tags=["History"])
