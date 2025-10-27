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
def get_aat(raw[4])  # close price
                    except Exception:
                        pass
                
                # Try each item
                for it in raw:
                    p = self._parse_price(it, depth + 1, visited)
                    if p is not None:
                        return p
            
            return None
        except RecursionError:
            return None
        except Exception:
            return None

    def _extract_balance_from_payload(self, payload: Any) -> Optional[float]:
        try:
            if payload is None:
                return None
            
            if isinstance(payload, dict):
                # Try direct balance fields
                for k in ("total_equity", "equity", "totalBalance", "total_balance"):
                    if k in payload:
                        try:
                            return float(payload[k])
                        except Exception:
                            pass
                
                # Try balance lists
                for lk in ("list", "balances", "rows", "data", "wallets"):
                    if lk in payload and isinstance(payload[lk], list):
                        total = 0.0
                        for item in payload[lk]:
                            try:
                                coin = item.get("coin") or item.get("symbol")
                                free = (item.get("free") or 
                                       item.get("available") or 
                                       item.get("walletBalance") or 
                                       item.get("available_balance"))
                                if free is None:
                                    continue
                                freef = float(free)
                                if coin and str(coin).upper() in ("USDT", "USDC"):
                                    total += freef
                            except Exception:
                                continue
                        if total > 0:
                            return total
                
                # Try coin-keyed balances
                if payload and all(isinstance(v, dict) for v in payload.values()):
                    total = 0.0
                    for coin, info in payload.items():
                        try:
                            available = (info.get("available") or 
                                       info.get("free") or 
                                       info.get("walletBalance"))
                            if available is None:
                                continue
                            if str(coin).upper() in ("USDT", "USDC"):
                                total += float(available)
                        except Exception:
                            continue
                    if total > 0:
                        return total
            
            if isinstance(payload, list):
                total = 0.0
                for item in payload:
                    try:
                        coin = item.get("coin") or item.get("symbol")
                        free = (item.get("free") or 
                               item.get("available") or 
                               item.get("walletBalance"))
                        if free is None:
                            continue
                        freef = float(free)
                        if coin and str(coin).upper() in ("USDT", "USDC"):
                            total += freef
                    except Exception:
                        continue
                if total > 0:
                    return total
            
            return None
        except Exception:
            return None

    def validate_account(self, account: Dict[str, Any]) -> Tuple[bool, Optional[float], str]:
        client = self._get_client(account)
        if not client:
            return False, None, "missing_api_credentials"
        
        raw_preview: Dict[str, Any] = {}
        balance = None
        last_err = ""
        
        # Try different balance endpoints
        candidate_attempts = [
            (["get_wallet_balance", "query_wallet_balance"], {"accountType": "UNIFIED"}),
            (["get_wallet_balance", "query_wallet_balance"], {"accountType": "SPOT"}),
            (["get_wallet_balance", "query_wallet_balance", "get_balances", "balance"], {})
        ]
        
        for candidate_methods, params in candidate_attempts:
            try:
                if params:
                    resp = self._try_methods(client, candidate_methods, **params)
                else:
                    resp = self._try_methods(client, candidate_methods)
                
                self._capture_preview(account, resp, label="balance")
                
                # Try to extract balance from different response structures
                payloads = []
                if isinstance(resp, dict):
                    if "result" in resp:
                        payloads.append(resp["result"])
                    if "data" in resp:
                        payloads.append(resp["data"])
                    payloads.append(resp)
                else:
                    payloads.append(resp)
                
                for payload in payloads:
                    bal = self._extract_balance_from_payload(payload)
                    if bal is not None:
                        balance = bal
                        break
                
                if balance is not None:
                    break
                    
            except Exception as e:
                last_err = str(e)
                try:
                    raw_preview[f"err_{len(raw_preview)+1}"] = f"error: {str(e)}"
                except Exception:
                    pass
                continue
        
        # Store debug info if enabled
        if TRADE_SETTINGS.get('debug_raw_responses', False) and raw_preview:
            try:
                account.setdefault('last_raw_preview', {})
                account['last_raw_preview'].update(safe_json(raw_preview))
                keys = list(account['last_raw_preview'].keys())
                if len(keys) > 3:
                    for k in keys[:-3]:
                        account['last_raw_preview'].pop(k, None)
            except Exception:
                pass
        
        if balance is not None:
            return True, balance, ""
        return False, None, last_err or "unrecognized_balance_shape"

    def _place_market_order(self, client: HTTP, symbol: str, side: str, qty: float, price_hint: Optional[float] = None) -> Dict[str, Any]:
        if TRADE_SETTINGS.get('dry_run', True):
            executed_price = float(price_hint) if price_hint is not None else None
            resp = {
                "simulated": True, 
                "symbol": symbol, 
                "side": side, 
                "qty": qty, 
                "executed_price": executed_price, 
                "time": now_iso()
            }
            return resp
        
        candidate_methods = ["place_active_order", "create_order", "place_order", "order"]
        last_exc = None
        
        for name in candidate_methods:
            meth = getattr(client, name, None)
            if not callable(meth):
                continue
            
            # Try different parameter formats
            attempts = [
                {"category": "linear", "symbol": symbol, "side": side, "orderType": "Market", "qty": qty},
                {"symbol": symbol, "side": side, "order_type": "Market", "qty": qty},
                {"symbol": symbol, "side": side, "type": "market", "qty": qty},
                {"symbol": symbol, "side": side, "orderType": "Market", "qty": qty},
            ]
            
            for params in attempts:
                try:
                    resp = meth(**params)
                    self._capture_preview({}, resp, label="order")
                    try:
                        return resp if isinstance(resp, dict) else {"result": resp}
                    except Exception:
                        return {"result": str(resp)}
                except Exception as e:
                    last_exc = e
                    continue
        
        self.log(f"Order placement failed for {symbol}: {last_exc}")
        return {"error": str(last_exc) if last_exc else "order_failed"}

    def _record_trade_entry(self, acct: Dict[str, Any], symbol: str, qty: float, entry_price: float, ts: int, simulated: bool, sl_price: Optional[float], tp_price: Optional[float]) -> str:
        tid = str(uuid.uuid4())
        trade = {
            "id": tid,
            "account_id": acct.get("id"),
            "account_name": acct.get("name"),
            "symbol": symbol,
            "side": "Buy",
            "qty": qty,
            "entry_price": entry_price,
            "entry_time": datetime.utcfromtimestamp(ts).isoformat(),
            "open": True,
            "simulated": bool(simulated),
            "stop_loss_price": sl_price,
            "take_profit_price": tp_price,
        }
        self.add_trade(trade)
        return tid

    def _finalize_trade(self, trade_id: str, exit_price: float, exit_ts: int, label: str, resp_summary: Optional[str], simulated: bool):
        trades = self._read_trades()
        entry = None
        for t in trades:
            if t.get('id') == trade_id:
                entry = t
                break
        
        if not entry:
            elapsed_s = 0
            profit_pct = 0.0
            rec = {
                "id": trade_id,
                "symbol": None,
                "side": "Buy",
                "qty": None,
                "entry_price": None,
                "entry_time": None,
                "exit_price": exit_price,
                "exit_time": datetime.utcfromtimestamp(exit_ts).isoformat(),
                "elapsed": fmt_elapsed(elapsed_s),
                "elapsed_seconds": elapsed_s,
                "profit_pct": profit_pct,
                "label": label,
                "simulated": bool(simulated),
                "resp_summary": resp_summary,
            }
            self.add_trade(rec)
            return
        
        entry_price = entry.get('entry_price')
        entry_time = entry.get('entry_time')
        
        try:
            entry_ts = int(datetime.fromisoformat(entry_time).timestamp()) if entry_time else exit_ts
        except Exception:
            try:
                entry_ts = int(datetime.fromisoformat(entry_time.replace('Z', '')).timestamp())
            except Exception:
                entry_ts = exit_ts
        
        elapsed_s = max(0, exit_ts - entry_ts)
        profit_pct = None
        if entry_price is not None and exit_price is not None:
            try:
                profit_pct = ((float(exit_price) - float(entry_price)) / float(entry_price)) * 100.0
            except Exception:
                profit_pct = None
        
        updates = {
            "exit_price": exit_price,
            "exit_time": datetime.utcfromtimestamp(exit_ts).isoformat(),
            "elapsed": fmt_elapsed(elapsed_s),
            "elapsed_seconds": elapsed_s,
            "profit_pct": profit_pct,
            "label": label,
            "open": False,
            "resp_summary": resp_summary,
            "simulated": bool(simulated),
        }
        self.update_trade(trade_id, updates)

    # ---- scoring & decision helpers ----
    def score_symbol(self, client: HTTP, symbol: str) -> Tuple[int, Dict[str, Any]]:
        """Return a score and diagnostics for a symbol.
        Diagnostics contains fields: rsi, ema50, ema200, momentum_pct, candle_ok, fib_levels, price, score_breakdown
        """
        diagnostics: Dict[str, Any] = {}
        try:
            # fetch klines (1m by default)
            raw_klines = self.safe_get_klines(client, symbol, interval='1', limit=200)
            self._capture_preview({}, raw_klines, label='klines')
        except Exception as e:
            return 0, {"error": f"klines_fetch_failed: {e}"}

        # parse klines to close prices & OHLC
        closes: List[float] = []
        ohlc: List[Dict[str, float]] = []
        try:
            # attempt to normalize several shapes
            if isinstance(raw_klines, dict) and 'result' in raw_klines:
                payload = raw_klines['result']
            elif isinstance(raw_klines, dict) and 'data' in raw_klines:
                payload = raw_klines['data']
            else:
                payload = raw_klines
            
            # payload may be list of lists or list of dicts
            if isinstance(payload, list):
                for item in payload:
                    if isinstance(item, (list, tuple)) and len(item) >= 5:
                        # common kline list: [t, open, high, low, close, ...]
                        o = float(item[1]); h = float(item[2]); l = float(item[3]); c = float(item[4])
                    elif isinstance(item, dict):
                        o = float(item.get('open') or item.get('Open') or item.get('o'))
                        h = float(item.get('high') or item.get('High') or item.get('h'))
                        l = float(item.get('low') or item.get('Low') or item.get('l'))
                        c = float(item.get('close') or item.get('Close') or item.get('c'))
                    else:
                        continue
                    closes.append(c)
                    ohlc.append({'open': o, 'high': h, 'low': l, 'close': c})
        except Exception:
            return 0, {"error": "klines_parse_failed"}

        if not closes:
            return 0, {"error": "no_closes"}

        # compute indicators
        rsi = calc_rsi(closes[-(SCORE_SETTINGS['rsi_period'] + 1) :])
        ema50 = calc_ema(closes[-100:], 50)
        ema200 = calc_ema(closes[-220:], 200)
        momentum_pct = ((closes[-1] - closes[-5]) / closes[-5]) * 100.0 if len(closes) >= 6 and closes[-5] != 0 else 0.0
        candle_ok = detect_bullish_candle(ohlc[-5:])

        # determine recent swing high/low (use past 50 candles)
        window = closes[-50:]
        swing_high = max(window)
        swing_low = min(window)
        fib = calc_fib_levels(swing_high, swing_low)
        current_price = closes[-1]

        score = 0
        breakdown = {}

        # RSI score
        if rsi is not None:
            breakdown['rsi'] = rsi
            if rsi <= SCORE_SETTINGS['rsi_oversold_threshold']:
                score += SCORE_SETTINGS['score_weights']['rsi']
        else:
            breakdown['rsi'] = None

        # momentum
        breakdown['momentum_pct'] = momentum_pct
        if abs(momentum_pct) >= SCORE_SETTINGS['momentum_entry_threshold_pct']:
            # positive momentum preferred
            if momentum_pct > 0:
                score += SCORE_SETTINGS['score_weights']['momentum']

        # EMA trend: price above EMA50 and EMA50 above EMA200
        breakdown['ema50'] = ema50
        breakdown['ema200'] = ema200
        if ema50 is not None and ema200 is not None:
            if current_price >= ema50 and ema50 >= ema200:
                score += SCORE_SETTINGS['score_weights']['ema']

        # candle
        breakdown['candle_ok'] = candle_ok
        if candle_ok:
            score += SCORE_SETTINGS['score_weights']['candle']

        # Fibonacci zone proximity
        breakdown['current_price'] = current_price
        breakdown['fib_levels'] = fib
        if price_in_zone(current_price, fib, lo_key='0.382', hi_key='0.618'):
            score += SCORE_SETTINGS['score_weights']['fib_zone']
            breakdown['fib_zone'] = True
        else:
            breakdown['fib_zone'] = False

        diagnostics.update(breakdown)
        diagnostics['score'] = score
        diagnostics['momentum_strong_pct'] = SCORE_SETTINGS['momentum_strong_pct']
        diagnostics['momentum_very_strong_pct'] = SCORE_SETTINGS['momentum_very_strong_pct']
        return score, diagnostics

    def attempt_trade_for_account(self, acct: Dict[str, Any]):
        """
        Updated flow:
        - For monitoring accounts, scan a set of allowed coins
        - Score each coin with score_symbol
        - Pick the highest scoring coin(s) (config: top 1)
        - Place buy and record trade; set SL from fib 78.6% and TP dynamic:
            - base TP = 1.272 extension
            - upgrade to 1.618 if momentum >= momentum_strong_pct or price > EMA50*1.01
            - upgrade to 2.618 if momentum >= momentum_very_strong_pct or pri
    def _scan_once(self):
        accounts = self.load_accounts()
        updated = False
        for acct in accounts:
            acct.setdefault('id', str(uuid.uuid4()))
            try:
                ok, bal, err = self.validate_account(acct)
                acct['validated'] = ok
                acct['balance'] = bal
                acct['last_validation_error'] = err
            except Exception as e:
                acct['validated'] = False
                acct['balance'] = None
                acct['last_validation_error'] = str(e)
            
            acct.setdefault('position', acct.get('position', 'closed'))
            acct.setdefault('monitoring', acct.get('monitoring', False))
            acct.setdefault('current_symbol', acct.get('current_symbol'))
            acct.setdefault('buy_price', acct.get('buy_price'))
            
            try:
                if acct.get('position') == 'open' and acct.get('open_trade_id'):
                    self._check_open_position(acct)
                else:
                    try:
                        self.attempt_trade_for_account(acct)
                    except Exception as e:
                        self.log(f"attempt_trade_for_account raised: {e}")
                
                acct['last_balance'] = acct.get('balance', acct.get('last_balance', 0.0))
                acct['last_validation_error'] = acct.get('last_validation_error')
            except Exception as e:
                self.log(f"Account scan error for {acct.get('id')}: {e}")
            updated = True
        
        if updated:
            self.save_accounts(accounts)

    def start(self):
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        t = threading.Thread(target=self._run_loop, daemon=True)
        self._threads.append(t)
        t.start()
        self.log('BotController started')

    def stop(self):
        self._stop_event.set()
        self._running = False
        self.log('Stop requested; waiting for threads to finish')
        for t in self._threads:
            if t.is_alive():
                t.join(timeout=1)
        self.log('Stopped')

    def _run_loop(self):
        while not self._stop_event.is_set():
            try:
                self._scan_once()
            except Exception as e:
                self.log(f"Run loop error: {e}")
            
            interval = int(TRADE_SETTINGS.get('scan_interval', 10))
            for _ in range(interval):
                if self._stop_event.is_set():
                    break
                time.sleep(1)


if __name__ == '__main__':
    bc = BotController()
    print("Superb Crypto Bot — debug run (dry_run is set to {} )".format(TRADE_SETTINGS.get('dry_run')))
    accts = bc.load_accounts()
    if not accts:
        print("No accounts found. Add accounts to accounts.json with api_key and api_secret to test validation.")
    else:
        for a in accts:
            ok, bal, err = bc.validate_account(a)
            print(f"Account id={a.get('id')} ok={ok} balance={bal} err={err}")
    bc._scan_once()
    print("Done")
