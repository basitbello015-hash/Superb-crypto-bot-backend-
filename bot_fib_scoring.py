from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# IMPORTANT: pybit unified_trading HTTP client
from pybit.unified_trading import HTTP

# -------------------- CONFIG --------------------

ALLOWED_COINS = [
    "ADAUSDT", "XRPUSDT", "TRXUSDT", "DOGEUSDT", "CHZUSDT", "VETUSDT", "BTTUSDT", "HOTUSDT",
    "XLMUSDT", "ZILUSDT", "IOTAUSDT", "SCUSDT", "DENTUSDT", "KEYUSDT", "WINUSDT", "CVCUSDT",
    "MTLUSDT", "CELRUSDT", "FUNUSDT", "STMXUSDT", "REEFUSDT", "ANKRUSDT", "ONEUSDT", "OGNUSDT",
    "CTSIUSDT", "DGBUSDT", "CKBUSDT", "ARPAUSDT", "MBLUSDT", "TROYUSDT", "PERLUSDT", "DOCKUSDT",
    "RENUSDT", "COTIUSDT", "MDTUSDT", "OXTUSDT", "PHAUSDT", "BANDUSDT", "GTOUSDT", "LOOMUSDT",
    "PONDUSDT", "FETUSDT", "SYSUSDT", "TLMUSDT", "NKNUSDT", "LINAUSDT", "ORNUSDT", "COSUSDT",
    "FLMUSDT", "ALICEUSDT",
]

# Risk rules: stop-loss will be set relative to fib 78.6% or fallback -1.0%
RISK_RULES = {
    "stop_loss": -1.0,    # fallback -1% if fib can't be computed
    "max_hold": 24 * 60 * 60,     # safety force exit after 24h by default
}

SCORE_SETTINGS = {
    "momentum_scale": 1.0,
    "rsi_period": 14,
    "rsi_oversold_threshold": 35,
    "rsi_overbought_threshold": 65,
    "momentum_entry_threshold_pct": 0.1,  # 0.1% default (quick momentum)
    "momentum_strong_pct": 0.5,           # >=0.5% considered strong momentum for TP extension
    "momentum_very_strong_pct": 1.5,      # >=1.5% considered very strong for 2.618
    "max_price_allowed": 1.2,
    "score_weights": {
        "rsi": 1,
        "momentum": 1,
        "ema": 1,
        "candle": 1,
        "fib_zone": 1,
    }
}

TRADE_SETTINGS = {
    "trade_allocation_pct": 0.01,
    "min_trade_amount": 5.0,
    "use_market_order": True,
    "test_on_testnet": False,
    "scan_interval": 10,
    "debug_raw_responses": False,
    "dry_run": True,
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ACCOUNTS_FILE = os.path.join(BASE_DIR, "accounts.json")
TRADES_FILE = os.path.join(BASE_DIR, "trades.json")

# Ensure files exist
for f in [ACCOUNTS_FILE, TRADES_FILE]:
    if not os.path.exists(f):
        with open(f, "w") as fp:
            fp.write("[]")

# -------------------- Utilities --------------------

def now_ts() -> int:
    return int(time.time())

def now_iso() -> str:
    return datetime.utcnow().isoformat()

def fmt_elapsed(seconds: int) -> str:
    m = seconds // 60
    s = seconds % 60
    return f"{m}m {s}s"

def safe_json(data: Any, max_depth: int = 4, _depth: int = 0, _seen: Optional[set] = None) -> Any:
    if _seen is None:
        _seen = set()
    try:
        if id(data) in _seen or _depth > max_depth:
            return "<recursion>"
        _seen.add(id(data))
    except Exception:
        pass

    if data is None:
        return None
    if isinstance(data, (str, int, float, bool)):
        return data
    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            try:
                out_k = str(k)
            except Exception:
                out_k = "<key>"
            out[out_k] = safe_json(v, max_depth, _depth + 1, _seen)
        return out
    if isinstance(data, (list, tuple)):
        return [safe_json(v, max_depth, _depth + 1, _seen) for v in data]
    try:
        json.dumps(data)
        return data
    except Exception:
        try:
            return str(data)
        except Exception:
            return "<unserializable>"

# -------------------- Technical helpers --------------------

def calc_ema(values: List[float], period: int) -> Optional[float]:
    if not values or len(values) < period:
        return None
    k = 2 / (period + 1)
    ema = sum(values[:period]) / period
    for price in values[period:]:
        ema = price * k + ema * (1 - k)
    return ema

def calc_rsi(closes: List[float], period: int = 14) -> Optional[float]:
    if not closes or len(closes) < period + 1:
        return None
    gains = []
    losses = []
    for i in range(1, period + 1):
        delta = closes[i] - closes[i - 1]
        if delta > 0:
            gains.append(delta)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(-delta)
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def detect_bullish_candle(candles: List[Dict[str, float]]) -> bool:
    # candles is list of dicts with keys: open, high, low, close
    if not candles:
        return False
    last = candles[-1]
    
    # bullish engulfing simple check
    if len(candles) >= 2:
        prev = candles[-2]
        if (prev['close'] < prev['open'] and 
            last['close'] > last['open'] and 
            last['close'] > prev['open'] and 
            last['open'] < prev['close']):
            return True
    
    # hammer check (small body, long lower wick)
    body = abs(last['close'] - last['open'])
    lower_wick = last['open'] - last['low'] if last['open'] > last['close'] else last['close'] - last['low']
    upper_wick = last['high'] - max(last['open'], last['close'])
    
    if body > 0 and lower_wick / body >= 2 and upper_wick / body <= 0.5:
        return True
    
    return False

def calc_fib_levels(high: float, low: float) -> Dict[str, float]:
    diff = high - low
    levels = {
        '0.0': high,
        '0.236': high - 0.236 * diff,
        '0.382': high - 0.382 * diff,
        '0.5': high - 0.5 * diff,
        '0.618': high - 0.618 * diff,
        '0.786': high - 0.786 * diff,
        '1.0': low,
        # extensions (relative to range)
        '1.272_ext': high + 1.272 * diff,
        '1.618_ext': high + 1.618 * diff,
        '2.0_ext': high + 2.0 * diff,
        '2.618_ext': high + 2.618 * diff,
    }
    return levels

def price_in_zone(price: float, levels: Dict[str, float], lo_key: str = '0.382', hi_key: str = '0.618') -> bool:
    try:
        lo = levels[lo_key]
        hi = levels[hi_key]
        return min(lo, hi) <= price <= max(lo, hi)
    except Exception:
        return False

# -------------------- Bot Controller --------------------

class BotController:
    def __init__(self, log_queue: Optional[threading.Queue] = None):
        self.log_queue = log_queue
        self._running = False
        self._stop_event = threading.Event()
        self._file_lock = threading.Lock()
        self._threads: List[threading.Thread] = []

        # Initialize files if they don't exist
        for path, default in ((ACCOUNTS_FILE, []), (TRADES_FILE, [])):
            if not os.path.exists(path):
                try:
                    with open(path, 'w') as f:
                        json.dump(default, f, indent=2)
                except Exception as e:
                    print(f"Error creating {path}: {e}")

    def is_running(self) -> bool:
        return bool(self._running and not self._stop_event.is_set())

    def log(self, msg: str):
        ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        line = f'[{ts}] {msg}'
        print(line)
        if self.log_queue:
            try:
                self.log_queue.put(line, block=False)
            except Exception:
                pass

    def load_accounts(self) -> List[Dict[str, Any]]:
        try:
            with open(ACCOUNTS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.log(f"Error loading accounts: {e}")
            return []

    def save_accounts(self, accounts: List[Dict[str, Any]]):
        with self._file_lock:
            try:
                with open(ACCOUNTS_FILE, 'w') as f:
                    json.dump(accounts, f, indent=2)
            except Exception as e:
                self.log(f"Error saving accounts: {e}")

    def _read_trades(self) -> List[Dict[str, Any]]:
        try:
            with open(TRADES_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.log(f"Error reading trades: {e}")
            return []

    def _write_trades(self, trades: List[Dict[str, Any]]):
        with self._file_lock:
            try:
                with open(TRADES_FILE, 'w') as f:
                    json.dump(trades, f, indent=2)
            except Exception as e:
                self.log(f"Error writing trades: {e}")

    def _sanitize_for_json(self, obj: Any) -> Any:
        try:
            return safe_json(obj, max_depth=6)
        except Exception:
            try:
                return json.loads(json.dumps(obj, default=str))
            except Exception:
                return str(obj)

    def add_trade(self, trade: Dict[str, Any]):
        with self._file_lock:
            trades = self._read_trades()
            trades.append(self._sanitize_for_json(trade))
            # Keep only last 500 trades to prevent file from growing too large
            if len(trades) > 500:
                trades = trades[-500:]
            self._write_trades(trades)

    def update_trade(self, trade_id: str, updates: Dict[str, Any]) -> bool:
        with self._file_lock:
            trades = self._read_trades()
            changed = False
            for t in trades:
                if t.get('id') == trade_id:
                    san = self._sanitize_for_json(updates)
                    if isinstance(san, dict):
                        t.update(san)
                    changed = True
                    break
            if changed:
                self._write_trades(trades)
            return changed

    def _get_client(self, account: Dict[str, Any]) -> Optional[HTTP]:
        try:
            key = account.get('api_key') or account.get('key') or account.get('apiKey')
            secret = account.get('api_secret') or account.get('secret') or account.get('apiSecret')
            if not key or not secret:
                return None
            client = HTTP(
                api_key=key, 
                api_secret=secret, 
                testnet=TRADE_SETTINGS.get('test_on_testnet', False)
            )
            return client
        except Exception as e:
            self.log(f"_get_client error: {e}")
            return None

    def _capture_preview(self, account: Dict[str, Any], resp: Any, label: str = 'resp'):
        try:
            if not TRADE_SETTINGS.get('debug_raw_responses', False):
                return
            preview = safe_json(resp, max_depth=3)
            account.setdefault('last_raw_preview', {})
            key = f"{label}-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
            account['last_raw_preview'][key] = preview
            # Keep only last 3 previews
            keys = list(account['last_raw_preview'].keys())
            if len(keys) > 3:
                for k in keys[:-3]:
                    account['last_raw_preview'].pop(k, None)
        except Exception:
            pass

    def _try_methods(self, client: HTTP, candidate_names: List[str], *args, **kwargs) -> Any:
        last_exc = None
        for name in candidate_names:
            try:
                meth = getattr(client, name, None)
                if not callable(meth):
                    continue
                try:
                    resp = meth(*args, **kwargs)
                except TypeError:
                    try:
                        resp = meth(**kwargs) if kwargs else meth(*args)
                    except Exception as e2:
                        last_exc = e2
                        continue
                except Exception as e:
                    last_exc = e
                    continue
                
                # Validate response
                try:
                    _ = json.dumps(resp, default=str)
                except Exception:
                    try:
                        resp = str(resp)[:1000]
                    except Exception:
                        resp = 'unserializable_response'
                
                # Check for API errors
                if isinstance(resp, dict):
                    for rc in ('ret_code', 'retCode', 'error_code', 'err_code'):
                        if rc in resp and resp.get(rc) not in (0, None, '0'):
                            last_exc = RuntimeError(f"API error from {name}: {resp.get(rc)}")
                            resp = None
                            break
                    if resp and resp.get('success') is False:
                        last_exc = RuntimeError(f"API reported failure from {name}")
                        resp = None
                
                if resp is not None:
                    return resp
            except Exception as e:
                last_exc = e
                continue
        
        if last_exc:
            raise last_exc
        raise RuntimeError('No candidate methods succeeded: ' + ','.join(candidate_names))

    def safe_get_ticker(self, client: HTTP, symbol: str) -> Any:
        candidates = [
            "ticker_price", "get_ticker", "get_symbol_ticker", 
            "latest_information_for_symbol", "tickers", "get_tickers", "get_ticker_price"
        ]
        try:
            return self._try_methods(client, candidates, symbol)
        except Exception:
            return self._try_methods(client, candidates, params={'symbol': symbol})

    def safe_get_klines(self, client: HTTP, symbol: str, interval: str = '1', limit: int = 200) -> Any:
        candidates = [
            "query_kline", "get_kline", "get_klines", 
            "query_candles", "get_candlesticks", "kline"
        ]
        try:
            return self._try_methods(client, candidates, symbol, interval, limit)
        except Exception:
            return self._try_methods(client, candidates, params={
                'symbol': symbol, 
                'interval': interval, 
                'limit': limit
            })

    def _parse_price(self, raw: Any, depth: int = 0, visited: Optional[set] = None) -> Optional[float]:
        if visited is None:
            visited = set()
        if depth > 8:
            return None
        
        try:
            rid = None
            try:
                rid = id(raw)
            except Exception:
                rid = None
            
            if rid is not None and rid in visited:
                return None
            if rid is not None:
                visited.add(rid)

            if raw is None:
                return None
            if isinstance(raw, (int, float)):
                return float(raw)
            if isinstance(raw, str):
                try:
                    return float(raw)
                except Exception:
                    return None
            
            if isinstance(raw, dict):
                # Try common price field names
                for k in ("price", "executed_price", "avgPrice", "last_price", "lastPrice", "close", "last", "p"):
                    if k in raw and raw[k] not in (None, ""):
                        try:
                            return float(raw[k])
                        except Exception:
                            pass
                
                # Try common result containers
                for key in ("result", "data", "list", "ticks", "rows", "items"):
                    if key in raw:
                        p = self._parse_price(raw[key], depth + 1, visited)
                        if p is not None:
                            return p
                
                # Try all values
                for v in raw.values():
                    p = self._parse_price(v, depth + 1, visited)
                    if p is not None:
                        return p
            
            if isinstance(raw, (list, tuple)) and raw:
                # Try OHLCV format [timestamp, open, high, low, close, volume]
                first = raw[0]
                if isinstance(first, (list, tuple)) and len(first) >= 5:
                    try:
                        return float(first[4])  # close price
                    except Exception:
                        pass
                
                # Try direct OHLCV
                if len(raw) >= 5 and all(isinstance(x, (int, float, str)) for x in raw[:5]):
                    try:
                        return float(raw[4])  # close price
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
            - upgrade to 2.618 if momentum >= momentum_very_strong_pct or price > EMA50*1.02
        """
        try:
            if not acct.get('monitoring'):
                return
            if acct.get('position') == 'open' or acct.get('open_trade_id'):
                return
            
            client = self._get_client(acct)
            if not client:
                self.log(f"No client for account {acct.get('name')}")
                acct.setdefault('last_validation_error', 'missing_client')
                return
            
            ok, bal, err = self.validate_account(acct)
            acct['validated'] = ok
            acct['balance'] = bal
            acct['last_validation_error'] = err
            if not ok or not bal:
                return
            
            allocation_pct = TRADE_SETTINGS.get('trade_allocation_pct', 0.01)
            usd_alloc = float(bal) * float(allocation_pct)
            if usd_alloc < TRADE_SETTINGS.get('min_trade_amount', 5.0):
                self.log(f"Computed allocation ${usd_alloc:.2f} below min; skipping")
                return

            # score symbols
            scores: List[Tuple[str, int, Dict[str, Any]]] = []
            for symbol in ALLOWED_COINS:
                try:
                    sc, diag = self.score_symbol(client, symbol)
                    if sc > 0:
                        scores.append((symbol, sc, diag))
                except Exception as e:
                    self.log(f"Scoring {symbol} failed: {e}")
                    continue

            if not scores:
                self.log("No symbols scored positive this scan.")
                return

            # pick best symbol(s) — default pick 1
            scores.sort(key=lambda x: x[1], reverse=True)
            best_symbol, best_score, best_diag = scores[0]
            self.log(f"Top symbol {best_symbol} score={best_score} diag={best_diag}")

            # compute qty based on current price
            price = best_diag.get('current_price')
            if not price or price <= 0:
                self.log(f"Invalid price for {best_symbol}; skip")
                return
            
            notional = usd_alloc
            if notional < TRADE_SETTINGS.get('min_trade_amount', 5.0):
                self.log(f"Notional ${notional:.2f} below min; skip")
                return
            
            qty = round(notional / price, 6)
            if qty <= 0:
                self.log(f"Computed qty <= 0 for {best_symbol}; skip")
                return

            # compute SL (fib 78.6) and dynamic TP (1.272 base -> 1.618 or 2.618 if strong)
            fib_levels = best_diag.get('fib_levels', {})
            sl_price = fib_levels.get('0.786') if fib_levels else None
            tp_1_272 = fib_levels.get('1.272_ext') if fib_levels else None
            tp_1_618 = fib_levels.get('1.618_ext') if fib_levels else None
            tp_2_618 = fib_levels.get('2.618_ext') if fib_levels else None

            # fallback stoploss: -RISK_RULES['stop_loss'] percent
            if sl_price is None:
                sl_price = price * (1 + (RISK_RULES['stop_loss'] / 100.0))

            # base TP
            tp_price = tp_1_272 if tp_1_272 is not None else price * 1.01272

            # dynamic upgrades based on momentum or EMA bias
            momentum_pct = best_diag.get('momentum_pct', 0.0)
            ema50 = best_diag.get('ema50')
            
            # upgrade to 1.618 if momentum strong or price clear above EMA50
            try:
                if (momentum_pct >= SCORE_SETTINGS['momentum_strong_pct']) or (ema50 and price > ema50 * 1.01):
                    if tp_1_618 is not None:
                        tp_price = tp_1_618
                # upgrade to 2.618 if very strong
                if (momentum_pct >= SCORE_SETTINGS['momentum_very_strong_pct']) or (ema50 and price > ema50 * 1.02):
                    if tp_2_618 is not None:
                        tp_price = tp_2_618
            except Exception:
                pass

            resp = self._place_market_order(client, best_symbol, 'Buy', qty, price_hint=price)
            simulated = bool(resp.get('simulated')) if isinstance(resp, dict) else True

            entry_price = None
            if isinstance(resp, dict):
                for k in ('executed_price', 'avgPrice', 'price', 'last_price', 'lastPrice'):
                    if k in resp and resp[k]:
                        try:
                            entry_price = float(resp[k])
                            break
                        except Exception:
                            continue
            if entry_price is None:
                entry_price = price

            ts = now_ts()
            trade_id = self._record_trade_entry(acct, best_symbol, qty, entry_price, ts, simulated, sl_price, tp_price)
            acct['position'] = 'open'
            acct['current_symbol'] = best_symbol
            acct['entry_price'] = entry_price
            acct['entry_qty'] = qty
            acct['entry_time'] = ts
            acct['open_trade_id'] = trade_id
            acct['buy_price'] = entry_price
            acct['stop_loss_price'] = sl_price
            acct['take_profit_price'] = tp_price
            acct['score'] = best_score
            self.log(f"Opened trade {trade_id} {best_symbol} qty={qty} entry={entry_price} SL={sl_price} TP={tp_price} simulated={simulated}")

        except Exception as e:
            self.log(f"attempt_trade_for_account error: {e}")

    def _check_open_position(self, acct: Dict[str, Any]):
        try:
            trade_id = acct.get('open_trade_id')
            if not acct.get('position') == 'open' or not trade_id:
                return
            
            client = self._get_client(acct)
            if not client:
                self.log(f"No client to check position for {acct.get('name')}")
                return
            
            symbol = acct.get('current_symbol')
            if not symbol:
                return
            
            entry_price = acct.get('entry_price')
            qty = acct.get('entry_qty')
            entry_ts = acct.get('entry_time') or now_ts()

            tick = None
            try:
                tick = self.safe_get_ticker(client, symbol)
                self._capture_preview(acct, tick, label='ticker_check')
            except Exception as e:
                self.log(f"Ticker for check failed: {e}")
                return
            
            price = self._parse_price(tick)
            if price is None:
                self.log(f"Could not parse current price for {symbol}")
                return

            elapsed_s = max(0, now_ts() - int(entry_ts))
            label = None
            should_close = False

            # check TP/SL based on recorded fib levels
            tp_price = acct.get('take_profit_price')
            sl_price = acct.get('stop_loss_price')

            # stop-loss (price fell to or below SL)
            if sl_price is not None and price <= sl_price:
                label = 'stop_loss'
                should_close = True

            # take-profit (price reached or exceeded TP)
            if not should_close and tp_price is not None and price >= tp_price:
                label = 'fib_take_profit'
                should_close = True

            # forced exit safety (very long hold)
            if not should_close and elapsed_s > RISK_RULES.get('max_hold', 24 * 60 * 60):
                label = 'forced_exit'
                should_close = True

            if should_close:
                resp = self._place_market_order(client, symbol, 'Sell', qty, price_hint=price)
                simulated = bool(resp.get('simulated')) if isinstance(resp, dict) else True
                
                exit_price = None
                if isinstance(resp, dict):
                    for k in ('executed_price', 'avgPrice', 'price', 'last_price', 'lastPrice'):
                        if k in resp and resp[k] is not None:
                            try:
                                exit_price = float(resp[k])
                                break
                            except Exception:
                                continue
                if exit_price is None:
                    exit_price = price
                
                exit_ts = now_ts()
                try:
                    final_profit = ((float(exit_price) - float(entry_price)) / float(entry_price)) * 100.0
                except Exception:
                    final_profit = None
                
                resp_summary = safe_json(resp) if isinstance(resp, (dict, list)) else str(resp)

                self._finalize_trade(trade_id, exit_price, exit_ts, label, resp_summary, simulated)

                acct['position'] = 'closed'
                acct.pop('entry_price', None)
                acct.pop('entry_qty', None)
                acct.pop('entry_time', None)
                acct.pop('open_trade_id', None)
                acct['current_symbol'] = None
                acct['buy_price'] = None
                acct.pop('stop_loss_price', None)
                acct.pop('take_profit_price', None)
                self.log(f"Closed trade {trade_id} for {acct.get('name')} label={label} exit_price={exit_price} profit_pct={final_profit} elapsed={fmt_elapsed(exit_ts - entry_ts)} simulated={simulated}")

        except Exception as e:
            self.log(f"_check_open_position error for acct {acct.get('id')}: {e}")

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
