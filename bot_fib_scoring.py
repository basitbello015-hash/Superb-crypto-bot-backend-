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

# Load configuration from config.json
def load_config():
    config_file = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            return json.load(f)
    return {}

# Initial config load
CONFIG = load_config()

ALLOWED_COINS = [
    "ADAUSDT", "XRPUSDT", "TRXUSDT", "DOGEUSDT", "CHZUSDT", "VETUSDT", "BTTUSDT", "HOTUSDT",
    "XLMUSDT", "ZILUSDT", "IOTAUSDT", "SCUSDT", "DENTUSDT", "KEYUSDT", "WINUSDT", "CVCUSDT",
    "MTLUSDT", "CELRUSDT", "FUNUSDT", "STMXUSDT", "REEFUSDT", "ANKRUSDT", "ONEUSDT", "OGNUSDT",
    "CTSIUSDT", "DGBUSDT", "CKBUSDT", "ARPAUSDT", "MBLUSDT", "TROYUSDT", "PERLUSDT", "DOCKUSDT",
    "RENUSDT", "COTIUSDT", "MDTUSDT", "OXTUSDT", "PHAUSDT", "BANDUSDT", "GTOUSDT", "LOOMUSDT",
    "PONDUSDT", "FETUSDT", "SYSUSDT", "TLMUSDT", "NKNUSDT", "LINAUSDT", "ORNUSDT", "COSUSDT",
    "FLMUSDT", "ALICEUSDT",
]

# Risk rules with config override
RISK_RULES = {
    "stop_loss": CONFIG.get("stopLoss", -1.0),
    "max_hold": CONFIG.get("maxHold", 24 * 60 * 60),
}

SCORE_SETTINGS = {
    "momentum_scale": CONFIG.get("momentumScale", 1.0),
    "rsi_period": CONFIG.get("rsiPeriod", 14),
    "rsi_oversold_threshold": CONFIG.get("rsiOversold", 35),
    "rsi_overbought_threshold": CONFIG.get("rsiOverbought", 65),
    "momentum_entry_threshold_pct": CONFIG.get("momentumEntryThreshold", 0.1),
    "momentum_strong_pct": CONFIG.get("momentumStrong", 0.5),
    "momentum_very_strong_pct": CONFIG.get("momentumVeryStrong", 1.5),
    "max_price_allowed": CONFIG.get("maxPriceAllowed", 1.2),
    "score_weights": {
        "rsi": CONFIG.get("scoreWeightRsi", 1),
        "momentum": CONFIG.get("scoreWeightMomentum", 1),
        "ema": CONFIG.get("scoreWeightEma", 1),
        "candle": CONFIG.get("scoreWeightCandle", 1),
        "fib_zone": CONFIG.get("scoreWeightFibZone", 1),
    }
}

TRADE_SETTINGS = {
    "trade_allocation_pct": CONFIG.get("tradeAllocation", 0.01),
    "min_trade_amount": CONFIG.get("minTradeAmount", 5.0),
    "use_market_order": CONFIG.get("useMarketOrder", True),
    "test_on_testnet": CONFIG.get("testOnTestnet", False),
    "scan_interval": CONFIG.get("scanInterval", 10),
    "debug_raw_responses": CONFIG.get("debugRawResponses", False),
    "dry_run": CONFIG.get("dryRun", True),
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ACCOUNTS_FILE = os.path.join(BASE_DIR, "app/accounts.json")
TRADES_FILE = os.path.join(BASE_DIR, "app/trades.json")

# Ensure app directory exists
os.makedirs(os.path.join(BASE_DIR, "app"), exist_ok=True)

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
    if not candles:
        return False
    last = candles[-1]

    if len(candles) >= 2:
        prev = candles[-2]
        if (prev['close'] < prev['open'] and 
            last['close'] > last['open'] and 
            last['close'] > prev['open'] and 
            last['open'] < prev['close']):
            return True

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

        for path, default in ((ACCOUNTS_FILE, []), (TRADES_FILE, [])):
            if not os.path.exists(path):
                try:
                    with open(path, 'w') as f:
                        json.dump(default, f, indent=2)
                except Exception as e:
                    print(f"Error creating {path}: {e}")

    def reload_config(self):
        """Reload configuration from config.json"""
        global CONFIG, RISK_RULES, SCORE_SETTINGS, TRADE_SETTINGS
        CONFIG = load_config()
        
        RISK_RULES.update({
            "stop_loss": CONFIG.get("stopLoss", -1.0),
            "max_hold": CONFIG.get("maxHold", 24 * 60 * 60),
        })
        
        SCORE_SETTINGS.update({
            "momentum_scale": CONFIG.get("momentumScale", 1.0),
            "rsi_period": CONFIG.get("rsiPeriod", 14),
            "rsi_oversold_threshold": CONFIG.get("rsiOversold", 35),
            "rsi_overbought_threshold": CONFIG.get("rsiOverbought", 65),
            "momentum_entry_threshold_pct": CONFIG.get("momentumEntryThreshold", 0.1),
            "momentum_strong_pct": CONFIG.get("momentumStrong", 0.5),
            "momentum_very_strong_pct": CONFIG.get("momentumVeryStrong", 1.5),
            "max_price_allowed": CONFIG.get("maxPriceAllowed", 1.2),
        })
        
        SCORE_SETTINGS["score_weights"].update({
            "rsi": CONFIG.get("scoreWeightRsi", 1),
            "momentum": CONFIG.get("scoreWeightMomentum", 1),
            "ema": CONFIG.get("scoreWeightEma", 1),
            "candle": CONFIG.get("scoreWeightCandle", 1),
            "fib_zone": CONFIG.get("scoreWeightFibZone", 1),
        })
        
        TRADE_SETTINGS.update({
            "trade_allocation_pct": CONFIG.get("tradeAllocation", 0.01),
            "min_trade_amount": CONFIG.get("minTradeAmount", 5.0),
            "use_market_order": CONFIG.get("useMarketOrder", True),
            "test_on_testnet": CONFIG.get("testOnTestnet", False),
            "scan_interval": CONFIG.get("scanInterval", 10),
            "debug_raw_responses": CONFIG.get("debugRawResponses", False),
            "dry_run": CONFIG.get("dryRun", True),
        })
        
        self.log("Configuration reloaded")

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

                try:
                    _ = json.dumps(resp, default=str)
                except Exception:
                    try:
                        resp = str(resp)[:1000]
                    except Exception:
                        resp = 'unserializable_response'

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
                for k in ("price", "executed_price", "avgPrice", "last_price", "lastPrice", "close", "last", "p"):
                    if k in raw and raw[k] not in (None, ""):
                        try:
                            return float(raw[k])
                        except Exception:
                            pass

                for key in ("result", "data", "list", "ticks", "rows", "items"):
                    if key in raw:
                        p = self._parse_price(raw[key], depth + 1, visited)
                        if p is not None:
                            return p

                for v in raw.values():
                    p = self._parse_price(v, depth + 1, visited)
                    if p is not None:
                        return p

            if isinstance(raw, (list, tuple)) and raw:
                first = raw[0]
                if isinstance(first, (list, tuple)) and len(first) >= 5:
                    try:
                        return float(first[4])
                    except Exception:
                        pass

                if len(raw) >= 5 and all(isinstance(x, (int, float, str)) for x in raw[:5]):
                    try:
                        return float(raw[4])
                    except Exception:
                        pass

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
                for k in ("total_equity", "equity", "totalBalance", "total_balance"):
                    if k in payload:
                        try:
                            return float(payload[k])
                        except Exception:
                            pass

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

    def score_symbol(self, client: HTTP, symbol: str) -> Tuple[int, Dict[str, Any]]:
        diagnostics: Dict[str, Any] = {}
        try:
            raw_klines = self.safe_get_klines(client, symbol, interval='1', limit=200)
            self._capture_preview({}, raw_klines, label='klines')
        except Exception as e:
            return 0, {"error": f"klines_fetch_failed: {e}"}

        closes: List[float] = []
        ohlc: List[Dict[str, float]] = []
        try:
            if isinstance(raw_klines, dict) and 'result' in raw_klines:
                payload = raw_klines['result']
            elif isinstance(raw_klines, dict) and 'data' in raw_klines:
                payload = raw_klines['data']
            else:
                payload = raw_klines

            if isinstance(payload, list):
                for item in payload:
                    if isinstance(item, (list, tuple)) and len(item) >= 5:
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

        rsi = calc_rsi(closes[-(SCORE_SETTINGS['rsi_period'] + 1) :])
        ema50 = calc_ema(closes[-100:], 50)
        ema200 = calc_ema(closes[-220:], 200)
        momentum_pct = ((closes[-1] - closes[-5]) / closes[-5]) * 100.0 if len(closes) >= 6 and closes[-5] != 0 else 0.0
        candle_ok = detect_bullish_candle(ohlc[-5:])

        window = closes[-50:]
        swing_high = max(window)
        swing_low = min(window)
        fib = calc_fib_levels(swing_high, swing_low)
        current_price = closes[-1]

        score = 0
        breakdown = {}

        if rsi is not None:
            breakdown['rsi'] = rsi
            if rsi <= SCORE_SETTINGS['rsi_oversold_threshold']:
                score += SCORE_SETTINGS['score_weights']['rsi']
        else:
            breakdown['rsi'] = None

        breakdown['momentum_pct'] = momentum_pct
        if abs(momentum_pct) >= SCORE_SETTINGS['momentum_entry_threshold_pct']:
            if momentum_pct > 0:
                score += SCORE_SETTINGS['score_weights']['momentum']

        breakdown['ema50'] = ema50
        breakdown['ema200'] = ema200
        if ema50 is not None and ema200 is not None:
            if current_price >= ema50 and ema50 >= ema200:
                score += SCORE_SETTINGS['score_weights']['ema']

        breakdown['candle_ok'] = candle_ok
        if candle_ok:
            score += SCORE_SETTINGS['score_weights']['candle']

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

            scores.sort(key=lambda x: x[1], reverse=True)
            best_symbol, best_score, best_diag = scores[0]
            self.log(f"Top symbol {best_symbol} score={best_score} diag={best_diag}")

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

            fib_levels = best_diag.get('fib_levels', {})
            sl_price = fib_levels.get('0.786') if fib_levels else None
            tp_1_272 = fib_levels.get('1.272_ext') if fib_levels else None
            tp_1_618 = fib_levels.get('1.618_ext') if fib_levels else None
            tp_2_618 = fib_levels.get('2.618_ext') if fib_levels else None

            if sl_price is None:
                sl_price = price * (1 + (RISK_RULES['stop_loss'] / 100.0))

            tp_price = tp_1_272 if tp_1_272 is not None else price * 1.01272

            momentum_pct = best_diag.get('momentum_pct', 0.0)
            ema50 = best_diag.get('ema50')

            try:
                if (momentum_pct >= SCORE_SETTINGS['momentum_strong_pct']) or (ema50 and price > ema50 * 1.01):
                    if tp_1_618 is not None:
                        tp_price = tp_1_618
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

            tp_price = acct.get('take_profit_price')
            sl_price = acct.get('stop_loss_price')

            if sl_price is not None and price <= sl_price:
                label = 'stop_loss'
                should_close = True

            if not should_close and tp_price is not None and price >= tp_price:
                label = 'fib_take_profit'
                should_close = True

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

    # -----------------------------
    # NEW HELPERS (ADDED & MERGED)
    # - should_enter_trade: modular entry check (not yet wired into attempt_trade_for_account)
    # - should_exit_trade: modular exit check (not yet used by _check_open_position)
    # These are safe, non-invasive additions; you can switch to them if you want later.
    # -----------------------------
    def should_enter_trade(self, close_prices: List[float], candles: List[Dict[str, float]]) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate RSI, momentum, candle structure, EMA, Fib zone, etc."""
        debug = {}

        # Not enough data
        if len(close_prices) < 20 or len(candles) < 20:
            return False, {"reason": "insufficient_data"}

        latest_price = close_prices[-1]

        # RSI
        rsi = calc_rsi(close_prices, SCORE_SETTINGS["rsi_period"])
        debug["rsi"] = rsi
        if rsi is None or rsi > SCORE_SETTINGS["rsi_oversold_threshold"]:
            return False, {"reason": "rsi_not_low", "rsi": rsi}

        # Momentum % change from previous candle
        momentum_pct = ((close_prices[-1] - close_prices[-2]) / close_prices[-2]) * 100
        debug["momentum_pct"] = momentum_pct
        if momentum_pct < SCORE_SETTINGS["momentum_entry_threshold_pct"]:
            return False, {"reason": "momentum_not_strong", "momentum": momentum_pct}

        # Bullish candle pattern
        bullish = detect_bullish_candle(candles)
        debug["bullish_candle"] = bullish
        if not bullish:
            return False, {"reason": "no_bullish_candle"}

        # EMA check
        ema = calc_ema(close_prices, 14)
        debug["ema"] = ema
        if ema is None or latest_price < ema:
            return False, {"reason": "below_ema"}

        # Fib zone check
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        fib = calc_fib_levels(max(highs[-20:]), min(lows[-20:]))
        debug["fib_levels"] = fib

        if not price_in_zone(latest_price, fib, "0.382", "0.618"):
            return False, {"reason": "not_in_fib_zone"}

        # ✅ ENTRY PASS
        return True, {
            "rsi": rsi,
            "momentum_pct": momentum_pct,
            "bullish_candle": bullish,
            "ema": ema,
            "fib_levels": fib,
        }

    def should_exit_trade(self, trade: Dict[str, Any], current_price: float, ts: int) -> Optional[str]:
        """Check SL, TP, max hold time. Returns a label if should exit."""
        if not trade.get("open"):
            return None

        entry = trade.get("entry_price")
        sl = trade.get("stop_loss_price")
        tp = trade.get("take_profit_price")
        entry_ts = None
        try:
            entry_ts = int(datetime.fromisoformat(trade["entry_time"]).timestamp())
        except Exception:
            entry_ts = trade.get("entry_time") or now_ts()

        # Stop loss
        if sl and current_price <= sl:
            return "stop_loss_hit"

        # Take profit
        if tp and current_price >= tp:
            return "take_profit_hit"

        # Max hold time
        try:
            if ts - int(entry_ts) >= RISK_RULES["max_hold"]:
                return "max_hold_expired"
        except Exception:
            pass

        return None

    # -----------------------------
    # Optional per-account threaded processor (NOT wired by default)
    # If you want to switch to this model, replace _scan_once/start logic to spawn per-account threads
    # -----------------------------
    def _process_account(self, account: Dict[str, Any]):
        """
        Alternate per-account loop that manages open trades and scans symbols per-account.
        Not used by default — kept as an option.
        """
        client = self._get_client(account)
        if not client:
            self.log(f"Skipping account (invalid keys): {account.get('name')}")
            return

        # Validate balance
        ok, balance, msg = self.validate_account(account)
        if not ok:
            self.log(f"Account {account.get('name')} failed validation: {msg}")
            return

        allocation = balance * TRADE_SETTINGS["trade_allocation_pct"]
        allocation = max(allocation, TRADE_SETTINGS["min_trade_amount"])

        while self.is_running() and not self._stop_event.is_set():
            ts = now_ts()

            # Load open trades for this account
            open_trades = [t for t in self._read_trades() if t.get("open") and t.get("account_id") == account.get("id")]

            # Manage open trades
            for t in open_trades:
                symbol = t["symbol"]
                try:
                    ticker = self.safe_get_ticker(client, symbol)
                    cur_price = self._parse_price(ticker)
                except Exception:
                    cur_price = None

                if cur_price is None:
                    continue

                reason = self.should_exit_trade(t, cur_price, ts)
                if reason:
                    # use existing finalize signature: label, resp_summary, simulated
                    resp = self._place_market_order(client, symbol, 'Sell', t.get('qty', 0), price_hint=cur_price)
                    simulated = bool(resp.get('simulated')) if isinstance(resp, dict) else True
                    resp_summary = safe_json(resp) if isinstance(resp, (dict, list)) else str(resp)
                    self._finalize_trade(t['id'], cur_price, ts, reason, resp_summary, simulated)

            # Scan symbols for entries (simple implementation)
            for symbol in ALLOWED_COINS:
                try:
                    raw_klines = self.safe_get_klines(client, symbol, interval='1', limit=200)
                    payload = None
                    if isinstance(raw_klines, dict) and 'result' in raw_klines:
                        payload = raw_klines['result']
                    elif isinstance(raw_klines, dict) and 'data' in raw_klines:
                        payload = raw_klines['data']
                    else:
                        payload = raw_klines

                    candles: List[Dict[str, float]] = []
                    closes: List[float] = []

                    if isinstance(payload, list):
                        for item in payload:
                            if isinstance(item, (list, tuple)) and len(item) >= 5:
                                o = float(item[1]); h = float(item[2]); l = float(item[3]); c = float(item[4])
                            elif isinstance(item, dict):
                                o = float(item.get('open') or item.get('Open') or item.get('o'))
                                h = float(item.get('high') or item.get('High') or item.get('h'))
                                l = float(item.get('low') or item.get('Low') or item.get('l'))
                                c = float(item.get('close') or item.get('Close') or item.get('c'))
                            else:
                                continue
                            candles.append({'open': o, 'high': h, 'low': l, 'close': c})
                            closes.append(c)
                    else:
                        continue

                    should_enter, info = self.should_enter_trade(closes, candles)
                    if not should_enter:
                        continue

                    entry_price = closes[-1]
                    qty = round(allocation / entry_price, 6)
                    if qty <= 0:
                        continue

                    # compute sl/tp
                    fib_levels = info.get('fib_levels', {})
                    sl_price = fib_levels.get('0.786') if fib_levels else entry_price * (1 + (RISK_RULES['stop_loss'] / 100.0))
                    tp_price = fib_levels.get('1.272_ext') if fib_levels else entry_price * 1.01272

                    resp = self._place_market_order(client, symbol, 'Buy', qty, price_hint=entry_price)
                    simulated = bool(resp.get('simulated')) if isinstance(resp, dict) else True

                    ts_now = now_ts()
                    tid = self._record_trade_entry(account, symbol, qty, entry_price, ts_now, simulated, sl_price, tp_price)
                    account['position'] = 'open'
                    account['current_symbol'] = symbol
                    account['open_trade_id'] = tid
                    self.log(f"[_process_account] Entered {symbol} qty={qty} entry={entry_price} trade_id={tid} simulated={simulated}")
                except Exception as e:
                    self.log(f"_process_account symbol scan error {symbol}: {e}")
                    continue

            time.sleep(int(TRADE_SETTINGS.get('scan_interval', 10)))

    # -----------------------------
    # START / STOP (kept original implementation)
    # -----------------------------
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
                self.reload_config()
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
