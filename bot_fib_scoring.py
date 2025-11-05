from __future__ import annotations

import json
import os
import threading
import time
import uuid
import sqlite3
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

APP_DIR = os.path.join(BASE_DIR, "app")
os.makedirs(APP_DIR, exist_ok=True)

ACCOUNTS_FILE = os.path.join(APP_DIR, "accounts.json")
TRADES_FILE = os.path.join(APP_DIR, "trades.json")
TRADES_DB = os.path.join(APP_DIR, "trades.db")
ACCOUNTS_DB = os.path.join(APP_DIR, "accounts.db")

# Ensure files exist (JSON fallback)
for f in [ACCOUNTS_FILE, TRADES_FILE]:
    if not os.path.exists(f):
        with open(f, "w") as fp:
            fp.write("[]")

# You can set environment variable USE_SQLITE=1 to switch to sqlite storage for trades/accounts
USE_SQLITE = os.environ.get("USE_SQLITE", "0") == "1"

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

# -------------------- Storage / Trade History Service --------------------

class StorageError(Exception):
    pass

class TradeStorage:
    """
    Unified trades storage. Uses SQLite when USE_SQLITE is True,
    otherwise falls back to JSON file backends (trades.json).
    Provides CRUD used by routes or services.
    """
    def __init__(self):
        self.use_sqlite = USE_SQLITE
        if self.use_sqlite:
            self._init_sqlite()
        else:
            # ensure JSON files exist
            if not os.path.exists(TRADES_FILE):
                with open(TRADES_FILE, "w") as f:
                    f.write("[]")
        self._file_lock = threading.Lock()

    # ---------- SQLite helpers ----------
    def _init_sqlite(self):
        conn = sqlite3.connect(TRADES_DB)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                created_at TEXT,
                updated_at TEXT,
                account_id TEXT,
                symbol TEXT,
                side TEXT,
                qty REAL,
                price REAL,
                status TEXT,
                meta TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _sqlite_conn(self):
        return sqlite3.connect(TRADES_DB, timeout=5)

    # ---------- JSON helpers ----------
    def _read_json_trades(self) -> List[Dict[str, Any]]:
        with self._file_lock:
            try:
                with open(TRADES_FILE, "r") as f:
                    return json.load(f)
            except Exception:
                return []

    def _write_json_trades(self, trades: List[Dict[str, Any]]):
        with self._file_lock:
            with open(TRADES_FILE, "w") as f:
                json.dump(trades, f, indent=2)

    # ---------- CRUD API ----------
    def list_trades(self, limit: int = 200) -> List[Dict[str, Any]]:
        if self.use_sqlite:
            conn = self._sqlite_conn()
            c = conn.cursor()
            rows = c.execute("SELECT id, created_at, updated_at, account_id, symbol, side, qty, price, status, meta FROM trades ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
            conn.close()
            out = []
            for r in rows:
                meta = {}
                try:
                    meta = json.loads(r[9]) if r[9] else {}
                except Exception:
                    meta = {"raw_meta": r[9]}
                out.append({
                    "id": r[0],
                    "created_at": r[1],
                    "updated_at": r[2],
                    "account_id": r[3],
                    "symbol": r[4],
                    "side": r[5],
                    "qty": r[6],
                    "price": r[7],
                    "status": r[8],
                    "meta": meta
                })
            return out
        else:
            trades = self._read_json_trades()
            return trades[-limit:][::-1]

    def get_trade(self, trade_id: str) -> Optional[Dict[str, Any]]:
        if self.use_sqlite:
            conn = self._sqlite_conn()
            c = conn.cursor()
            r = c.execute("SELECT id, created_at, updated_at, account_id, symbol, side, qty, price, status, meta FROM trades WHERE id = ?", (trade_id,)).fetchone()
            conn.close()
            if not r:
                return None
            meta = {}
            try:
                meta = json.loads(r[9]) if r[9] else {}
            except Exception:
                meta = {"raw_meta": r[9]}
            return {
                "id": r[0],
                "created_at": r[1],
                "updated_at": r[2],
                "account_id": r[3],
                "symbol": r[4],
                "side": r[5],
                "qty": r[6],
                "price": r[7],
                "status": r[8],
                "meta": meta
            }
        else:
            trades = self._read_json_trades()
            for t in trades:
                if t.get("id") == trade_id:
                    return t
            return None

    def create_trade(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        trade = dict(trade)  # copy
        if not trade.get("id"):
            trade["id"] = str(uuid.uuid4())
        now = now_iso()
        trade.setdefault("created_at", now)
        trade["updated_at"] = now
        trade = safe_json(trade, max_depth=6)

        if self.use_sqlite:
            conn = self._sqlite_conn()
            c = conn.cursor()
            meta_str = json.dumps(trade.get("meta", {}), default=str)
            c.execute("""
                INSERT OR REPLACE INTO trades (id, created_at, updated_at, account_id, symbol, side, qty, price, status, meta)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade["id"], trade["created_at"], trade["updated_at"], trade.get("account_id"),
                trade.get("symbol"), trade.get("side"), trade.get("qty"), trade.get("price"),
                trade.get("status"), meta_str
            ))
            conn.commit()
            conn.close()
            return trade
        else:
            with self._file_lock:
                trades = self._read_json_trades()
                trades.append(trade)
                # keep last 500
                if len(trades) > 500:
                    trades = trades[-500:]
                self._write_json_trades(trades)
            return trade

    def update_trade(self, trade_id: str, updates: Dict[str, Any]) -> bool:
        now = now_iso()
        updates = safe_json(updates, max_depth=6)
        if self.use_sqlite:
            # load existing, merge meta if present
            existing = self.get_trade(trade_id)
            if not existing:
                return False
            # merge meta dictionaries
            meta = existing.get("meta", {})
            if isinstance(updates.get("meta"), dict):
                meta.update(updates.get("meta"))
            meta_str = json.dumps(meta, default=str)
            fields = {
                "updated_at": now,
                "account_id": updates.get("account_id", existing.get("account_id")),
                "symbol": updates.get("symbol", existing.get("symbol")),
                "side": updates.get("side", existing.get("side")),
                "qty": updates.get("qty", existing.get("qty")),
                "price": updates.get("price", existing.get("price")),
                "status": updates.get("status", existing.get("status")),
                "meta": meta_str
            }
            conn = self._sqlite_conn()
            c = conn.cursor()
            c.execute("""
                UPDATE trades SET updated_at=?, account_id=?, symbol=?, side=?, qty=?, price=?, status=?, meta=?
                WHERE id=?
            """, (fields["updated_at"], fields["account_id"], fields["symbol"], fields["side"], fields["qty"], fields["price"], fields["status"], fields["meta"], trade_id))
            conn.commit()
            changed = c.rowcount > 0
            conn.close()
            return changed
        else:
            with self._file_lock:
                trades = self._read_json_trades()
                changed = False
                for t in trades:
                    if t.get("id") == trade_id:
                        t.update(updates)
                        t["updated_at"] = now
                        changed = True
                        break
                if changed:
                    self._write_json_trades(trades)
                return changed

    def delete_trade(self, trade_id: str) -> bool:
        if self.use_sqlite:
            conn = self._sqlite_conn()
            c = conn.cursor()
            c.execute("DELETE FROM trades WHERE id = ?", (trade_id,))
            conn.commit()
            changed = c.rowcount > 0
            conn.close()
            return changed
        else:
            with self._file_lock:
                trades = self._read_json_trades()
                new = [t for t in trades if t.get("id") != trade_id]
                if len(new) != len(trades):
                    self._write_json_trades(new)
                    return True
                return False

# -------------------- Bot Controller --------------------

class BotController:
    def __init__(self, log_queue: Optional[threading.Queue] = None):
        self.log_queue = log_queue
        self._running = False
        self._stop_event = threading.Event()
        self._file_lock = threading.Lock()
        self._threads: List[threading.Thread] = []
        # storage instance for trades/history
        self.storage = TradeStorage()

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
            if USE_SQLITE:
                conn = sqlite3.connect(ACCOUNTS_DB)
                c = conn.cursor()
                c.execute("""
                    CREATE TABLE IF NOT EXISTS accounts (
                        id TEXT PRIMARY KEY,
                        label TEXT,
                        api_key TEXT,
                        api_secret TEXT,
                        meta TEXT
                    )
                """)
                rows = c.execute("SELECT id, label, api_key, api_secret, meta FROM accounts").fetchall()
                conn.close()
                out = []
                for r in rows:
                    meta = {}
                    try:
                        meta = json.loads(r[4]) if r[4] else {}
                    except Exception:
                        meta = {"raw_meta": r[4]}
                    out.append({
                        "id": r[0],
                        "label": r[1],
                        "api_key": r[2],
                        "api_secret": r[3],
                        "meta": meta
                    })
                return out
            else:
                with open(ACCOUNTS_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.log(f"Error loading accounts: {e}")
            return []

    def save_accounts(self, accounts: List[Dict[str, Any]]):
        with self._file_lock:
            try:
                if USE_SQLITE:
                    conn = sqlite3.connect(ACCOUNTS_DB)
                    c = conn.cursor()
                    c.execute("""
                        CREATE TABLE IF NOT EXISTS accounts (
                            id TEXT PRIMARY KEY,
                            label TEXT,
                            api_key TEXT,
                            api_secret TEXT,
                            meta TEXT
                        )
                    """)
                    for a in accounts:
                        aid = a.get("id") or str(uuid.uuid4())
                        c.execute("""
                            INSERT OR REPLACE INTO accounts (id, label, api_key, api_secret, meta)
                            VALUES (?, ?, ?, ?, ?)
                        """, (aid, a.get("label"), a.get("api_key"), a.get("api_secret"), json.dumps(a.get("meta", {}), default=str)))
                    conn.commit()
                    conn.close()
                else:
                    with open(ACCOUNTS_FILE, 'w') as f:
                        json.dump(accounts, f, indent=2)
            except Exception as e:
                self.log(f"Error saving accounts: {e}")

    def _read_trades(self) -> List[Dict[str, Any]]:
        # convenience wrapper to keep compatibility with older code
        return self.storage.list_trades(limit=500)

    def _write_trades(self, trades: List[Dict[str, Any]]):
        # not used directly when using storage, but keep for compatibility
        if USE_SQLITE:
            # naive upsert: replace all - for small volumes this is acceptable; adapt if needed
            conn = sqlite3.connect(TRADES_DB)
            c = conn.cursor()
            for t in trades:
                c.execute("""
                    INSERT OR REPLACE INTO trades (id, created_at, updated_at, account_id, symbol, side, qty, price, status, meta)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    t.get("id"), t.get("created_at"), t.get("updated_at"), t.get("account_id"),
                    t.get("symbol"), t.get("side"), t.get("qty"), t.get("price"),
                    t.get("status"), json.dumps(t.get("meta", {}), default=str)
                ))
            conn.commit()
            conn.close()
        else:
            with open(TRADES_FILE, 'w') as f:
                json.dump(trades, f, indent=2)

    def _sanitize_for_json(self, obj: Any) -> Any:
        try:
            return safe_json(obj, max_depth=6)
        except Exception:
            try:
                return json.loads(json.dumps(obj, default=str))
            except Exception:
                return str(obj)

    def add_trade(self, trade: Dict[str, Any]):
        try:
            created = self.storage.create_trade(trade)
            self.log(f"Trade added: {created.get('id')}")
        except Exception as e:
            self.log(f"add_trade error: {e}")

    def update_trade(self, trade_id: str, updates: Dict[str, Any]) -> bool:
        try:
            return self.storage.update_trade(trade_id, updates)
        except Exception as e:
            self.log(f"update_trade error: {e}")
            return False

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

    def place_order(self, account: Dict[str, Any], symbol: str, side: str, qty: float, price: Optional[float] = None, order_type: str = "Market", meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Wrapper to place an order. Will:
          - create a trade record in storage (status=created)
          - attempt to place via pybit client (best-effort; wrapped in try/except)
          - update trade record with response / status
        Returns the trade record (including response in meta if available).
        """
        trade = {
            "id": str(uuid.uuid4()),
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "account_id": account.get("id") or account.get("label") or "<unknown>",
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "status": "created",
            "meta": meta or {}
        }
        # store initial trade
        try:
            self.storage.create_trade(trade)
        except Exception as e:
            self.log(f"Failed to persist created trade: {e}")

        # If dry run, skip executing order
        if TRADE_SETTINGS.get("dry_run", True):
            trade["status"] = "dry_run"
            trade["meta"].update({"note": "dry run - no order created"})
            self.storage.update_trade(trade["id"], trade)
            self.log(f"Dry run order (not sent): {trade['id']}")
            return trade

        client = self._get_client(account)
        if not client:
            trade["status"] = "no_client"
            trade["meta"].update({"error": "missing api credentials"})
            self.storage.update_trade(trade["id"], trade)
            self.log(f"No client for account: {trade['account_id']}")
            return trade

        try:
            # best-effort: try some common pybit unified_trading method names; keep wrapped to avoid breaking
            response = None
            if TRADE_SETTINGS.get('use_market_order', True) or order_type.lower() == "market":
                try:
                    response = client.place_active_order(symbol=symbol, side=side, orderType="Market", qty=qty)
                except Exception:
                    # alternative naming
                    response = client.place_active_order(symbol=symbol, side=side, order_type="Market", qty=qty)
            else:
                # limit order attempt
                try:
                    response = client.place_active_order(symbol=symbol, side=side, orderType="Limit", qty=qty, price=price)
                except Exception:
                    response = client.place_active_order(symbol=symbol, side=side, order_type="Limit", qty=qty, price=price)

            trade["status"] = "placed"
            trade["meta"].update({"order_response": safe_json(response)})
            trade["updated_at"] = now_iso()
            self.storage.update_trade(trade["id"], trade)
            self.log(f"Order placed: {trade['id']}")
            return trade
        except Exception as e:
            trade["status"] = "error"
            trade["meta"].update({"error": str(e)})
            trade["updated_at"] = now_iso()
            try:
                self.storage.update_trade(trade["id"], trade)
            except Exception:
                self.log("Failed to persist error state for trade")
            self.log(f"place_order error: {e}")
            return trade

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
                # Reload config before each scan
                self.reload_config()
                # Scan logic would go here
                self.log("Scan cycle completed")
            except Exception as e:
                self.log(f"Run loop error: {e}")

            interval = int(TRADE_SETTINGS.get('scan_interval', 10))
            for _ in range(interval):
                if self._stop_event.is_set():
                    break
                time.sleep(1)


# -------------------- CLI / Example usage --------------------

if __name__ == '__main__':
    bc = BotController()
    print("Crypto Bot — Configuration via /api/config")
    # Simple test actions for quick manual checks:
    # 1) create a dry-run trade
    test_account = {"id": "local-test", "label": "local", "api_key": None, "api_secret": None}
    sample_trade = bc.place_order(test_account, "DOGEUSDT", "Buy", 10.0, price=None, order_type="Market", meta={"reason": "sanity-check"})
    print("Sample trade:", sample_trade)
    bc._run_loop()
