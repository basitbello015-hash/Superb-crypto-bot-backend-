from future import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

IMPORTANT: pybit unified_trading HTTP client

from pybit.unified_trading import HTTP

-------------------- CONFIG --------------------

Load configuration from config.json

def load_config():
config_file = os.path.join(os.path.dirname(file), "config.json")
if os.path.exists(config_file):
with open(config_file, "r") as f:
return json.load(f)
return {}

Initial config load

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

Risk rules with config override

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

BASE_DIR = os.path.dirname(os.path.abspath(file))

ACCOUNTS_FILE = os.path.join(BASE_DIR, "app/accounts.json")
TRADES_FILE = os.path.join(BASE_DIR, "app/trades.json")

Ensure app directory exists

os.makedirs(os.path.join(BASE_DIR, "app"), exist_ok=True)

Ensure files exist

for f in [ACCOUNTS_FILE, TRADES_FILE]:
if not os.path.exists(f):
with open(f, "w") as fp:
fp.write("[]")

-------------------- Utilities --------------------

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

-------------------- Technical helpers --------------------

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

-------------------- Bot Controller --------------------

class BotController:
def init(self, log_queue: Optional[threading.Queue] = None):
self.log_queue = log_queue
self._running = False
self._stop_event = threading.Event()
self._file_lock = threading.Lock()
self._threads: List[threading.Thread] = []

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

if name == 'main':
bc = BotController()
print("Crypto Bot — Configuration via /api/config")
bc._run_loop()
