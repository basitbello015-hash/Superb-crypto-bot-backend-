# services/history_service.py
import json
import os
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# assume app/trades.json lives relative to project root; adjust if your path differs
TRADES_FILE = os.path.join(os.path.dirname(BASE_DIR), "app", "trades.json")
_lock = threading.Lock()


def _ensure_file():
    folder = os.path.dirname(TRADES_FILE)
    os.makedirs(folder, exist_ok=True)
    if not os.path.exists(TRADES_FILE):
        with open(TRADES_FILE, "w") as f:
            json.dump([], f)


def _read_trades() -> List[Dict[str, Any]]:
    _ensure_file()
    try:
        with open(TRADES_FILE, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except Exception:
        return []


def _write_trades(trades: List[Dict[str, Any]]):
    _ensure_file()
    with _lock:
        with open(TRADES_FILE, "w") as f:
            json.dump(trades, f, indent=2, default=str)


def _sanitize_trade(tr: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "id": str(tr.get("id") or tr.get("trade_id") or str(uuid.uuid4())),
        "symbol": tr.get("symbol"),
        "side": tr.get("side"),
        "qty": tr.get("qty") or tr.get("quantity"),
        "price": tr.get("price"),
        "status": tr.get("status"),
        "profit": tr.get("profit") or (tr.get("meta", {}) or {}).get("profit"),
        "created_at": tr.get("created_at") or tr.get("createdAt") or tr.get("timestamp"),
    }
    return out


def get_trades(limit: int = 50, offset: int = 0, symbol: Optional[str] = None, status: Optional[str] = None) -> Tuple[List[Dict[str, Any]], int]:
    """
    Return (trades_slice, total_count)
    Filters by symbol and status if provided (case-insensitive).
    """
    all_trades = _read_trades()
    # ensure proper structure
    clean = [_sanitize_trade(t) for t in all_trades]

    if symbol:
        symbol_up = symbol.upper()
        clean = [t for t in clean if (t.get("symbol") or "").upper() == symbol_up]

    if status:
        status_low = status.lower()
        clean = [t for t in clean if (t.get("status") or "").lower() == status_low]

    total = len(clean)
    slice_trades = clean[offset: offset + limit]
    return slice_trades, total


def get_trade_by_id(trade_id: str) -> Optional[Dict[str, Any]]:
    all_trades = _read_trades()
    for t in all_trades:
        if str(t.get("id")) == trade_id or str(t.get("trade_id")) == trade_id:
            return _sanitize_trade(t)
    return None


def append_trade(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a trade to trades.json. Returns the saved trade object (sanitized).
    If payload has no id, one will be generated.
    """
    with _lock:
        trades = _read_trades()
        new_trade = payload.copy()
        if not new_trade.get("id"):
            new_trade["id"] = str(uuid.uuid4())
        trades.append(new_trade)
        _write_trades(trades)
    return _sanitize_trade(new_trade)
