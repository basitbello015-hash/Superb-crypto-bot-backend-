# services/auth_service.py
import json
import os
import threading
from passlib.context import CryptContext
from typing import Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USERS_FILE = os.path.join(BASE_DIR, "app", "users.json")
_lock = threading.Lock()

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

def _ensure_file():
    folder = os.path.dirname(USERS_FILE)
    os.makedirs(folder, exist_ok=True)
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w") as f:
            json.dump({"users": []}, f, indent=2)

def _load():
    _ensure_file()
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def _save(data):
    _ensure_file()
    with open(USERS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def create_user(username: str, password: str) -> Optional[dict]:
    with _lock:
        db = _load()
        if any(u["username"] == username for u in db.get("users", [])):
            return None
        hashed = pwd_ctx.hash(password)
        user = {"username": username, "password": hashed}
        db["users"].append(user)
        _save(db)
        return {"username": username}

def authenticate(username: str, password: str) -> Optional[dict]:
    db = _load()
    for u in db.get("users", []):
        if u["username"] == username and pwd_ctx.verify(password, u["password"]):
            return {"username": username}
    return None
