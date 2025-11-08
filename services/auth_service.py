import json
import os
from passlib.context import CryptContext
from typing import Optional

USERS_FILE = "app/users.json"
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Ensure file exists
if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, "w") as f:
        json.dump({"users": []}, f, indent=4)


def load_users():
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def save_users(data):
    with open(USERS_FILE, "w") as f:
        json.dump(data, f, indent=4)


def create_user(username: str, password: str):
    db = load_users()

    if any(u["username"] == username for u in db["users"]):
        return None  # user exists

    hashed = pwd_ctx.hash(password)

    new_user = {"username": username, "password": hashed}
    db["users"].append(new_user)
    save_users(db)
    return new_user


def authenticate_user(username: str, password: str) -> Optional[dict]:
    db = load_users()

    for u in db["users"]:
        if u["username"] == username and pwd_ctx.verify(password, u["password"]):
            return u

    return None
