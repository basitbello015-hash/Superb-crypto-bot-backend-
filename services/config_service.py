import json
import os

CONFIG_FILE = "config.json"

DEFAULT_CONFIG = {
    "exchange": "Bybit",
    "strategy": "scalping",
    "risk": "medium",
    "tradeAllocation": 0.1,
    "minTradeAmount": 10,
    "scanInterval": 60
}

def get_config():
    if not os.path.exists(CONFIG_FILE):
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)

def save_config(data: dict):
    config = get_config()
    config.update(data)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)
    return {"status": "saved", "data": config}

def reset_config():
    save_config(DEFAULT_CONFIG)
    return {"status": "reset", "defaults": DEFAULT_CONFIG}
