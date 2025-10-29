def get_config():
    return {"exchange": "Bybit", "strategy": "scalping", "risk": "medium"}

def save_config(data: dict):
    return {"status": "saved", "data": data}

def reset_config():
    return {"status": "reset", "defaults": get_config()}
