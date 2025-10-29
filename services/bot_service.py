from bot_fib_scoring import BC

def get_status():
    return {"running": BC.is_running()}

def start_bot():
    if BC.is_running():
        return {"status": "already running"}
    BC.start()
    return {"status": "bot started"}

def stop_bot():
    if not BC.is_running():
        return {"status": "already stopped"}
    BC.stop()
    return {"status": "bot stopped"}
