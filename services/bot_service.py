from bot_fib_scoring import BotController

def get_status():
    return {"running": BotController.is_running()}

def start_bot():
    if BotController.is_running():
        return {"status": "already running"}
    BotController.start()
    return {"status": "bot started"}

def stop_bot():
    if not BotController.is_running():
        return {"status": "already stopped"}
    BotController.stop()
    return {"status": "bot stopped"}
