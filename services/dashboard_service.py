# services/dashboard_service.py
from app_state import bc

def get_dashboard_data():
    # You can modify these fields based on your bot logic

    return {
        "total_balance": bc.get_total_balance() if hasattr(bc, "get_total_balance") else 0,
        "active_trades": len(bc.active_positions) if hasattr(bc, "active_positions") else 0,
        "today_pnl": bc.get_today_pnl() if hasattr(bc, "get_today_pnl") else 0,
        "win_rate": bc.get_win_rate() if hasattr(bc, "get_win_rate") else 0,
    }
