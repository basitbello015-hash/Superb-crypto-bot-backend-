from fastapi import APIRouter
from services.history_service import get_dashboard_data

router = APIRouter()

@router.get("")
def history():
    return get_history()
