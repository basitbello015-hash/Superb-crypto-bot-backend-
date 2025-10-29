from fastapi import APIRouter
from services.dashboard_service import get_dashboard_data

router = APIRouter()

@router.get("")
def dashboard():
    return get_dashboard_data()
