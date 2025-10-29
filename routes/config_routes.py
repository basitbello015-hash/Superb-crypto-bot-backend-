from fastapi import APIRouter
from services.config_service import get_config, save_config, reset_config

router = APIRouter()

@router.get("")
def fetch_config():
    return get_config()

@router.post("")
def update_config(data: dict):
    return save_config(data)

@router.post("/reset")
def reset_to_defaults():
    return reset_config()
