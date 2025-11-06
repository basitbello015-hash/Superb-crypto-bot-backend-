# routes/control_router.py
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from app_state import bc

templates = Jinja2Templates(directory="templates")

router = APIRouter(prefix="/control", tags=["Control"])

@router.get("/")
def control_page(request: Request):
    status = bc.is_running  
    return templates.TemplateResponse(
        "control.html",
        {
            "request": request,
            "active": "control",
            "status": "Running" if status else "Stopped"
        }
    )

@router.post("/start")
def start_bot():
    bc.start()
    return {"status": "started"}

@router.post("/stop")
def stop_bot():
    bc.stop()
    return {"status": "stopped"}
