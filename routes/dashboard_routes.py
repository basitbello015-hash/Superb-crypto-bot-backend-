# routes/dashboard_router.py
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

from services.dashboard_service import get_dashboard_data

templates = Jinja2Templates(directory="templates")

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

@router.get("/")
def dashboard(request: Request):
    data = get_dashboard_data()
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "data": data}
    )
