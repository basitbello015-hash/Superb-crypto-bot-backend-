from fastapi import APIRouter
from pydantic import BaseModel
from .auth_service import verify_login_key

router = APIRouter()

class LoginBody(BaseModel):
    key: str

@router.post("/login")
def login(body: LoginBody):
    token = verify_login_key(body.key)
    return {"token": token}
