from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import OAuth2PasswordRequestForm
from services.auth_service import create_user, authenticate_user
from utils.jwt_utils import create_access_token, verify_token

router = APIRouter()


@router.post("/register")
def register(data: dict):
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password required")

    user = create_user(username, password)

    if not user:
        raise HTTPException(status_code=409, detail="User already exists")

    return {"status": "success", "message": "User registered"}


@router.post("/login")
def login(form: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form.username, form.password)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = create_access_token({"sub": user["username"]})
    return {"access_token": token, "token_type": "bearer"}


@router.get("/verify")
def verify(token_data: dict = Depends(verify_token)):
    return {"valid": True, "user": token_data["sub"]}
