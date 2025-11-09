# routes/auth_routes.py
from fastapi import APIRouter, Response, Request, HTTPException, status, Depends, Form
from fastapi.security import OAuth2PasswordRequestForm
from services.auth_service import authenticate, create_user
from utils.jwt_utils import create_access_token, create_refresh_token, verify_refresh_token

router = APIRouter(prefix="/api/auth", tags=["Auth"])

@router.post("/register")
def register(username: str = Form(...), password: str = Form(...)):
    created = create_user(username, password)
    if not created:
        raise HTTPException(status_code=409, detail="User already exists")
    return {"status": "ok", "username": created["username"]}

# Accepts form or JSON (we'll support form via OAuth2PasswordRequestForm)
@router.post("/login")
def login(response: Response, form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    access = create_access_token({"sub": user["username"]})
    refresh = create_refresh_token({"sub": user["username"]})

    # Set refresh token as httpOnly cookie — secure & sameSite as needed
    response.set_cookie(
        key="refresh_token",
        value=refresh,
        httponly=True,
        secure=True,           # set False for localhost testing with http
        samesite="none",       # for cross-site cookies; set to "lax" if same-site
        max_age=7 * 24 * 60 * 60
    )
    return {"access_token": access, "token_type": "bearer"}


@router.post("/refresh")
def refresh_token(request: Request):
    refresh = request.cookies.get("refresh_token")
    if not refresh:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing refresh token")
    payload = verify_refresh_token(refresh)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")
    username = payload.get("sub")
    new_access = create_access_token({"sub": username})
    return {"access_token": new_access}


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie("refresh_token")
    return {"status": "logged_out"}
