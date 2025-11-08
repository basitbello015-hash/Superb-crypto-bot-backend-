
# routes/auth_router.py
from fastapi import APIRouter, Depends, HTTPException, status, Response, Request, Form
from fastapi.security import OAuth2PasswordRequestForm
from services.auth_service import authenticate
from jwt_utils import (
    create_access_token,
    create_refresh_token,
    verify_access_token,
    verify_refresh_token,
)

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/login")
def login(response: Response, username: str = Form(...), password: str = Form(...)):
    user = authenticate(username, password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid login")

    access = create_access_token({"sub": username})
    refresh = create_refresh_token({"sub": username})

    # ✅ Refresh token stored securely in httpOnly cookie
    response.set_cookie(
        "refresh_token",
        refresh,
        httponly=True,
        secure=True,
        samesite="none",
        max_age=7 * 24 * 60 * 60,
    )

    return {"access_token": access, "token_type": "bearer"}


@router.post("/refresh")
def refresh_token(request: Request):
    refresh = request.cookies.get("refresh_token")

    if not refresh:
        raise HTTPException(401, "No refresh token")

    payload = verify_refresh_token(refresh)

    if not payload:
        raise HTTPException(401, "Invalid refresh token")

    username = payload["sub"]
    new_access = create_access_token({"sub": username})

    return {"access_token": new_access}


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie("refresh_token")
    return {"status": "logged_out"}
