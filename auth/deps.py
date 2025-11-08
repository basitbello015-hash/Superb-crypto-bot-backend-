from fastapi import Header, HTTPException, status
from .auth_service import verify_token

def require_auth(authorization: str = Header(None)):
    """
    Extracts and validates Bearer token from headers.
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization format")

    token = authorization.split("Bearer ")[1]
    verify_token(token)
    return True
