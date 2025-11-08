import os
from fastapi import HTTPException, status

# Load keys from environment
MASTER_KEY = os.getenv("MGX_MASTER_KEY", "")
STATIC_TOKEN = os.getenv("MGX_STATIC_TOKEN", "")

def verify_login_key(key: str) -> str:
    """
    Accepts only the master key.
    Returns a static token that frontend will store & use.
    """
    if key != MASTER_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid key"
        )
    return STATIC_TOKEN

def verify_token(token: str):
    """
    Checks if the token provided in Authorization header is valid.
    """
    if token != STATIC_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Invalid or missing token"
        )
