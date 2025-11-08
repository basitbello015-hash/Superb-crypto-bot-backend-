import time
from fastapi import HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
import jwt
import os

SECRET_KEY = os.getenv("JWT_SECRET", "MGX_SUPER_SECRET")
ALGORITHM = "HS256"
EXPIRE_MIN = 60 * 24  # 24 hours

oauth2 = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def create_access_token(data: dict):
    payload = data.copy()
    payload["exp"] = time.time() + EXPIRE_MIN

    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str = Depends(oauth2)):
    try:
        decoded = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return decoded
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")
