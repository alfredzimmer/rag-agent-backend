import os
import hashlib
import logging
import secrets
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4
import jwt
from pydantic import BaseModel, ConfigDict, Field
from fastapi import APIRouter, Depends, HTTPException, Header, status
from rag_agent_server.api.dependency import get_agent

logger = logging.getLogger(__name__)

JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 30
PASSWORD_HASH_ITERATIONS = 100_000

router = APIRouter(
    prefix="/api/auth",
    tags=["Authentication"]
)

def get_jwt_secret_key() -> str:
    secret_key = os.getenv("JWT_SECRET_KEY")
    if not secret_key:
        raise RuntimeError("JWT_SECRET_KEY must be configured")
    return secret_key

# Pydantic schemas
class UserCredentials(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(..., min_length=2, max_length=50)
    password: str = Field(..., min_length=6, max_length=100)

class AuthResponse(BaseModel):
    user_id: UUID
    username: str
    token: str

class UserResponse(BaseModel):
    user_id: UUID
    username: str

# Password Hashing Helper functions
def hash_password(password: str, salt: bytes | None = None) -> str:
    if salt is None:
        salt = secrets.token_bytes(16)
    pwd_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_HASH_ITERATIONS,
    )
    return f"{salt.hex()}:{pwd_hash.hex()}"

def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt_hex, hash_hex = stored_hash.split(":")
        salt = bytes.fromhex(salt_hex)
        pwd_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            PASSWORD_HASH_ITERATIONS,
        )
        return secrets.compare_digest(pwd_hash.hex(), hash_hex)
    except Exception:
        return False

async def init_auth_db(pool):
    logger.info("Initializing authentication schema")
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id UUID PRIMARY KEY,
                    username VARCHAR(255) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    conversation_id UUID PRIMARY KEY,
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    title VARCHAR(255) DEFAULT 'New Chat',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
    logger.info("Authentication schema ready")

# Authorization Dependency
async def get_current_user(
    authorization: str = Header(None, description="Bearer token"),
) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header. Format: 'Bearer <token>'"
        )

    token = authorization.split(" ")[1]

    try:
        payload = jwt.decode(token, get_jwt_secret_key(), algorithms=[JWT_ALGORITHM])
        user_id_str = payload.get("user_id")
        username = payload.get("username")
        if not user_id_str or not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload."
            )
        return {
            "user_id": UUID(user_id_str),
            "username": username
        }
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session has expired. Please log in again."
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session has expired or is invalid. Please log in again."
        )

# Routes
@router.post("/register", response_model=UserResponse)
async def register_user(creds: UserCredentials, agent=Depends(get_agent)):
    username = creds.username.strip().lower()

    async with agent.pool.connection() as conn:
        async with conn.cursor() as cur:
            # Check if username exists
            await cur.execute("SELECT id FROM users WHERE username = %s", (username,))
            row = await cur.fetchone()
            if row:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username is already taken."
                )

            # Create user
            user_id = uuid4()
            pwd_hash = hash_password(creds.password)
            await cur.execute(
                "INSERT INTO users (id, username, password_hash) VALUES (%s, %s, %s)",
                (user_id, username, pwd_hash)
            )

            return UserResponse(user_id=user_id, username=username)

@router.post("/login", response_model=AuthResponse)
async def login_user(creds: UserCredentials, agent=Depends(get_agent)):
    username = creds.username.strip().lower()

    async with agent.pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT id, password_hash FROM users WHERE username = %s", (username,))
            row = await cur.fetchone()
            if not row:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Incorrect username or password."
                )

            user_id, pwd_hash = row
            if not verify_password(creds.password, pwd_hash):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Incorrect username or password."
                )

            # Generate a secure signed JWT
            issued_at = datetime.now(timezone.utc)
            expiration = issued_at + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
            payload = {
                "user_id": str(user_id),
                "username": username,
                "iat": issued_at,
                "exp": expiration,
            }
            token = jwt.encode(payload, get_jwt_secret_key(), algorithm=JWT_ALGORITHM)

            return AuthResponse(
                user_id=user_id,
                username=username,
                token=token
            )
