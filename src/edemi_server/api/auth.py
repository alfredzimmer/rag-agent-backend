import os
import hashlib
import secrets
from typing import Dict
from uuid import UUID, uuid4
import jwt
import datetime
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, Header, status
from edemi_server.api.dependency import get_agent

JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 30

router = APIRouter(
    prefix="/api/auth",
    tags=["Authentication"]
)

# In-memory dictionary for active session tokens
# Maps token string -> {"user_id": UUID, "username": str}
ACTIVE_SESSIONS: Dict[str, dict] = {}

def get_jwt_secret_key() -> str:
    secret_key = os.getenv("JWT_SECRET_KEY")
    if not secret_key:
        raise RuntimeError("JWT_SECRET_KEY must be configured")
    return secret_key

# Pydantic schemas
class UserCredentials(BaseModel):
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
def hash_password(password: str, salt: bytes = None) -> str:
    if salt is None:
        salt = secrets.token_bytes(16)
    # Using 100,000 iterations of PBKDF2-HMAC-SHA256
    pwd_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
    return f"{salt.hex()}:{pwd_hash.hex()}"

def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt_hex, hash_hex = stored_hash.split(":")
        salt = bytes.fromhex(salt_hex)
        pwd_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
        return pwd_hash.hex() == hash_hex
    except Exception:
        return False

# Database Initialization and Data Migration
async def init_auth_db(pool):
    print("MIGRATION: Starting database tables initialization and data migration...")
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            # 1. Create users table
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id UUID PRIMARY KEY,
                    username VARCHAR(255) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            # 2. Create user_sessions table
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    conversation_id UUID PRIMARY KEY,
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    title VARCHAR(255) DEFAULT 'New Chat',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            await cur.execute("""
                ALTER TABLE user_sessions ADD COLUMN IF NOT EXISTS title VARCHAR(255) DEFAULT 'New Chat';
            """)
            # Create active_sessions table for token persistence across restarts
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS active_sessions (
                    token VARCHAR(255) PRIMARY KEY,
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    username VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 3. Ensure 'alfred' exists
            alfred_id = "a1f8ed00-1111-2222-3333-444455556666"
            await cur.execute("SELECT id FROM users WHERE username = 'alfred'")
            row = await cur.fetchone()
            if not row:
                default_user_password = os.getenv("DEFAULT_USER_PASSWORD")
                if not default_user_password:
                    raise RuntimeError(
                        "DEFAULT_USER_PASSWORD must be configured to create the default user"
                    )
                print("MIGRATION: Registering default user 'alfred'...")
                pwd_hash = hash_password(default_user_password)
                await cur.execute(
                    "INSERT INTO users (id, username, password_hash) VALUES (%s, %s, %s)",
                    (alfred_id, "alfred", pwd_hash)
                )

            # 4. Migrate existing checkpoints metadata to alfred
            await cur.execute("""
                SELECT COUNT(*) FROM checkpoints
                WHERE metadata ? 'user_id' AND metadata->>'user_id' != %s
            """, (alfred_id,))
            to_update_count = (await cur.fetchone())[0]
            if to_update_count > 0:
                print(f"MIGRATION: Mapping {to_update_count} checkpoints metadata user_id to alfred...")
                await cur.execute("""
                    UPDATE checkpoints
                    SET metadata = jsonb_set(metadata, '{user_id}', %s::jsonb)
                    WHERE metadata ? 'user_id' AND metadata->>'user_id' != %s;
                """, (f'"{alfred_id}"', alfred_id))

            # 5. Populate user_sessions for all existing unique thread_id (sessions)
            await cur.execute("SELECT DISTINCT thread_id FROM checkpoints")
            rows = await cur.fetchall()
            existing_threads = [r[0] for r in rows]

            migrated_count = 0
            for thread_id in existing_threads:
                try:
                    # Verify thread_id is a valid UUID
                    UUID(thread_id)
                    await cur.execute(
                        "INSERT INTO user_sessions (conversation_id, user_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                        (thread_id, alfred_id)
                    )
                    migrated_count += 1
                except ValueError:
                    # Ignore non-UUID thread IDs
                    pass
            if migrated_count > 0:
                print(f"MIGRATION: Successfully mapped {migrated_count} existing conversation sessions to user 'alfred'.")
    print("MIGRATION: Database initialization and migration completed successfully.")

# Authorization Dependency
async def get_current_user(
    authorization: str = Header(None, description="Bearer token"),
    agent=Depends(get_agent)
) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header. Format: 'Bearer <token>'"
        )

    token = authorization.split(" ")[1]

    # 1. Try to verify JWT signature and expiration statelessly
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
        # 2. Fall back to database lookup for legacy/non-JWT session tokens
        try:
            async with agent.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT user_id, username FROM active_sessions WHERE token = %s",
                        (token,)
                    )
                    row = await cur.fetchone()
                    if row:
                        return {
                            "user_id": row[0],
                            "username": row[1]
                        }
        except Exception as db_err:
            print(f"Error querying active_sessions table for legacy token: {db_err}")

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
            expiration = datetime.datetime.utcnow() + datetime.timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
            payload = {
                "user_id": str(user_id),
                "username": username,
                "exp": expiration
            }
            token = jwt.encode(payload, get_jwt_secret_key(), algorithm=JWT_ALGORITHM)

            # Save it to active_sessions database table for persistent tracking
            try:
                await cur.execute(
                    "INSERT INTO active_sessions (token, user_id, username) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                    (token, user_id, username)
                )
            except Exception as db_err:
                print(f"Warning: Failed to save active session to DB: {db_err}")

            return AuthResponse(
                user_id=user_id,
                username=username,
                token=token
            )
