"""
Session storage backend for RAG agent conversations.

This module provides different storage backends for conversation sessions:
- In-memory (development)
- Redis (production)
"""

import json
import os
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage


class SessionStorage(ABC):
    """Abstract base class for session storage backends."""
    
    @abstractmethod
    def save_session(self, session_id: str, messages: List[BaseMessage]) -> None:
        """Save conversation messages for a session."""
        pass
    
    @abstractmethod
    def load_session(self, session_id: str) -> List[BaseMessage]:
        """Load conversation messages for a session."""
        pass
    
    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """Delete a session. Returns True if session existed."""
        pass
    
    @abstractmethod
    def list_sessions(self) -> List[str]:
        """List all active session IDs."""
        pass
    
    @abstractmethod
    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        pass


class RedisStorage(SessionStorage):
    """Redis-based session storage (production)."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None
    ):
        """Initialize Redis storage.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password (if required)
        """
        try:
            import redis
        except ImportError:
            raise ImportError(
                "Redis is not installed in this environment. Run: uv sync"
            )
        
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False  # We'll handle encoding ourselves
        )
        
        # Test connection
        try:
            self.redis_client.ping()
        except redis.ConnectionError as e:
            raise ConnectionError(
                f"Cannot connect to Redis at {host}:{port}. "
                f"Make sure Redis is running. Error: {e}"
            )
    
    def _serialize_message(self, msg: BaseMessage) -> dict:
        """Serialize a LangChain message to dict."""
        if isinstance(msg, dict):
            return msg
        elif isinstance(msg, HumanMessage):
            return {"type": "human", "content": msg.content}
        elif isinstance(msg, AIMessage):
            return {"type": "ai", "content": msg.content}
        elif isinstance(msg, ToolMessage):
            return {
                "type": "tool",
                "content": msg.content,
                "tool_call_id": msg.tool_call_id
            }
        else:
            return {"type": "system", "content": str(msg)}
    
    def _deserialize_message(self, data: dict) -> BaseMessage:
        """Deserialize a dict to LangChain message."""
        msg_type = data.get("type", "system")
        
        if msg_type == "human":
            return HumanMessage(content=data["content"])
        elif msg_type == "ai":
            return AIMessage(content=data["content"])
        elif msg_type == "tool":
            return ToolMessage(
                content=data["content"],
                tool_call_id=data.get("tool_call_id", "")
            )
        else:
            # System message as dict
            return data
    
    def save_session(self, session_id: str, messages: List[BaseMessage]) -> None:
        key = f"session:{session_id}"
        
        # Serialize messages
        serialized = [self._serialize_message(msg) for msg in messages]
        
        # Save to Redis
        self.redis_client.set(
            key,
            json.dumps(serialized)
        )
    
    def load_session(self, session_id: str) -> List[BaseMessage]:
        """Load session from Redis."""
        key = f"session:{session_id}"
        data = self.redis_client.get(key)
        
        if data is None:
            return []
        
        # Deserialize messages
        serialized = json.loads(data)
        return [self._deserialize_message(msg) for msg in serialized]
    
    def delete_session(self, session_id: str) -> bool:
        """Delete session from Redis."""
        key = f"session:{session_id}"
        result = self.redis_client.delete(key)
        return result > 0
    
    def list_sessions(self) -> List[str]:
        """List all session IDs."""
        keys = self.redis_client.keys("session:*")
        return [key.decode().replace("session:", "") for key in keys]
    
    def session_exists(self, session_id: str) -> bool:
        """Check if session exists in Redis."""
        key = f"session:{session_id}"
        return self.redis_client.exists(key) > 0


# Global storage instance (can be configured)
_storage: Optional[SessionStorage] = None


def get_storage() -> SessionStorage:
    """Get the global storage instance."""
    global _storage
    
    if _storage is None:
        # Try to use Redis if available, fall back to memory
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        redis_password = os.getenv("REDIS_PASSWORD")
        
        try:
            _storage = RedisStorage(
                host=redis_host,
                port=redis_port,
                password=redis_password
            )
            print(f"✓ Using Redis session storage at {redis_host}:{redis_port}")
        except (ImportError, ConnectionError) as e:
            print(f"⚠ Redis not available ({e})")
    
    return _storage


def set_storage(storage: SessionStorage) -> None:
    """Set the global storage instance."""
    global _storage
    _storage = storage
