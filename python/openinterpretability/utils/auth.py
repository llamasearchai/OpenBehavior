"""
Authentication utilities for OpenInterpretability API.
"""

import hashlib
import hmac
import os
import time
from typing import Optional
from fastapi import HTTPException, Header, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .config import config_manager

security = HTTPBearer(auto_error=False)


def generate_api_key() -> str:
    """Generate a new API key."""
    import secrets
    return f"oi_{secrets.token_urlsafe(32)}"


def hash_api_key(api_key: str) -> str:
    """Hash an API key for storage."""
    salt = os.getenv("API_KEY_SALT", "openinterpretability_salt")
    return hashlib.pbkdf2_hex(api_key.encode(), salt.encode(), 100000)


def verify_api_key(api_key: str) -> bool:
    """
    Verify if the provided API key is valid.
    
    Args:
        api_key: API key to verify
        
    Returns:
        True if valid, False otherwise
    """
    # In production, this would check against a database or key store
    # For now, accept any non-empty key or specific test keys
    valid_keys = {
        "development-key",
        "test-api-key",
        os.getenv("MASTER_API_KEY", "")
    }
    
    return api_key in valid_keys or len(api_key) >= 32  # Accept long keys


async def get_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> str:
    """
    Extract and validate API key from request.
    
    Args:
        credentials: HTTP authorization credentials
        
    Returns:
        API key string
        
    Raises:
        HTTPException: If API key is invalid or missing
    """
    # For development/testing, allow requests without API key
    api_key_required = os.getenv("API_KEY_REQUIRED", "false").lower() == "true"
    
    if not api_key_required:
        return "development-key"
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    api_key = credentials.credentials
    
    if not verify_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return api_key


class APIKeyManager:
    """Manage API keys and authentication."""
    
    def __init__(self):
        self.keys = {}  # In production, use a database
        
    def create_key(self, user_id: str, description: str = "") -> str:
        """Create a new API key for a user."""
        api_key = generate_api_key()
        key_hash = hash_api_key(api_key)
        
        self.keys[key_hash] = {
            "user_id": user_id,
            "description": description,
            "created_at": time.time(),
            "last_used": None,
            "usage_count": 0,
            "active": True
        }
        
        return api_key
    
    def validate_key(self, api_key: str) -> Optional[dict]:
        """Validate an API key and return key info."""
        key_hash = hash_api_key(api_key)
        
        if key_hash in self.keys:
            key_info = self.keys[key_hash]
            if key_info["active"]:
                # Update usage tracking
                key_info["last_used"] = time.time()
                key_info["usage_count"] += 1
                return key_info
        
        return None
    
    def revoke_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        key_hash = hash_api_key(api_key)
        
        if key_hash in self.keys:
            self.keys[key_hash]["active"] = False
            return True
        
        return False
    
    def list_keys(self, user_id: str) -> list:
        """List all API keys for a user."""
        user_keys = []
        
        for key_hash, key_info in self.keys.items():
            if key_info["user_id"] == user_id:
                user_keys.append({
                    "hash": key_hash[:8] + "...",  # Truncated for security
                    "description": key_info["description"],
                    "created_at": key_info["created_at"],
                    "last_used": key_info["last_used"],
                    "usage_count": key_info["usage_count"],
                    "active": key_info["active"]
                })
        
        return user_keys


# Global API key manager
api_key_manager = APIKeyManager() 