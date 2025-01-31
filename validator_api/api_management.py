import json
import secrets

from fastapi import APIRouter, Depends, Header, HTTPException
from loguru import logger

from shared.settings import shared_settings

router = APIRouter()


# Load and save functions for API keys
def load_api_keys():
    try:
        with open(shared_settings.API_KEYS_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_api_keys(api_keys):
    with open(shared_settings.API_KEYS_FILE, "w") as f:
        json.dump(api_keys, f)


# Use lifespan to initialize API keys
_keys = load_api_keys()
logger.info(f"Loaded API keys: {_keys}")
# save_api_keys(_keys)


# Dependency to validate the admin key
def validate_admin_key(admin_key: str = Header(...)):
    if admin_key != shared_settings.ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin key")


# Dependency to validate API keys
def validate_api_key(api_key: str = Header(...)):
    if api_key not in _keys:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return _keys[api_key]


@router.post("/create-api-key/")
def create_api_key(rate_limit: int, admin_key: str = Depends(validate_admin_key)):
    """Creates a new API key with a specified rate limit."""
    global _keys
    new_api_key = secrets.token_hex(16)
    _keys[new_api_key] = {"rate_limit": rate_limit, "usage": 0}
    save_api_keys(_keys)
    _keys = load_api_keys()
    return {"message": "API key created", "api_key": new_api_key}


@router.put("/modify-api-key/{api_key}")
def modify_api_key(api_key: str, rate_limit: int, admin_key: str = Depends(validate_admin_key)):
    """Modifies the rate limit of an existing API key."""
    if api_key not in _keys:
        raise HTTPException(status_code=404, detail="API key not found")
    _keys[api_key]["rate_limit"] = rate_limit
    save_api_keys(_keys)
    return {"message": "API key updated", "api_key": api_key}


@router.delete("/delete-api-key/{api_key}")
def delete_api_key(api_key: str, admin_key: str = Depends(validate_admin_key)):
    """Deletes an existing API key."""
    if api_key not in _keys:
        raise HTTPException(status_code=404, detail="API key not found")
    del _keys[api_key]
    save_api_keys(_keys)
    return {"message": "API key deleted"}
