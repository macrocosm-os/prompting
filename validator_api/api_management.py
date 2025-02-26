import json
import secrets
from typing import Dict, Any

from fastapi import APIRouter, Depends, Header, HTTPException, status, Path, Query
from loguru import logger
from pydantic import BaseModel, Field

from shared import settings

shared_settings = settings.shared_settings

router = APIRouter()


# Models for API management
class ApiKeyResponse(BaseModel):
    """Response model for API key operations."""
    
    message: str = Field(..., description="Status message about the operation")
    api_key: str = Field(..., description="The API key value")


class ApiKeyDeleteResponse(BaseModel):
    """Response model for API key deletion."""
    
    message: str = Field(..., description="Status message about the deletion")


class ApiKeyInfo(BaseModel):
    """Model for API key information."""
    
    rate_limit: int = Field(..., description="Maximum number of requests allowed per time period")
    usage: int = Field(..., description="Current usage count")


# Load and save functions for API keys
def load_api_keys():
    try:
        with open(shared_settings.API_KEYS_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"API keys are not found: {shared_settings.API_KEYS_FILE}")
        return {}
    except json.JSONDecodeError:
        logger.exception("JSON decode error when reading API keys")
        return {}


def save_api_keys(api_keys):
    with open(shared_settings.API_KEYS_FILE, "w") as f:
        json.dump(api_keys, f)


# Use lifespan to initialize API keys
_keys = load_api_keys()


# Dependency to validate the admin key
def validate_admin_key(admin_key: str = Header(..., description="Admin key for API management operations")):
    """
    Validates the admin key for protected operations.
    
    Raises:
        HTTPException: If the admin key is invalid
    """
    if admin_key != shared_settings.ADMIN_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Invalid admin key"
        )


# Dependency to validate API keys
def validate_api_key(api_key: str = Header(..., description="API key for authentication")):
    """
    Validates the API key for protected endpoints.
    
    Returns:
        dict: API key metadata including rate limit and usage
        
    Raises:
        HTTPException: If the API key is invalid
    """
    if api_key not in _keys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Invalid API key"
        )
    return _keys[api_key]


@router.post(
    "/create-api-key/",
    summary="Create new API key",
    description="Creates a new API key with a specified rate limit. Requires admin authentication.",
    status_code=status.HTTP_201_CREATED,
    response_model=ApiKeyResponse,
    responses={
        status.HTTP_201_CREATED: {
            "description": "API key created successfully",
            "model": ApiKeyResponse,
            "content": {
                "application/json": {
                    "example": {"message": "API key created", "api_key": "1234567890abcdef"}
                }
            }
        },
        status.HTTP_403_FORBIDDEN: {
            "description": "Invalid admin key"
        }
    }
)
def create_api_key(
    rate_limit: int = Query(
        ..., 
        description="Maximum number of requests allowed per time period",
        example=100,
        ge=1
    ), 
    admin_key: str = Depends(validate_admin_key)
):
    """
    Creates a new API key with a specified rate limit.
    
    The endpoint generates a secure random API key and associates it with the specified rate limit.
    The API key is stored persistently and can be used immediately.
    
    Parameters:
    - **rate_limit** (int): Maximum number of requests allowed per time period. Must be greater than 0.
    - **admin_key** (str): Admin authentication key (passed as a header)
    
    Returns:
    - A JSON object containing the newly created API key and a success message
    """
    global _keys
    new_api_key = secrets.token_hex(16)
    _keys[new_api_key] = {"rate_limit": rate_limit, "usage": 0}
    save_api_keys(_keys)
    _keys = load_api_keys()
    return {"message": "API key created", "api_key": new_api_key}


@router.put(
    "/modify-api-key/{api_key}",
    summary="Modify API key",
    description="Updates the rate limit for an existing API key. Requires admin authentication.",
    response_model=ApiKeyResponse,
    responses={
        status.HTTP_200_OK: {
            "description": "API key updated successfully",
            "model": ApiKeyResponse,
            "content": {
                "application/json": {
                    "example": {"message": "API key updated", "api_key": "1234567890abcdef"}
                }
            }
        },
        status.HTTP_403_FORBIDDEN: {
            "description": "Invalid admin key"
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "API key not found"
        }
    }
)
def modify_api_key(
    api_key: str = Path(
        ..., 
        description="The API key to modify",
        example="1234567890abcdef"
    ), 
    rate_limit: int = Query(
        ..., 
        description="New maximum number of requests allowed per time period",
        example=200,
        ge=1
    ), 
    admin_key: str = Depends(validate_admin_key)
):
    """
    Modifies the rate limit of an existing API key.
    
    Parameters:
    - **api_key** (str): The API key to modify (path parameter)
    - **rate_limit** (int): New maximum number of requests allowed per time period. Must be greater than 0.
    - **admin_key** (str): Admin authentication key (passed as a header)
    
    Returns:
    - A JSON object confirming the API key was updated
    
    Raises:
    - HTTPException 404: If the API key doesn't exist
    """
    if api_key not in _keys:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="API key not found"
        )
    _keys[api_key]["rate_limit"] = rate_limit
    save_api_keys(_keys)
    return {"message": "API key updated", "api_key": api_key}


@router.delete(
    "/delete-api-key/{api_key}",
    summary="Delete API key",
    description="Removes an existing API key. Requires admin authentication.",
    response_model=ApiKeyDeleteResponse,
    responses={
        status.HTTP_200_OK: {
            "description": "API key deleted successfully",
            "model": ApiKeyDeleteResponse,
            "content": {
                "application/json": {
                    "example": {"message": "API key deleted"}
                }
            }
        },
        status.HTTP_403_FORBIDDEN: {
            "description": "Invalid admin key"
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "API key not found"
        }
    }
)
def delete_api_key(
    api_key: str = Path(
        ..., 
        description="The API key to delete",
        example="1234567890abcdef"
    ), 
    admin_key: str = Depends(validate_admin_key)
):
    """
    Deletes an existing API key.
    
    Parameters:
    - **api_key** (str): The API key to delete (path parameter)
    - **admin_key** (str): Admin authentication key (passed as a header)
    
    Returns:
    - A JSON object confirming the API key was deleted
    
    Raises:
    - HTTPException 404: If the API key doesn't exist
    """
    if api_key not in _keys:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="API key not found"
        )
    del _keys[api_key]
    save_api_keys(_keys)
    return {"message": "API key deleted"}
