from typing import Any, Optional
from fastapi.responses import JSONResponse

def success_response(data: Any, message: str = "Success", status_code: int = 200):
    """Standard success response"""
    return JSONResponse(
        status_code=status_code,
        content={
            "success": True,
            "message": message,
            "data": data
        }
    )

def error_response(message: str, status_code: int = 400, details: Optional[dict] = None):
    """Standard error response"""
    content = {
        "success": False,
        "message": message
    }
    if details:
        content["details"] = details
    
    return JSONResponse(
        status_code=status_code,
        content=content
    )
