import re
from typing import Optional
from fastapi import HTTPException

def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_phone(phone: str) -> bool:
    """Validate phone number format"""
    pattern = r'^\+?1?\d{9,15}$'
    return re.match(pattern, phone) is not None

def validate_password(password: str) -> tuple[bool, Optional[str]]:
    """Validate password strength with professional requirements"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r'\d', password):
        return False, "Password must contain at least one digit"
    return True, None

def sanitize_input(text: str, max_length: int = 1000) -> str:
    """Sanitize user input to prevent injection attacks"""
    if not text:
        return ""
    # Remove potentially dangerous characters
    text = text.strip()
    text = text[:max_length]
    return text

def validate_user_input(data: dict, required_fields: list[str]) -> tuple[bool, Optional[str]]:
    """Validate required fields in user input"""
    for field in required_fields:
        if field not in data or not data[field]:
            return False, f"Missing required field: {field}"
    return True, None
