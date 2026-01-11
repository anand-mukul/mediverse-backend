from fastapi import Request
import time
import logging
import json

logger = logging.getLogger(__name__)

async def log_requests(request: Request, call_next):
    """Log all incoming requests with detailed information"""
    start_time = time.time()
    
    # Extract useful request info
    method = request.method
    path = request.url.path
    client_host = request.client.host if request.client else "unknown"
    
    logger.info(f"📥 {method} {path} from {client_host}")
    
    try:
        # Process request
        response = await call_next(request)
        
        # Log response with performance metrics
        process_time = time.time() - start_time
        status_code = response.status_code
        
        # Color code by status
        if status_code >= 500:
            logger.error(f"🔴 {method} {path} - {status_code} ({process_time:.2f}s)")
        elif status_code >= 400:
            logger.warning(f"🟡 {method} {path} - {status_code} ({process_time:.2f}s)")
        else:
            logger.info(f"🟢 {method} {path} - {status_code} ({process_time:.2f}s)")
        
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"💥 {method} {path} - Exception ({process_time:.2f}s): {str(e)}")
        raise
