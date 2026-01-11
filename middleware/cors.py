from fastapi.middleware.cors import CORSMiddleware
import os

def setup_cors(app):
    """Configure CORS middleware with environment-aware settings"""
    
    # Define allowed origins based on environment
    allowed_origins = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ]
    
    # Add production domains from environment variable
    prod_domains = os.getenv('ALLOWED_ORIGINS', '').split(',')
    allowed_origins.extend([domain.strip() for domain in prod_domains if domain.strip()])
    
    # Add Vercel deployments
    allowed_origins.extend([
        "https://*.vercel.app",
    ])
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=[
            "Content-Type",
            "Authorization",
            "Accept",
            "Origin",
            "Access-Control-Request-Method",
            "Access-Control-Request-Headers",
        ],
        expose_headers=[
            "Content-Type",
            "Authorization",
        ],
        max_age=3600,  # Cache preflight requests for 1 hour
    )
