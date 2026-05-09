"""FastAPI application entry point."""
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.config import settings
from app.api.routes import upload as upload_routes
from app.api.routes import chat as chat_routes
from app.api.routes import sessions as sessions_routes
from app.api.routes import documents as documents_routes
from app.api.routes import dashboard as dashboard_routes
from app.models.schemas import HealthResponse
import logging
import traceback
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="PDF Chatbot API",
    description="RAG-based PDF chatbot using OpenAI and Zilliz",
    version="1.0.0"
)

# Add exception handler to log all unhandled errors
from fastapi.exceptions import HTTPException as FastAPIHTTPException

@app.exception_handler(FastAPIHTTPException)
async def http_exception_handler(request: Request, exc: FastAPIHTTPException):
    """Log HTTP exceptions."""
    import sys
    print(f">>> HTTP EXCEPTION: {exc.status_code} - {exc.detail}", flush=True)
    sys.stdout.flush()
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to log all unhandled errors."""
    import sys
    error_trace = traceback.format_exc()
    print(f">>> GLOBAL EXCEPTION: {type(exc).__name__}: {str(exc)}", flush=True)
    print(f">>> TRACEBACK:\n{error_trace}", flush=True)
    sys.stdout.flush()
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"Internal server error: {str(exc)}",
            "error_type": type(exc).__name__
        }
    )

# Add middleware to log requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    import sys
    print(f">>> INCOMING REQUEST: {request.method} {request.url.path}", flush=True)
    sys.stdout.flush()
    try:
        response = await call_next(request)
        print(f">>> RESPONSE: {response.status_code} for {request.method} {request.url.path}", flush=True)
        sys.stdout.flush()
        return response
    except Exception as e:
        print(f">>> MIDDLEWARE ERROR: {type(e).__name__}: {str(e)}", flush=True)
        print(traceback.format_exc(), flush=True)
        sys.stdout.flush()
        raise

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
print("INFO: Registering routes...")
app.include_router(upload_routes.router)
print("INFO: Upload router registered")
app.include_router(chat_routes.router)
print("INFO: Chat router registered")
app.include_router(sessions_routes.router)
print("INFO: Sessions router registered")
app.include_router(documents_routes.router)
print("INFO: Documents router registered")
app.include_router(dashboard_routes.router)
print("INFO: Dashboard router registered")
print("INFO: All routes registered successfully")


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        message="PDF Chatbot API is running"
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    logger.info("Health check requested")
    return HealthResponse(
        status="healthy",
        message="Service is operational"
    )


@app.get("/test-log")
async def test_log():
    """Test endpoint to verify logging works."""
    logger.info("Test log endpoint called")
    logger.warning("This is a warning message")
    logger.error("This is an error message (test)")
    return {"message": "Check console for logs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level="info"
    )
