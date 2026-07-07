from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.api.routes import router
from app.core.logger import log_info, log_error
from fastapi.middleware.cors import CORSMiddleware
import time
import os

app = FastAPI(title="LangGraph SQL Agent API")

ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8000").split(",")

# Add CORS support
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS, 
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log_error(f"Unhandled exception: {request.method} {request.url}", error=exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please check logs for details."},
    )

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    log_info(f"{request.method} {request.url} - {response.status_code} - {process_time:.2f}ms")
    return response

app.include_router(router)

@app.get("/")
def health():
    return {"status": "ok", "message": "SQL Agent API is running"}

@app.on_event("startup")
async def startup_event():
    log_info("Application starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    log_info("Application shutting down...")