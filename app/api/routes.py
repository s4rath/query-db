import time
from fastapi import APIRouter, HTTPException

from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, List
from app.services.agent import run_agent
from app.core.database import get_database, get_database_from_url
from app.core.config import settings
from app.core.logger import log_info, log_error

router = APIRouter()

# --- Models ---

class DBRequest(BaseModel):
    db_url: str

class DBResponse(BaseModel):
    status: str
    tables: List[str]

class Query(BaseModel):
    input: str
    db_url: Optional[str] = None
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    api_base: Optional[str] = None
    db_schema: Optional[dict] = Field(None, alias="schema")

class ChatResponse(BaseModel):
    response: str
    sql_query: Optional[str] = None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    api_calls: int
    duration_ms: int

# --- Endpoints ---

@router.post("/db/tables", response_model=DBResponse)
def get_db_tables(request: DBRequest):
    """Validates a database URL and returns a list of usable tables."""
    try:
        log_info(f"Fetching tables for dynamic DB")
        db = get_database_from_url(request.db_url)
        tables = db.get_usable_table_names()
        return DBResponse(status="success", tables=tables)
    except Exception as e:
        log_error("Error fetching tables", error=e)
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/chat", response_model=ChatResponse)
def chat(query: Query):
    """
    Executes a natural language query against a database.
    If db_url is provided, it uses it; otherwise falls back to local chinook.db.
    """
    start_time = time.perf_counter()
    try:
        if query.db_url:
            log_info("Using dynamic database for chat")
            db = get_database_from_url(query.db_url, schema=query.db_schema)
        else:
            log_info("Using default local database for chat")
            db = get_database(settings.DB_PATH)
            
        result = run_agent(
            user_input=query.input,
            db=db,
            api_key=query.api_key,
            model_name=query.model_name,
            api_base=query.api_base,
            schema=query.db_schema,
            start_time=start_time
        )

        return ChatResponse(**result)
    except Exception as e:
        log_error("Error in chat endpoint", error=e)
        raise HTTPException(status_code=500, detail=str(e))