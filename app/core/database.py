import json
import hashlib
from langchain_community.utilities import SQLDatabase
from functools import lru_cache
from sqlalchemy import create_engine, event
from .logger import log_info


# ─── SSL Context (Supabase/Neon) ──────────────────────────
# def _make_ssl_context():
#     """Creates a basic SSL context for cloud DBs that require it."""
#     ctx = ssl.create_default_context()
#     ctx.check_hostname = False
#     ctx.verify_mode = ssl.CERT_NONE
#     return ctx

# ─── Engine Cache ──────────────────────────────────────────
@lru_cache(maxsize=20)
def _get_cached_engine_internal(db_url: str):
    """
    Internal cached engine creator. Reuses connection pools per URL.
    """
    masked_url = db_url.split("@")[-1] if "@" in db_url else db_url
    log_info(f"Creating new SQLAlchemy engine for: ...@{masked_url}")

    connect_args = {}
    
    # Check stats for debugging (optional log)
    # stats = _get_cached_engine_internal.cache_info()
    # log_info(f"Engine Cache Stats: Hits={stats.hits}, Misses={stats.misses}")

    engine = create_engine(
        db_url,
        pool_size=10,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=1800,
        connect_args=connect_args,
    )

    # Driver-level read-only protection
    @event.listens_for(engine, "before_execute")
    def before_execute(conn, clauseelement, multiparams, params, execution_options):
        query_str = str(clauseelement).strip().upper()
        BLOCKED = [
            "DROP", "DELETE", "INSERT", "UPDATE",
            "TRUNCATE", "ALTER", "CREATE", "EXEC",
            "EXECUTE", "LOAD_FILE", "INTO OUTFILE"
        ]
        for keyword in BLOCKED:
            if f" {keyword} " in f" {query_str} " or query_str.startswith(keyword):
                raise ValueError(f"CRITICAL: Blocked statement: {keyword}")

    return engine

def get_cached_engine(db_url: str):
    """Normalized entry point for engine caching."""
    stats = _get_cached_engine_internal.cache_info()
    log_info(f"Engine Cache: Hits={stats.hits}, Misses={stats.misses}, Size={stats.currsize}")
    return _get_cached_engine_internal(db_url.strip())

# ─── SQLDatabase Cache ────────────────────────────────────
_SCHEMA_DATA_STORE = {}

@lru_cache(maxsize=20)
def _get_cached_db_internal(db_url: str, schema_hash: str) -> SQLDatabase:
    """Internal cached DB wrapper."""
    engine = get_cached_engine(db_url)
    
    custom_table_info = _SCHEMA_DATA_STORE.get(schema_hash)
    if custom_table_info:
        log_info(f"Creating SQLDatabase with cached custom_table_info (hash: {schema_hash})")
        return SQLDatabase(
            engine,
            custom_table_info=custom_table_info,
            include_tables=list(custom_table_info.keys())
        )
    
    log_info("Creating SQLDatabase with live introspection (no schema provided)")
    return SQLDatabase(engine)


def get_cached_db(db_url: str, schema: dict = None) -> SQLDatabase:
    """Normalized entry point for SQLDatabase caching."""
    schema_hash = "none"
    if schema and "tables" in schema:
        # Transform backend schema to LangChain custom_table_info format
        custom_table_info = {
            t["table_name"]: t["schema_text"]
            for t in schema["tables"]
        }
        # Generate stable hash for the cache key
        schema_hash = hashlib.md5(
            json.dumps(custom_table_info, sort_keys=True).encode()
        ).hexdigest()
        
        # Store data for the cached function to retrieve on miss
        _SCHEMA_DATA_STORE[schema_hash] = custom_table_info
    
    return _get_cached_db_internal(db_url.strip(), schema_hash)

# ─── Schema Cache ─────────────────────────────────────────
@lru_cache(maxsize=20)
def _get_cached_tables_internal(db_url: str) -> str:
    """Internal cached table discovery."""
    db = get_cached_db(db_url)
    tables = db.get_usable_table_names()
    log_info(f"Retrieved and cached tables for DB.")
    
    # Check stats
    stats = _get_cached_tables_internal.cache_info()
    log_info(f"Table Cache Stats: Hits={stats.hits}, Misses={stats.misses}")
    
    return ", ".join(tables)

def get_cached_tables(db_url: str) -> str:
    """Normalized entry point for schema caching."""
    return _get_cached_tables_internal(db_url.strip())

# ─── Legacy/Local Accessors ────────────────────────────────
def get_database(db_path: str) -> SQLDatabase:
    """Gets a local SQLite database instance."""
    prefix = "sqlite:////" if db_path.startswith("/") else "sqlite:///"
    return get_cached_db(f"{prefix}{db_path}")

def get_database_from_url(db_url: str, schema: dict = None) -> SQLDatabase:
    """Legacy wrapper for backward compatibility."""
    return get_cached_db(db_url, schema=schema)

# ─── Cache Management ─────────────────────────────────────
def invalidate_db_cache():
    """Clears all database-related caches."""
    _get_cached_engine_internal.cache_clear()
    _get_cached_db_internal.cache_clear()
    _get_cached_tables_internal.cache_clear()
    log_info("All database caches invalidated.")

def get_cache_stats() -> dict:
    """Useful for monitoring cache efficiency."""
    return {
        "engine_cache": _get_cached_engine_internal.cache_info()._asdict(),
        "db_cache": _get_cached_db_internal.cache_info()._asdict(),
        "tables_cache": _get_cached_tables_internal.cache_info()._asdict(),
    }