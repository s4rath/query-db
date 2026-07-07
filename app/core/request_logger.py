import os
import json
import threading
from datetime import datetime
from urllib.parse import urlparse

from pathlib import Path

class RequestLogger:
    """
    Singleton logger that records each AI request to a JSONL file.
    Features: 2MB rotation, hostname masking, and thread-safety.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(RequestLogger, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        # Calculate absolute path to project root/logs
        self.log_dir = Path(__file__).parent.parent.parent / "logs"
        self._ensure_dir()
        self._initialized = True

    def _ensure_dir(self):
        try:
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir, exist_ok=True)
        except Exception:
            pass

    def _get_log_file(self):
        """
        Determines the current log file based on date and size (2MB limit).
        Format: sql_agent_YYYY-MM-DD[_N].log
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        base_name = f"sql_agent_{date_str}"
        ext = ".log"
        max_size = 2 * 1024 * 1024 # 2MB
        
        suffix = 0
        while True:
            file_name = f"{base_name}_{suffix}{ext}" if suffix > 0 else f"{base_name}{ext}"
            file_path = os.path.join(self.log_dir, file_name)
            
            if not os.path.exists(file_path):
                return file_path
            
            if os.path.getsize(file_path) < max_size:
                return file_path
            
            suffix += 1

    def log_request(self, 
                    question: str, 
                    sql_query: str, 
                    final_answer: str, 
                    tokens: dict, 
                    api_calls: int, 
                    model: str, 
                    duration_ms: int,
                    success: bool = True,
                    error: str = None):
        """
        Serializes a single request execution to the latest log file.
        """
        # Mask db_url to show only host
        try:
            db_host = "localhost"
        except Exception:
            db_host = "unknown"

        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "duration_ms": duration_ms,
            "question": question,
            "sql_query": sql_query,
            "final_answer": final_answer,
            "model": model,
            "db_host": db_host,
            "tokens": tokens,
            "api_calls": api_calls,
            "success": success,
            "error": error
        }

        try:
            log_file = self._get_log_file()
            # Thread-safe append
            with self._lock:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            # Report the error to the console for debugging
            from .logger import log_error
            log_error(f"Request logging to file failed", error=e)

# Pre-instantiated singleton instance
request_logger = RequestLogger()
