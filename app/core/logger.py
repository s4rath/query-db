import logging
import sys
from typing import Any

# Configure logging
def setup_logger(name: str = "sql_agent_api"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger

logger = setup_logger()

def log_info(message: str, **kwargs: Any):
    extra = f" | {kwargs}" if kwargs else ""
    logger.info(f"{message}{extra}")

def log_error(message: str, error: Exception = None, **kwargs: Any):
    extra = f" | {kwargs}" if kwargs else ""
    if error:
        logger.error(f"{message}{extra}", exc_info=True)
    else:
        logger.error(f"{message}{extra}")
