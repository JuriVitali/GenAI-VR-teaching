import structlog
import logging
import logging.handlers
import sys
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
LOG_FILE_PATH = os.getenv("LOG_FILE_PATH")

def setup_logging(service_name: str, log_file_path: str = LOG_FILE_PATH):
    # Ensure directory exists
    log_path = Path(log_file_path).absolute()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.handlers.RotatingFileHandler(
        str(log_path), maxBytes=10*1024*1024, backupCount=5
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s", 
        handlers=[console_handler, file_handler],
        force=True
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            lambda _, __, event_dict: {**event_dict, "service": service_name},
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer() 
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    return structlog.get_logger()