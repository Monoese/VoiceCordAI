import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config import Config

# Create logs directory if it doesn't exist
logs_dir = Path(Config.BASE_DIR) / "logs"
logs_dir.mkdir(exist_ok=True)

# Configure the root logger
def setup_logger(name=None):
    """
    Set up and configure a logger with the given name.
    If name is None, configure the root logger.
    """
    logger = logging.getLogger(name)
    
    # Only configure if handlers haven't been added yet
    if not logger.handlers:
        logger.setLevel(Config.LOG_LEVEL)
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        
        # Create file handler for logging to a file with rotation
        log_file = logs_dir / f"{name if name else 'root'}.log"
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=Config.LOG_MAX_SIZE, 
            backupCount=Config.LOG_BACKUP_COUNT
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(Config.LOG_LEVEL)
        
        # Create console handler for logging to console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(Config.LOG_CONSOLE_LEVEL)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

# Create loggers for each module
def get_logger(name):
    """Get a logger for the specified module."""
    return setup_logger(name)

# Set up the root logger
setup_logger()