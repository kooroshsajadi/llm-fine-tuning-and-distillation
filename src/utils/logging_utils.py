import logging
from pathlib import Path
from typing import Optional

def setup_logger(
    name: str,
    log_dir: str = "logs",
    log_level: str = "INFO",
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with console and file output for pipeline modules.

    Args:
        name (str): Name of the logger (e.g., 'pdf_classifier').
        log_dir (str): Directory to store log files (default: 'logs').
        log_level (str): Logging level (e.g., 'DEBUG', 'INFO', 'ERROR').
        log_file (Optional[str]): Custom log file name (default: <name>.log).

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Avoid duplicate handlers if logger is already configured
    if logger.handlers:
        return logger

    # Create log directory
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    # Default log file name if not provided
    log_file = log_file or f"{name}.log"
    log_file_path = log_dir_path / log_file

    # Define log format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    try:
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.error("Failed to set up file handler: %s", str(e))

    return logger