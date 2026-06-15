import logging
import sys
from pathlib import Path


def init_logger(
    logger_name: str,
    log_file_path: Path = Path("logs/info.log"),
) -> logging.Logger:
    """Initialize a logger that outputs to a file and stdout.

    Parameters
    ----------
    logger_name : str
        Name of the logger.
    log_file_path : Path, optional
        Path to the log file, by default Path("logs/info.log")

    Returns
    -------
    logging.Logger
        Initialized logger.
    """
    log_file_path = Path(log_file_path)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
