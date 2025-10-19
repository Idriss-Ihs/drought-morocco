import logging
from pathlib import Path
import yaml

def load_config(path="src/config/settings.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def setup_logger(name: str = "drought", log_file: str = None, level=logging.INFO):
    """Configure a file + console logger with consistent format."""
    cfg = load_config()
    log_path = log_file or cfg["paths"]["logs"]

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:  # avoid duplicate handlers
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh = logging.FileHandler(log_path)
        fh.setFormatter(formatter)
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger
