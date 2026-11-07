"""Configuration du logging structuré."""
import logging
import sys
from pathlib import Path

from config import LOG_FILE, LOG_LEVEL

def configurer_logging(nom: str = "facial") -> logging.Logger:
    logger = logging.getLogger(nom)
    if logger.handlers:
        return logger  # déjà configuré

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # Fichier
    try:
        fichier = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fichier.setFormatter(fmt)
        logger.addHandler(fichier)
    except Exception:
        pass  # on continue sans fichier si impossible

    return logger

log = configurer_logging()
