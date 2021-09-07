# titanic_classification/config.py
# Configurations.

import logging.config
import sys
from pathlib import Path

from rich.logging import RichHandler

# Repository
AUTHOR = "Anel"
REPO = "Anel_Repo"

# Directories
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
LOGS_DIR = Path(BASE_DIR, "logs")
DATA_DIR = Path(BASE_DIR, "data")

# Create dirs
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# URL
DATASET_URL = "http://bit.ly/kaggletrain"

# Files
TRAIN_DATASET_PATH = Path(BASE_DIR, "data/titanic.csv")
PROCESSED_TRAIN_DATASET_PATH = Path(BASE_DIR, "data/titanic_train_processed.csv")
PROCESSED_TEST_DATASET_PATH = Path(BASE_DIR, "data/titanic_test_processed.csv")

# Prediction
PREDICTION_USER_DATA = Path(BASE_DIR, "data/prediction_data.csv")

# MODEL
MODEL_PATH = Path(BASE_DIR, "titanic_classifier.pkl")

# Features
FEATURES = [
    "Pclass",
    "Age",
    "Fare",
    "Title",
    "Embarked",
    "Fam_type",
    "Ticket_len",
    "Ticket_2letter",
]

# Logger
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {
            "format": "%(levelname)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "info.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "error.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
    },
    "loggers": {
        "root": {
            "handlers": ["console", "info", "error"],
            "level": logging.INFO,
            "propagate": True,
        },
    },
}

logging.config.dictConfig(logging_config)
logger = logging.getLogger("root")

logger.handlers[0] = RichHandler(markup=True)
