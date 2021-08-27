# app/cli.py
# Command line interface (CLI) application.


import warnings
from pathlib import Path

import typer

from app import config
from titanic_classification import utils
from app.config import logger


# Ignore warning
warnings.filterwarnings("ignore")

# Typer CLI app
app = typer.Typer()


@app.command()
def download_data():
    """Load data from URL and save to local drive."""
    # Download data
    utils.download_data_from_url(config.DATASET_URL)

    # Sample messages (note we use configured `logger` now)
    # logger.debug("Used for debugging your code.")
    # logger.info("Informative messages from your code.")
    # logger.warning("Everything works but there is something to be aware of.")
    # logger.error("There's been a mistake with the process.")
    # ogger.critical("There is something terribly wrong and process may terminate.")


@app.command()
def show_dataset_head(params_fp: Path = Path(config.CONFIG_DIR, "params.json")) -> None:
    logger.info("TODO show_dataset_head")


@app.command()
def compute_features(params_fp: Path = Path(config.CONFIG_DIR, "params.json")) -> None:
    logger.info("TODO compute_features")


@app.command()
def get_features():
    logger.info("TODO get_features")


@app.command()
def train_model():
    logger.info("TODO train_model")
