# app/cli.py
# Command line interface (CLI) application.


import warnings
from pathlib import Path

import typer

from app import config
from app.config import logger
from titanic_classification import data, train, utils, evaluate

# Ignore warning
warnings.filterwarnings("ignore")

# Typer CLI app
app = typer.Typer()


@app.command()
def download_data():
    """Load data from URL and save to local drive."""
    # Download data
    utils.download_data_from_url(config.DATASET_URL)


@app.command()
def show_dataset_head(params_fp: Path = Path(config.CONFIG_DIR, "params.json")) -> None:
    dataset_df = utils.load_data(config.TRAIN_DATASET_PATH)
    print(dataset_df.head())
    logger.info("✅ Dataset head shown")


@app.command()
def process_dataset(params_fp: Path = Path(config.CONFIG_DIR, "params.json")) -> None:
    data.process_dataset()
    logger.info("✅ Data processed")


@app.command()
def train_model():
    logger.info("✅ Model trained")
    train.train()
    
@app.command()
def eval_model():
    logger.info("✅ Model Evaluated")
    evaluate.eval_on_test_data()
