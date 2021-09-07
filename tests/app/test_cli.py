#!/usr/bin/env python3
"""
Created on Wed Aug 25 23:22:32 2021

@author: anelmusic
"""

# tests/app/test_cli.py
# Test app/cli.py components.

from pathlib import Path

from typer.testing import CliRunner

from app import config
from app.cli import app

runner = CliRunner()


def test_download_data():
    result = runner.invoke(app, ["download-data"])
    assert result.exit_code == 0
    assert "Data" in result.stdout


def test_show_dataset_head(params_fp: Path = Path(config.CONFIG_DIR, "params.json")) -> None:
    result = runner.invoke(app, ["show-dataset-head"])
    assert result.exit_code == 0
    assert "TODO" in result.stdout


def test_compute_features(params_fp: Path = Path(config.CONFIG_DIR, "params.json")) -> None:
    result = runner.invoke(app, ["get-features"])
    assert result.exit_code == 0
    assert "TODO" in result.stdout


def test_get_features():
    result = runner.invoke(app, ["get-features"])
    assert result.exit_code == 0
    assert "TODO" in result.stdout


def train_model():
    result = runner.invoke(app, ["train-mode"])
    assert result.exit_code == 0
    assert "TODO" in result.stdout
