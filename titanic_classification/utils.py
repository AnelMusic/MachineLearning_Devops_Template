import json
import os

import mlflow.sklearn
import pandas as pd
import wget

from app import config
from app.config import logger


def download_data_from_url(url):
    dataset_filepath = str(config.TRAIN_DATASET_PATH)

    if not os.path.isfile(dataset_filepath):
        wget.download(url, dataset_filepath)
        logger.info("✅ Data downloaded!")
    else:
        logger.info("✅ Data already present!")


def load_data(path):
    dataframe = pd.read_csv(path)
    return dataframe.iloc[:, 1:]


def save_data(data_frame, path):
    data_frame.to_csv(path)


def get_artificial_feature_list(num_features):
    return ["F_" + str(x) for x in range(num_features)]


def split_to_feature_target_df(data_frame):
    Y = data_frame["Survived"]
    X = data_frame.drop("Survived", axis=1)
    return X, Y


def save_best_model():
    # Reading Pandas Dataframe from mlflow
    df = mlflow.search_runs(filter_string="metrics.rmse < 1")
    # Fetching Run ID for
    run_id = df.loc[df["metrics.rmse"].idxmin()]["run_id"]
    # Load model
    model = mlflow.sklearn.load_model("runs:/" + run_id + "/model")
    model_params = model.get_params()
    model_info = {"params": model_params, "run_id": run_id}

    try:
        with open(config.BEST_MODEL_PATH, "a") as fp:
            json.dump(model_info, fp)
    except OSError as e:
        print(repr(e))
        logger.error(repr(e))


def load_best_model():
    try:
        with open(config.BEST_MODEL_PATH) as fp:
            data = json.load(fp)
            return mlflow.sklearn.load_model("runs:/" + str(data["run_id"]) + "/model")

    except OSError as e:
        print(repr(e))
        logger.error(repr(e))
