import os

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
