#!/usr/bin/env python3
"""
Created on Fri Aug 27 02:02:44 2021

@author: anelmusic
"""

import pickle

from app import config
from app.config import logger
from titanic_classification import utils


def train():
    """


    Returns
    -------
    None.

    """

    data_df = utils.load_data(config.PROCESSED_TRAIN_DATASET_PATH)
    X_train, Y_train = utils.split_to_feature_target_df(data_df)
    model = utils.load_best_model()
    model.fit(X_train, Y_train)

    try:
        with open(config.MODEL_PATH, "wb") as fid:
            pickle.dump(model, fid)
    except OSError as e:
        print(repr(e))
        logger.error(repr(e))
