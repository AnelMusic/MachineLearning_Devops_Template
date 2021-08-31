#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 02:02:44 2021

@author: anelmusic
"""

import json
import pickle

from sklearn.model_selection import cross_val_score

from app import config
from titanic_classification import models, utils
from app.config import logger


def train():

    data_df = utils.load_data(config.PROCESSED_TRAIN_DATASET_PATH)
    X_train, Y_train = utils.split_to_feature_target_df(data_df)
    model = models.rf_classifier
    model.fit(X_train, Y_train)
    cv_score = cross_val_score(model, X_train, Y_train, cv=5).mean()

    try:
        with open("train_cross_val_score.json", "a") as fp:
            cv_score = {"model": repr(model), "cv_score": cv_score}
            json.dump(cv_score, fp)
            fp.write("\n")
    except OSError as e:
        print(repr(e))
        logger.error(repr(e))

    try:
        with open(config.MODEL_PATH, "wb") as fid:
            pickle.dump(model, fid)
    except OSError as e:
        print(repr(e))
        logger.error(repr(e))