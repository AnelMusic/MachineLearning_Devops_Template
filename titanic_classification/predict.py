#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 01:41:51 2021

@author: anelmusic
"""
import pickle

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

from app import config
from app.config import logger
from titanic_classification import utils


def predict_sample():
    data_df = utils.load_data(config.PREDICTION_USER_DATA)


    try:
        with open(config.MODEL_PATH, "rb") as fid:
            model = pickle.load(fid)
            prediction = model.predict(data_df)
            print(str(prediction))
            logger.info(repr(model)+ "\npredicted {} on test_user_data".format(prediction))
    except OSError as e:
        print(repr(e))
        logger.error(repr(e))


predict_sample()
