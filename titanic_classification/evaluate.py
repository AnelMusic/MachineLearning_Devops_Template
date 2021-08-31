#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 01:41:51 2021

@author: anelmusic
"""
import pickle

import numpy as np

from app import config
from app.config import logger
from titanic_classification import utils

from sklearn.metrics import precision_recall_fscore_support
import json

def eval_on_test_data():
    data_df = utils.load_data(config.PROCESSED_TEST_DATASET_PATH)
    X_test, Y_test = utils.split_to_feature_target_df(data_df)
    try:
        with open(config.MODEL_PATH, "rb") as fid:
            model = pickle.load(fid)

            Y_test_predicted = model.predict(X_test)

            # Performance
            metrics = {"overall": {}, "class": {}}

            classes = [0,1] # binary classification


            # Overall metrics
            overall_metrics = precision_recall_fscore_support(Y_test, Y_test_predicted, average="weighted")
            metrics["overall"]["precision"] = overall_metrics[0]
            metrics["overall"]["recall"] = overall_metrics[1]
            metrics["overall"]["f1"] = overall_metrics[2]
            metrics["overall"]["num_samples"] = np.float64(len(Y_test_predicted))

            # Per-class metrics
            class_metrics = precision_recall_fscore_support(Y_test, Y_test_predicted, average=None)

            for i, _class in enumerate(classes):
                metrics["class"][_class] = {
                    "precision": class_metrics[0][i],
                    "recall": class_metrics[1][i],
                    "f1": class_metrics[2][i],
                    "num_samples": np.float64(class_metrics[3][i]),
                }
            logger.info('Performance:\n'+json.dumps(metrics, indent=2))

    except OSError as e:
        print(repr(e))
        logger.error(repr(e))


eval_on_test_data()