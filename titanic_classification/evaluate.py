#!/usr/bin/env python3
"""
Created on Sun Aug 29 01:41:51 2021

@author: anelmusic
"""
import json

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from app import config
from app.config import logger
from titanic_classification import utils


def eval_on_test_data():
    """


    Returns
    -------
    None.

    """
    data_df = utils.load_data(config.PROCESSED_TEST_DATASET_PATH)
    X_test, Y_test = utils.split_to_feature_target_df(data_df)

    model = utils.load_best_model()

    Y_test_predicted = model.predict(X_test)

    # Performance
    metrics = {"overall": {}, "class": {}}

    classes = [0, 1]  # binary classification

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
    logger.info("Performance:\n" + json.dumps(metrics, indent=2))

    try:
        with open(config.BEST_MODEL_PERFORMANCE_PATH, "w") as fid:
            json.dump(metrics, fid)
    except OSError as e:
        print(repr(e))
        logger.error(repr(e))
