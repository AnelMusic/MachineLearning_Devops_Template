#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 20:41:08 2021

@author: anelmusic
"""


import json
import pickle

from sklearn.model_selection import cross_val_score

from app import config
from titanic_classification import models, utils
from app.config import logger
import os
import warnings
import sys
import numpy as np
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from urllib.parse import urlparse

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


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

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    data_df = utils.load_data(config.PROCESSED_TRAIN_DATASET_PATH)
    X_train, Y_train = utils.split_to_feature_target_df(data_df)

    data_df_test = utils.load_data(config.PROCESSED_TEST_DATASET_PATH)
    X_test, Y_test = utils.split_to_feature_target_df(data_df_test)


    estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    with mlflow.start_run():
        lr = rf_classifier = RandomForestClassifier(random_state=0, n_estimators=estimators, max_depth=depth)
        lr.fit(X_train, Y_train)

        predicted = lr.predict(X_test)

        (rmse, mae, r2) = eval_metrics(Y_test, predicted)

        print("RandomForestClassifier (estimators=%f, depth=%f):" % (estimators, depth))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", estimators)
        mlflow.log_param("l1_ratio", depth)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel")
        else:
            mlflow.sklearn.log_model(lr, "model")