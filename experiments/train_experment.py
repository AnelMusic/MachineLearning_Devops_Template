#!/usr/bin/env python3
"""
Created on Wed Sep  1 20:41:08 2021

@author: anelmusic
"""


import sys
import warnings
from urllib.parse import urlparse

import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from app import config
from titanic_classification import utils


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    seed = np.random.seed(40)

    data_df = utils.load_data(config.PROCESSED_TRAIN_DATASET_PATH)
    X_train, Y_train = utils.split_to_feature_target_df(data_df)

    data_df_test = utils.load_data(config.PROCESSED_TEST_DATASET_PATH)
    X_test, Y_test = utils.split_to_feature_target_df(data_df_test)

    estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    with mlflow.start_run():
        rf_classifier = RandomForestClassifier(
            random_state=seed, n_estimators=estimators, max_depth=depth
        )
        rf_classifier.fit(X_train, Y_train)

        predicted = rf_classifier.predict(X_test)

        (rmse, mae, r2) = eval_metrics(Y_test, predicted)

        print(f"RandomForestClassifier (estimators={estimators:f}, depth={depth:f}):")
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
            mlflow.sklearn.log_model(rf_classifier, "model", registered_model_name="RForest")
        else:
            mlflow.sklearn.log_model(rf_classifier, "model")

    utils.save_best_model()
