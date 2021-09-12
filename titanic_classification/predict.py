#!/usr/bin/env python3
"""
Created on Sun Aug 29 01:41:51 2021

@author: anelmusic
"""

from app import config
from app.config import logger
from titanic_classification import utils

"""

Load best model not here random model
"""


def predict_sample_processed():
    """


    Returns
    -------
    None.

    """
    data_df = utils.load_data(config.PREDICTION_USER_DATA)

    model = utils.load_best_model()
    prediction = model.predict(data_df)
    print(str(prediction))
    logger.info(repr(model) + f"\npredicted {prediction} on test_user_data")


predict_sample_processed()
