#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 19:17:12 2021

@author: anelmusic
"""




import great_expectations as ge
from app import config
from titanic_classification import utils

import pytest



@pytest.fixture(scope = 'module')
def dataset_df():
    data_df = utils.load_data(config.TRAIN_DATASET_PATH)
    return  ge.dataset.PandasDataset(data_df)


def test_features_present(dataset_df):
    expected_columns = ["Survived", "Pclass",
                        "Name", "Sex", "Age",
                        "SibSp", "Parch", "Ticket",
                        "Fare", "Cabin", "Embarked"
                        ]
    result = dataset_df.expect_table_columns_to_match_ordered_list(
        column_list=expected_columns
        )
    assert result["success"]

def test_has_unique_name(dataset_df):
    result = dataset_df.expect_column_values_to_be_unique(column="Name")
    assert result["success"]

def test_has_unique_feature(dataset_df):
    result = dataset_df.expect_column_values_to_be_unique(column="Name")
    assert result["success"]

def test_target_not_null(dataset_df):
    result = dataset_df.expect_column_values_to_not_be_null(column="Survived")
    assert result["success"]

def test_number_of_samples(dataset_df):
    assert len(dataset_df) > 500

def test_name_type_str(dataset_df):
    result = dataset_df.expect_column_values_to_be_of_type(
        column="Name", type_="str"
        )
    assert result["success"]

def test_age_type_float(dataset_df):
    result = dataset_df.expect_column_values_to_be_of_type(
        column="Age", type_ ="float"
        )
    assert result["success"]
