import great_expectations as ge
import pytest

from app import config
from titanic_classification import utils


@pytest.fixture(scope="module")
def dataset_df():
    data_df = utils.load_data(config.PROCESSED_TRAIN_DATASET_PATH)
    return ge.dataset.PandasDataset(data_df)


def test_features_present(dataset_df):
    expected_columns = [*utils.get_artificial_feature_list(89), *["Survived"]]
    result = dataset_df.expect_table_columns_to_match_ordered_list(column_list=expected_columns)
    assert result["success"]


def test_target_not_null(dataset_df):
    result = dataset_df.expect_column_values_to_not_be_null(column="Survived")
    assert result["success"]


def test_number_of_samples(dataset_df):
    assert len(dataset_df) > 500


def test_no_feature_nan(dataset_df):
    assert not dataset_df.isnull().values.any()
