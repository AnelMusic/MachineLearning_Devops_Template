import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import titanic_classification.feature_engineering as fe
from app import config
from titanic_classification import utils

"""
methods to be tested

"""


def run_preprocessing_pipeline(data_df):
    """


    Parameters
    ----------
    data_df : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    features = config.FEATURES
    numerical_cols = ["Age", "Fare"]
    categorical_cols = [
        "Pclass",
        "Title",
        "Embarked",
        "Fam_type",
        "Ticket_len",
        "Ticket_2letter",
    ]

    X = data_df[features]
    y = data_df["Survived"]

    numerical_transformer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Bundle preprocessing for numerical and categorical data
    column_trans = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    processed = column_trans.fit_transform(X)

    # Pipeline result = scipy.sparse.csr_matrix
    # Must be transformed otherwise its interpreted as 1D array
    processed = processed.toarray()

    num_samples, num_features = processed.shape

    feature_list = utils.get_artificial_feature_list(num_features)
    pocessed_df = pd.DataFrame(data=processed, columns=feature_list)

    pocessed_df["Survived"] = y

    return pocessed_df


def split_and_store_dataset(data_frame):
    train, test = train_test_split(data_frame, test_size=0.2)
    utils.save_data(train, config.PROCESSED_TRAIN_DATASET_PATH)
    utils.save_data(test, config.PROCESSED_TEST_DATASET_PATH)


def process_dataset():
    data_df = utils.load_data(config.TRAIN_DATASET_PATH)
    fe.perform_feature_engineering(data_df)
    print(list(data_df.columns.values))
    processed_data_df = run_preprocessing_pipeline(data_df)
    split_and_store_dataset(processed_data_df)


process_dataset()
