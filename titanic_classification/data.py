

from app import config
from titanic_classification import utils
from app.config import logger

import titanic_classification.feature_engineering as fe

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import pandas as pd


"""
methods to be tested

"""
def run_preprocessing_pipeline(data_df):


    features = ['Pclass', 'Age', 'Fare', 'Title', 'Embarked',
                'Fam_type', 'Ticket_len', 'Ticket_2letter'
                ]
    numerical_cols = ['Age','Fare']
    categorical_cols = ['Pclass', 'Title',
                        'Embarked', 'Fam_type',
                        'Ticket_len', 'Ticket_2letter'
                        ]

    X = data_df[features]
    # evtl here try except
    y = data_df['Survived']
    # zwecks feature in test_data

    numerical_transformer = SimpleImputer(strategy='median')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])


    # Bundle preprocessing for numerical and categorical data
    column_trans = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    processed = column_trans.fit_transform(X)

    # Pipeline result = scipy.sparse.csr_matrix
    # Must be transformed otherwise its interpreted as 1D array
    processed = processed.toarray()


    num_samples, num_features = processed.shape
    feature_list = utils.get_artificial_feature_list(num_features)

    pocessed_df = pd.DataFrame(data=processed, columns = feature_list)

    # Append Target var after feature
    pocessed_df['Survived'] = y
    utils.save_data(pocessed_df, config.PROCESSED_DATASET_PATH)

    """
    TODO
    Preprocessing Train data different than test data !!!!
    Feature Survived not inside Test data
    Must be tackled
    """


def process_dataset():
    data_df = utils.load_data(config.DATASET_PATH)
    fe.perform_feature_engineering(data_df)
    run_preprocessing_pipeline(data_df)

    # Test
    processed_data = utils.load_data(config.PROCESSED_DATASET_PATH)
    print(processed_data.head)


process_dataset()

