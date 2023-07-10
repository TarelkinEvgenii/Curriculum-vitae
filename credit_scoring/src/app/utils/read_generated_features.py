import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def read_train_generated_features(path_to_input: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Функция читает сгенерированные файлы."""
    log_format = '[%(asctime)s] %(name)-25s %(levelname)-8s %(message)s'
    logging.basicConfig(
        format=log_format,
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    # importing features from hw4
    logger.info('Start reading files')
    bureau = pd.read_csv(
        os.path.join(
            path_to_input,
            'features_bureau.csv',
        ),
    )
    credit_card_balance = pd.read_csv(
        os.path.join(path_to_input, 'features_credit_card_balance.csv'),
    )
    installments_payments = pd.read_csv(
        os.path.join(path_to_input, 'installments_payments.csv'),
    )
    previous_application = pd.read_csv(
        os.path.join(path_to_input, 'previous_application.csv'),
    )
    application_train = pd.read_csv(
        os.path.join(path_to_input, 'features_application_train.csv'),
    )

    # importing original train file to get TARGER values
    train_original = pd.read_csv(
        os.path.join(path_to_input, 'original_train_file.csv'),
    )
    train_original = train_original[['SK_ID_CURR', 'TARGET']]
    logger.info('Start merging files')
    merged_features = pd.merge(
        train_original,
        application_train,
        how='inner',
        on='SK_ID_CURR',
    )
    merged_features = pd.merge(
        merged_features,
        previous_application,
        how='inner',
        on='SK_ID_CURR',
    )
    merged_features = pd.merge(
        merged_features,
        installments_payments,
        how='inner',
        on='SK_ID_CURR',
    )
    merged_features = pd.merge(
        merged_features,
        credit_card_balance,
        how='inner',
        on='SK_ID_CURR',
    )
    merged_features = pd.merge(
        merged_features,
        bureau,
        how='inner',
        on='SK_ID_CURR',
    )

    train_y = merged_features[['TARGET']]
    feature_columns = list(merged_features.columns)
    feature_columns.remove('SK_ID_CURR')
    feature_columns.remove('TARGET')
    train_x = merged_features[feature_columns]

    logger.info('Start dealing with Nan and Inf values')
    # Так как часть колонок слишком разряжена и колонки вычислены сложно,
    # то есть предположение,
    # что есть колонки состоящие из Nan значений полностью, поэтому попробуем удалить их
    train_x.dropna(axis=1, how='all', inplace=True)
    # Для логистической регрессии и решающего дерева нельзя иметь Nan значения,
    # поэтому заполним их
    train_x.replace([np.inf, -np.inf], np.nan, inplace=True)
    for ic in tqdm(train_x.columns[train_x.isnull().any(axis=0)]):
        train_x[ic].fillna(train_x[ic].median(), inplace=True)
    # df.isnull().any(axis=0) gives True/False flag (Boolean value series)

    return train_x, train_y
