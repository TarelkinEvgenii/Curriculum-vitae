import logging
import os
import pickle

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from tqdm import tqdm


def main(path_to_input: str, path_to_output: str) -> None:
    """Функция для применения catboost."""
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
        os.path.join(path_to_input, 'features_installments_payments.csv'),
    )
    previous_application = pd.read_csv(
        os.path.join(path_to_input, 'features_previous_application.csv'),
    )
    application_train = pd.read_csv(
        os.path.join(path_to_input, 'features_application_train.csv'),
    )

    # importing original train file to get TARGER values
    train_original = pd.read_csv(
        os.path.join(path_to_input, 'application_train.csv'),
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

    logger.info('Initialization of parameters for training')

    depth = pd.DataFrame(np.arange(4, 11, 1), columns=['depth'])
    l2_leaf_reg = pd.DataFrame(np.logspace(-1, 2, 4), columns=['l2_leaf_reg'])

    parameter_search = pd.merge(depth, l2_leaf_reg, how='cross')

    logger.info('Start training')
    np_train_x = np.array(train_x)
    train_y = np.array(train_y).ravel()
    my_cv = StratifiedKFold(n_splits=5)
    all_scores = []
    for my_params in tqdm(parameter_search.values):
        clf = CatBoostClassifier(
            depth=my_params[0],
            l2_leaf_reg=my_params[1],
            silent=True,
            task_type='GPU',
            devices='0:1',
            border_count=254,
        )
        scores = cross_val_score(
            clf,
            np_train_x,
            train_y,
            cv=my_cv,
            scoring='roc_auc',
        )
        all_scores.append(np.mean(scores))

    my_result = pd.concat(
        [parameter_search, pd.DataFrame(all_scores, columns=['roc_auc'])],
        axis=1,
    ).sort_values('roc_auc', ascending=False)

    logger.info('Feature scoring')
    logger.info('\n' + str(my_result.head()))

    clf = CatBoostClassifier(
        depth=my_result.iloc[0].iloc[0],
        l2_leaf_reg=my_result.iloc[0].iloc[1],
        silent=True,
        task_type='GPU',
        devices='0:1',
        border_count=254,
    )
    clf.fit(np_train_x, train_y)

    logger.info('Saving model')
    with open(
        os.path.join(
            path_to_output,
            'hw_7_catboost_model.pkl',
        ), 'wb',
    ) as myfile:
        pickle.dump(clf, myfile)
    logger.info('Model saved')


if __name__ == '__main__':
    path_to_input = 'C:\\Users\\user\\Desktop\\SHIFT\\credit_scoring\\data\\'
    path_to_output = 'C:\\Users\\user\\Desktop\\SHIFT\\credit_scoring\\data\\'
    main(path_to_input, path_to_output)
