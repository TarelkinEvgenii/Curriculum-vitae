import logging
import os
import pickle

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm


def main(path_to_input: str, path_to_output: str) -> None:
    """Функция для блендинга."""
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

    # Так как catboost сам умеет работать с Nan,
    # то подготовка для catboost закончена
    data_catboost = train_x.copy(deep=True)

    # Заполним Nan
    train_x.replace([np.inf, -np.inf], np.nan, inplace=True)
    for ic in tqdm(train_x.columns[train_x.isnull().any(axis=0)]):
        train_x[ic].fillna(train_x[ic].median(), inplace=True)
    # df.isnull().any(axis=0) gives True/False flag (Boolean value series)

    # Для деревьев необязательно использовать StandartScaller,
    # поэтому предобработка для деревьев закончена
    data_forests = train_x.copy(deep=True)

    logger.info('Using of StandardScaler')
    scaler = StandardScaler()
    data_regression = pd.DataFrame(scaler.fit_transform(train_x))

    logger.info('Start training')
    np.random.seed(seed=1)  # for reproducibility
    my_cv = StratifiedKFold(n_splits=5)
    # Чтобы сохранить разбиение StratifiedKFold создадим массив индексов
    # и используем их разбиения датасета
    folds = pd.DataFrame(my_cv.split(data_regression, np.array(train_y).ravel()))[1]
    train_indicies = np.concatenate([folds[0], folds[1], folds[2]])
    holdout_indicies = folds[3]
    test_indicies = folds[4]

    # Инициализация лучших параметров алгоритмов, полученных на предыдущем шаге
    rf = RandomForestClassifier(
        n_estimators=60,
        max_depth=10,
        random_state=1,
    )
    dt = DecisionTreeClassifier(
        splitter='best',
        max_depth=6,
        random_state=0,
    )
    ctbst = CatBoostClassifier(
        depth=8,
        l2_leaf_reg=10,
        silent=True,
        border_count=254,
    )
    lr = LogisticRegression(
        penalty='l2',
        C=0.01,
        solver='newton-cholesky',
    )

    meta_model = clone(lr)

    logger.info('Getting features from test')
    logger.info('Random Forest training')
    rf.fit(
        data_forests.iloc[train_indicies],
        np.array(train_y.iloc[train_indicies]).ravel(),
    )
    logger.info('Decision Tree training')
    dt.fit(
        data_forests.iloc[train_indicies],
        np.array(train_y.iloc[train_indicies]).ravel(),
    )
    logger.info('Catboost training')
    ctbst.fit(
        data_catboost.iloc[train_indicies],
        np.array(train_y.iloc[train_indicies]).ravel(),
    )
    logger.info('Linear Regression training')
    lr.fit(
        data_regression.iloc[train_indicies],
        np.array(train_y.iloc[train_indicies]).ravel(),
    )

    logger.info('Getting predictions from test')
    dt_predictions_test = dt.predict_proba(data_forests.iloc[test_indicies])[:, 1]
    lr_predictions_test = lr.predict_proba(data_regression.iloc[test_indicies])[:, 1]
    ctbst_predictions_test = ctbst.predict_proba(data_catboost.iloc[test_indicies])[:, 1]
    rf_predictions_test = rf.predict_proba(data_forests.iloc[test_indicies])[:, 1]

    ctbst_predictions_holdout = ctbst.predict_proba(data_catboost.iloc[holdout_indicies])[:, 1]
    rf_predictions_holdout = rf.predict_proba(data_forests.iloc[holdout_indicies])[:, 1]
    dt_predictions_holdout = dt.predict_proba(data_forests.iloc[holdout_indicies])[:, 1]
    lr_predictions_holdout = lr.predict_proba(data_regression.iloc[holdout_indicies])[:, 1]

    prediction_test = pd.concat(
        [
            pd.DataFrame(ctbst_predictions_test),
            pd.DataFrame(rf_predictions_test),
            pd.DataFrame(dt_predictions_test),
            pd.DataFrame(lr_predictions_test),
        ],
        axis=1,
    )
    prediction_holdout = pd.concat(
        [
            pd.DataFrame(ctbst_predictions_holdout),
            pd.DataFrame(rf_predictions_holdout),
            pd.DataFrame(dt_predictions_holdout),
            pd.DataFrame(lr_predictions_holdout),
        ],
        axis=1,
    )

    meta_model.fit(
        np.array(prediction_holdout),
        np.array(train_y.iloc[holdout_indicies]).ravel(),
    )

    prediction_of_meta_model = meta_model.predict_proba(prediction_test)[:, 1]
    logger.info(
        'Meta model roc_auc: {}'.format(
            roc_auc_score(
                np.array(train_y.iloc[test_indicies]).ravel(),
                prediction_of_meta_model,
            ),
        ),
    )

    logger.info('Saving meta model')
    with open(
        os.path.join(
            path_to_output,
            'hw_7_meta_model_blending.pkl',
        ), 'wb',
    ) as my_file:
        pickle.dump(meta_model, my_file)
    logger.info('Model saved')


if __name__ == '__main__':
    path_to_input = '../../../data/'
    path_to_output = '.\\'
    main(path_to_input, path_to_output)
