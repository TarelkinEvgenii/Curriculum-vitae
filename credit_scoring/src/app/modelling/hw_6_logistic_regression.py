import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.app.utils.read_generated_features import read_train_generated_features


def main(path_to_input: str, path_to_output: str) -> None:
    """Функция для logistic regression."""
    log_format = '[%(asctime)s] %(name)-25s %(levelname)-8s %(message)s'
    logging.basicConfig(
        format=log_format,
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    train_x, train_y = read_train_generated_features(path_to_input)

    # Надо чтобы обучалась логистическая регрессия совместно со StandartScaler
    pipe_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(penalty='l2', solver='newton-cholesky')),
    ])

    parameters = [{'clf__C': list(np.logspace(-2, 4, 7))}]
    my_cv = StratifiedKFold(n_splits=5)

    grid_search = GridSearchCV(
        estimator=pipe_lr,
        param_grid=parameters,
        scoring='roc_auc',
        cv=my_cv,
        verbose=2,
        refit=True,
    )

    train_y = np.array(train_y).ravel()
    grid_search.fit(train_x, train_y)

    logger.info(f'Best searched params are : {grid_search.best_params_}')

    logger.info(f'Best model params: {grid_search.best_estimator_.get_params()}')
    # Best training data roc_auc
    logger.info(f'Best training roc_auc: {grid_search.best_score_}')

    best_pipeline = grid_search.best_estimator_

    feature_importance = pd.DataFrame(
        np.abs(best_pipeline['clf'].coef_[0]),
        columns=['np.abs(feature_importance)'],
    )
    feature_importance = pd.concat(
        [
            feature_importance,
            pd.DataFrame(
                train_x.columns,
                columns=['column_name'],
            ),
        ],
        axis=1,
    ).sort_values('np.abs(feature_importance)',
                  ascending=False,
                  ).reset_index(drop=True)
    logger.info('Top 10 features')
    logger.info('\n' + str(feature_importance.head(10)))

    logger.info('Saving model')
    # Сохранять модель надо тоже с обученным StandardScaler
    # Поэтому сохраним обученный пайплайн
    with open(
        os.path.join(
            path_to_output,
            'hw_6_regression_model.pkl',
        ), 'wb',
    ) as f:
        pickle.dump(best_pipeline, f)
    logger.info('Model saved')


if __name__ == '__main__':
    path_to_input = '.\\'
    path_to_output = '.\\'
    main(path_to_input, path_to_output)
