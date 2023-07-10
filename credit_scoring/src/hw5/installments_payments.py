import os
import sys

import numpy as np
import pandas as pd

sys.path.append(r'C:\Users\user\Desktop\SHIFT\credit_scoring')

from src.app.utils.find_importance import find_importance


def main(path_to_file: str, path_to_save: str) -> None:
    """Функция для нахождения важных строк в файле фич installments_payments."""
    np.seterr(divide='ignore', invalid='ignore')
    alpha = 0.05

    train = pd.read_csv(os.path.join(path_to_file, 'application_train.csv'))
    train = train[['SK_ID_CURR', 'TARGET']]
    installments_payments = pd.read_csv(
        os.path.join(path_to_file, 'installments_payments.csv'),
    )
    installments_payments = pd.merge(
        installments_payments,
        train,
        how='inner',
        on='SK_ID_CURR',
    )
    installments_payments_copy = installments_payments.copy(deep=True)

    columns_for_tests = list(installments_payments_copy.columns)
    columns_for_tests.remove('SK_ID_CURR')
    columns_for_tests.remove('TARGET')
    columns_for_tests.remove('SK_ID_PREV')
    important_columns = find_importance(
        installments_payments_copy,
        columns_for_tests,
        alpha=alpha,
        tagret_name='TARGET',
    )
    important_columns.insert(0, 'TARGET')
    important_columns.insert(0, 'SK_ID_CURR')
    installments_payments = installments_payments[important_columns]
    installments_payments.to_csv(
        os.path.join(path_to_save, 'important_installments_payments_train.csv'), index=False,
    )


if __name__ == '__main__':
    path_to_file = r'C:\\Users\\user\\Desktop\\SHIFT\\credit_scoring\\data\\'
    path_to_save = r'C:\\Users\\user\\Desktop\\SHIFT\\credit_scoring\\data\\'
    main(path_to_file, path_to_save)
