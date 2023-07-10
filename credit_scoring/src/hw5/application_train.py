import os
import sys

import numpy as np
import pandas as pd

sys.path.append(r'C:\Users\user\Desktop\SHIFT\credit_scoring')

from src.app.utils.find_importance import find_importance


def main(path_to_file: str, path_to_save: str) -> None:
    """Функция для нахождения важных строк в файле фич application_train."""
    np.seterr(divide='ignore', invalid='ignore')
    alpha = 0.05

    train = pd.read_csv(os.path.join(path_to_file, 'application_train.csv'))
    train_copy = train.copy(deep=True)
    columns_for_tests = list(train_copy.columns)
    columns_for_tests.remove('SK_ID_CURR')
    columns_for_tests.remove('TARGET')
    important_columns = find_importance(
        train_copy,
        columns_for_tests,
        alpha=alpha,
        tagret_name='TARGET',
    )
    important_columns.insert(0, 'TARGET')
    important_columns.insert(0, 'SK_ID_CURR')
    train = train[important_columns]
    train.to_csv(os.path.join(path_to_save, 'important_application_train.csv'), index=False)


if __name__ == '__main__':
    path_to_file = r'C:\\Users\\user\\Desktop\\SHIFT\\credit_scoring\\data\\'
    path_to_save = r'C:\\Users\\user\\Desktop\\SHIFT\\credit_scoring\\data\\'
    main(path_to_file, path_to_save)
