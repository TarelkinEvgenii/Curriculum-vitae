import pandas as pd
from scipy.stats import anderson, mannwhitneyu, ttest_ind
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def find_importance(
    df_dataset: pd.DataFrame,
    columns_for_tests: list,
    alpha: float,
    tagret_name: str,
) -> list:
    """Функция, которая находит значимые столбцы."""
    important_columns = []
    for column in tqdm(columns_for_tests):
        if df_dataset[column].dtype == 'object':
            le = LabelEncoder()
            df_dataset[column] = le.fit_transform(df_dataset[column])

        stat, crit_val, sign_level = anderson(
            df_dataset[df_dataset[tagret_name] == 0][column], 'norm',
        )

        stat_2, crit_val_2, sign_level_2 = anderson(
            df_dataset[df_dataset[tagret_name] == 1][column], 'norm',
        )

        if stat >= crit_val[4] or stat_2 >= crit_val_2[4]:
            # распределено не нормально
            _, p_mw = mannwhitneyu(
                df_dataset[df_dataset[tagret_name] == 0][column],
                df_dataset[df_dataset[tagret_name] == 1][column],
            )
            if p_mw < alpha:
                # Признак значимый
                important_columns.append(column)
        else:
            # распределены нормально
            _, p_tt = ttest_ind(
                df_dataset[df_dataset[tagret_name] == 0][column],
                df_dataset[df_dataset[tagret_name] == 1][column],
            )
            if p_tt < alpha:
                # Признак значимый
                important_columns.append(column)
    return important_columns
