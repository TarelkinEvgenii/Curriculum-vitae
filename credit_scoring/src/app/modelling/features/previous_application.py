from typing import Tuple

import numpy as np
import pandas as pd


# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df: pd.DataFrame, nan_as_category: bool = True) -> Tuple[pd.DataFrame, list]:
    """Функция для OHE."""
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [cc for cc in df.columns if cc not in original_columns]
    return df, new_columns


def main() -> None:
    """Функция генерации фич из файла previous_application."""
    previous = pd.read_csv(
        r'C:\Users\user\Desktop\SHIFT\credit_scoring\data\previous_application.csv',
    )
    previous, cat_cols = one_hot_encoder(previous, nan_as_category=True)
    # Days 365.243 values -> nan
    previous['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    previous['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    previous['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    previous['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    previous['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
    # Add feature: value ask / value received percentage
    previous['APP_CREDIT_PERC'] = previous['AMT_APPLICATION'] / previous['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['max', 'mean'],
        'AMT_APPLICATION': ['max', 'mean'],
        'AMT_CREDIT': ['max', 'mean'],
        'APP_CREDIT_PERC': ['max', 'mean'],
        'AMT_DOWN_PAYMENT': ['max', 'mean'],
        'AMT_GOODS_PRICE': ['max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['max', 'mean'],
        'RATE_DOWN_PAYMENT': ['max', 'mean'],
        'DAYS_DECISION': ['max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = previous.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(
        ['PREV_' + ec[0] + '_' + ec[1].upper() for ec in prev_agg.columns.tolist()],
    )
    # Previous Applications: Approved Applications - only numerical features
    approved = previous[previous['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(
        ['APPROVED_' + ec[0] + '_' + ec[1].upper() for ec in approved_agg.columns.tolist()],
    )
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = previous[previous['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(
        ['REFUSED_' + ec[0] + '_' + ec[1].upper() for ec in refused_agg.columns.tolist()],
    )
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    prev_agg.reset_index(inplace=True)
    prev_agg.to_csv(
        r'C:\Users\user\Desktop\SHIFT\credit_scoring\data\features_previous_application.csv',
        index=False,
    )


if __name__ == '__main__':
    main()
