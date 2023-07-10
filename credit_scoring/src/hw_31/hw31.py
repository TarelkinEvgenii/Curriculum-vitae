import datetime
import logging
import os
from dataclasses import asdict, dataclass

import pandas as pd


@dataclass
class PosCashBalanceIDs(object):
    """Датакласс для получения файла PosCashBalanceIDs."""

    SK_ID_PREV: int
    SK_ID_CURR: int
    NAME_CONTRACT_STATUS: str


@dataclass
class AmtCredit(object):
    """Датакласс для получения файла AmtCredit."""

    CREDIT_CURRENCY: str
    AMT_CREDIT_MAX_OVERDUE: float
    AMT_CREDIT_SUM: float
    AMT_CREDIT_SUM_DEBT: float
    AMT_CREDIT_SUM_LIMIT: float
    AMT_CREDIT_SUM_OVERDUE: float
    AMT_ANNUITY: float


PATH_TO_SOURCE_LOG_FILE = r'C:\\Users\\user\\Desktop\\SHIFT\\credit_scoring\\data\\'
PATH_TO_SAVE = r'C:\\Users\\user\\Desktop\\SHIFT\\credit_scoring\\data\\'


def main() -> None:
    """Функция для разделения склееного файла логов на два файла.

    Эти два файла должны соответствовать оригинальным файлам.
    """
    start_of_process = datetime.datetime.now()
    log_format = '[%(asctime)s] %(name)-25s %(levelname)-8s %(message)s'
    logging.basicConfig(
        format=log_format,
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    logger.info('Start reading log file')
    json_log = pd.read_json(
        os.path.join(
            PATH_TO_SOURCE_LOG_FILE,
            'POS_CASH_balance_plus_bureau-001.log',
        ),
        lines=True,
    )
    logger.info('Reading is done')

    logger.info('Splitting json data to different types')
    bureau = json_log.loc[json_log['type'] == 'bureau']
    pos_cash_balance = json_log.loc[json_log['type'] == 'POS_CASH_balance']

    logger.info('Start preprocessing bureau')
    logger.info('bureau data column normizile')
    bureau.reset_index(inplace=True, drop=True)
    bureau = normalize_column(bureau, 'data')

    logger.info('Changing bureau column names')
    dict_with_headers = dict(
        zip(
            list(bureau.columns),
            [element.replace('record.', '') for element in list(bureau.columns)],
        ),
    )
    bureau.rename(inplace=True, columns=dict_with_headers)

    logger.info('Start preprocessing AmtCredit column (eval)')
    bureau['AmtCredit'] = bureau['AmtCredit'].apply(lambda xl: eval(xl))
    logger.info('Converting to dataclass is done')

    logger.info('Converting AmtCredit column to dict')
    bureau['AmtCredit'] = bureau['AmtCredit'].apply(lambda xl: asdict(xl))

    logger.info('Normalizing AmtCredit column')
    bureau = normalize_column(bureau, 'AmtCredit')

    bureau.drop('type', axis=1, inplace=True)
    bureau = bureau[[
        'SK_ID_CURR',
        'SK_ID_BUREAU',
        'CREDIT_ACTIVE',
        'CREDIT_CURRENCY',
        'DAYS_CREDIT',
        'CREDIT_DAY_OVERDUE',
        'DAYS_CREDIT_ENDDATE',
        'DAYS_ENDDATE_FACT',
        'AMT_CREDIT_MAX_OVERDUE',
        'CNT_CREDIT_PROLONG',
        'AMT_CREDIT_SUM',
        'AMT_CREDIT_SUM_DEBT',
        'AMT_CREDIT_SUM_LIMIT',
        'AMT_CREDIT_SUM_OVERDUE',
        'CREDIT_TYPE',
        'DAYS_CREDIT_UPDATE',
        'AMT_ANNUITY',
    ]]
    logger.info('Start saving bureau.csv')
    bureau.to_csv(
        os.path.join(PATH_TO_SAVE, 'log_bureau.csv'),
        header=True,
        index=False,
    )
    logger.info('Saving bureau.csv is done')

    logger.info('Start preprocessing pos_cash_balance')
    logger.info('pos_cash_balance data column normizile')
    pos_cash_balance.reset_index(inplace=True, drop=True)
    pos_cash_balance = normalize_column(pos_cash_balance, 'data')

    logger.info('pos_cash_balance records column explode')
    pos_cash_balance = pos_cash_balance.explode('records')
    pos_cash_balance.reset_index(inplace=True, drop=True)

    logger.info('normalize records column')
    pos_cash_balance = normalize_column(pos_cash_balance, 'records')

    logger.info('Start converting PosCashBalanceIDs column to dataclass')
    pos_cash_balance['PosCashBalanceIDs'] = pos_cash_balance[
        'PosCashBalanceIDs'
    ].apply(
        lambda xl: eval(xl),
    )
    logger.info('Converting to PosCashBalanceIDs dataclass is done')

    logger.info('Converting PosCashBalanceIDs column to dict')
    pos_cash_balance['PosCashBalanceIDs'] = pos_cash_balance[
        'PosCashBalanceIDs'
    ].apply(lambda xl: asdict(xl))

    logger.info('Normalizing PosCashBalanceIDs column')
    pos_cash_balance = normalize_column(pos_cash_balance, 'PosCashBalanceIDs')

    pos_cash_balance.drop('type', axis=1, inplace=True)
    pos_cash_balance = pos_cash_balance[
        [
            'SK_ID_PREV',
            'SK_ID_CURR',
            'MONTHS_BALANCE',
            'CNT_INSTALMENT',
            'CNT_INSTALMENT_FUTURE',
            'NAME_CONTRACT_STATUS',
            'SK_DPD',
            'SK_DPD_DEF',
        ]
    ]

    logger.info('Start saving POS_CASH_balance.csv')
    pos_cash_balance.to_csv(
        os.path.join(PATH_TO_SAVE, 'log_POS_CASH_balance.csv'),
        header=True,
        index=False,
    )
    logger.info('Saving POS_CASH_balance.csv is done')

    logger.info(
        f'All process takes {datetime.datetime.now() - start_of_process}',
    )


def normalize_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Заменяет колонку со словарем на несколько колонок.

    Имена новых колонок - ключи словаря
    Значения в новых колонок - значения по заданному ключу в словаре
    """
    return pd.concat(
        [
            df,
            pd.json_normalize(df[column]),
        ],
        axis=1,
    ).drop(columns=[column])


if __name__ == '__main__':
    main()
