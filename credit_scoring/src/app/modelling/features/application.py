import numpy as np
import pandas as pd


def main() -> None:
    """Функция генерации фич из файла application."""
    path = r'C:\Users\user\Desktop\SHIFT\credit_scoring\data\application_test.csv'

    df_test = pd.read_csv(path)
    features_application = df_test[['SK_ID_CURR']]

    # 	1. Кол-во документов
    features_application['DOCUMENTS_PROVIDED'] = df_test[
        [f'FLAG_DOCUMENT_{i}' for i in range(2, 22)]
    ].sum(axis=1)
    # 2. Есть ли полная информация о доме.
    # Найдите все колонки, которые описывают характеристики дома
    # и посчитайте кол-во непустых характеристик.
    # Если кол-во пропусков меньше 30, то значение признака 1. Иначе 0
    list_of_column_names = [
        'APARTMENTS_AVG',
        'BASEMENTAREA_AVG',
        'YEARS_BEGINEXPLUATATION_AVG',
        'YEARS_BUILD_AVG',
        'COMMONAREA_AVG',
        'ELEVATORS_AVG',
        'ENTRANCES_AVG',
        'FLOORSMAX_AVG',
        'FLOORSMIN_AVG',
        'LANDAREA_AVG',
        'LIVINGAPARTMENTS_AVG',
        'LIVINGAREA_AVG',
        'NONLIVINGAPARTMENTS_AVG',
        'NONLIVINGAREA_AVG',
        'APARTMENTS_MODE',
        'BASEMENTAREA_MODE',
        'YEARS_BEGINEXPLUATATION_MODE',
        'YEARS_BUILD_MODE',
        'COMMONAREA_MODE',
        'ELEVATORS_MODE',
        'ENTRANCES_MODE',
        'FLOORSMAX_MODE',
        'FLOORSMIN_MODE',
        'LANDAREA_MODE',
        'LIVINGAPARTMENTS_MODE',
        'LIVINGAREA_MODE',
        'NONLIVINGAPARTMENTS_MODE',
        'NONLIVINGAREA_MODE',
        'APARTMENTS_MEDI',
        'BASEMENTAREA_MEDI',
        'YEARS_BEGINEXPLUATATION_MEDI',
        'YEARS_BUILD_MEDI',
        'COMMONAREA_MEDI',
        'ELEVATORS_MEDI',
        'ENTRANCES_MEDI',
        'FLOORSMAX_MEDI',
        'FLOORSMIN_MEDI',
        'LANDAREA_MEDI',
        'LIVINGAPARTMENTS_MEDI',
        'LIVINGAREA_MEDI',
        'NONLIVINGAPARTMENTS_MEDI',
        'NONLIVINGAREA_MEDI',
        'FONDKAPREMONT_MODE',
        'HOUSETYPE_MODE',
        'TOTALAREA_MODE',
        'WALLSMATERIAL_MODE',
        'EMERGENCYSTATE_MODE',
    ]
    question_2_data = df_test[list_of_column_names].isna()
    # Если пропуск, то 1, если не пропуск, то 0
    question_2_data = question_2_data.where(question_2_data is False, 1).where(
        question_2_data is True, 0,
    ).sum(axis=1)
    features_application['HOME_FEATURE'] = question_2_data.where(
        question_2_data >= 30, 1,
    ).where(question_2_data < 30, 0).astype('int')

    # 3. Кол-во полных лет
    features_application['YEARS_OLD'] = pd.DataFrame(
        np.floor(-df_test['DAYS_BIRTH'] / 365),
    ).astype('int')

    # 4. Год смены документа (Под годом подразумевается возраст.)
    # (у нас ведь нет конкретной даты подачи заявки.)
    features_application['YEARS_ID_CHANGED'] = pd.DataFrame(
        np.floor(-df_test['DAYS_ID_PUBLISH'] / 365),
    ).astype('int')

    # 5. Разница во времени между сменой документа и возрастом на момент
    # смены документы (фактическая смена, а не когда был должен)
    # Эквивалентно возрасту смены документа
    features_application['DIFFERENCE_ID_CHANGE_AND_OLDNESS'] = pd.DataFrame(
        np.floor((
                         df_test['DAYS_ID_PUBLISH'] - df_test['DAYS_BIRTH']
                 ) / 365,
                 ),
    ).astype('int')

    # 6. Признак задержки смены документа.
    # Документ выдается или меняется в 14, 20 и 45 лет
    question_6_data = features_application['DIFFERENCE_ID_CHANGE_AND_OLDNESS']

    features_application['DOCUMENT_CHANGE_DELAY'] = np.logical_or(
        np.logical_or(
            np.logical_and(question_6_data > 14, question_6_data < 20),
            np.logical_and(question_6_data > 20, question_6_data < 45),
        ),
        question_6_data > 45,
    )

    # 7. Доля денег которые клиент отдает на займ за год
    features_application['ANNUITY_TO_INCOME_RATIO'] = df_test[
        'AMT_ANNUITY'] / df_test['AMT_INCOME_TOTAL']

    # 8. Кол-во детей в семье на одного взрослого
    # Примечание: Так как в CNT_FAM_MEMBERS нет нулей, а только числа > 1, то доп проверок не надо
    features_application['CHILDREN_PER_ADULT'] = df_test['CNT_CHILDREN'] / (
        df_test['CNT_FAM_MEMBERS'] - df_test['CNT_CHILDREN']
    )

    # 9. Доход на ребенка
    features_application['INC_PER_CHLD'] = df_test['AMT_INCOME_TOTAL'] / (
        1 + df_test['CNT_CHILDREN']
    )

    # 10. Доход на взрослого
    features_application['INC_PER_ADULT'] = df_test['AMT_INCOME_TOTAL'] / (
            df_test['CNT_FAM_MEMBERS'] - df_test['CNT_CHILDREN']
    )

    # 11. Процентная ставка
    # Версия 1
    features_application['INTEREST_RATE_v1'] = df_test['AMT_ANNUITY'] / df_test['AMT_CREDIT']
    # Версия 2
    # найти для конкретного кредита сколько месяцев он платится - пусть это CNT_payments
    # И по формуле
    # (AMT_ANNUITY * CNT_payments) /  AMT_CREDIT = (1 + INTEREST_RATE) ^ (CNT_payments/12)
    # Вычислить INTEREST_RATE
    bureau = pd.read_csv(r'C:\Users\user\Desktop\SHIFT\home-credit-default-risk\bureau.csv')
    bureau_balance = pd.read_csv(
        r'C:\Users\user\Desktop\SHIFT\home-credit-default-risk\bureau_balance.csv',
    )
    bureau_balance = bureau_balance.groupby('SK_ID_BUREAU').count()
    bureau_balance.reset_index(inplace=True)
    bureau_balance.drop(columns='STATUS', axis=1, inplace=True)
    merged_bureau_and_bureau_balance = pd.merge(
        left=bureau,
        right=bureau_balance,
        how='left',
        on='SK_ID_BUREAU',
    )

    merged_bureau_and_bureau_balance = merged_bureau_and_bureau_balance[[
        'SK_ID_CURR',
        'MONTHS_BALANCE',
    ]]

    merged_bureau_and_bureau_balance = merged_bureau_and_bureau_balance.groupby('SK_ID_CURR').sum()
    merged_bureau_and_bureau_balance.reset_index(inplace=True)

    merged_bureau_and_bureau_balance.loc[
        merged_bureau_and_bureau_balance.MONTHS_BALANCE == 0, 'MONTHS_BALANCE',
    ] = np.nan

    merged_test_bureau = pd.merge(
        left=df_test,
        right=merged_bureau_and_bureau_balance,
        how='left',
        on='SK_ID_CURR',
    )

    # И по формуле
    # (AMT_ANNUITY * CNT_payments) /  AMT_CREDIT = (1 + INTEREST_RATE) ^ (CNT_payments/12)
    features_application['INTEREST_RATE_v2'] = np.power(
        ((
          merged_test_bureau['AMT_ANNUITY'] * merged_test_bureau['MONTHS_BALANCE']
         ) /
         merged_test_bureau['AMT_CREDIT']),
        1 / (merged_test_bureau['MONTHS_BALANCE'] / 12
             ),
    ) - 1

    # 12. Взвешенный скор внешних источников. Подумайте какие веса им задать.
    # Я бы попробовал вычислить систему линейных выражений
    # w1 * EXT_SOURCE_1 + w2 * EXT_SOURCE_1 + w3 * EXT_SOURCE_1 = TARGET
    # но оперативки не хватило для этого и не понятно можно ли трейн использовать или нет
    # Поэтому задам следующим образом,
    # так как у нас пустых в EXT_SOURCE_1 примерно 42%,
    # а EXT_SOURCE_3 17.8%, а в EXT_SOURCE_2 0%,
    # то увеличу влияние, там где больше пропущенных
    features_application['WEIGHTED_EXT_SOURCE'] = 0.58 * df_test['EXT_SOURCE_1'] + 0.42 * df_test[
        'EXT_SOURCE_1'
    ] + 0 * df_test['EXT_SOURCE_1']

    # 13. Поделим людей на группы в зависимости от пола и образования.
    # В каждой группе посчитаем средний доход.
    # Сделаем признак разница между средним доходом в группе и доходом заявителя

    question_13_data = df_test.groupby(by=['CODE_GENDER', 'NAME_EDUCATION_TYPE']).agg(['mean'])
    question_13_data = question_13_data.loc[:, ('AMT_INCOME_TOTAL', 'mean')]

    question_13_data_v2 = pd.DataFrame(
        question_13_data.loc[list(zip(df_test['CODE_GENDER'], df_test['NAME_EDUCATION_TYPE']))],
    )
    question_13_data_v2.columns = pd.Index(
        [e[0] + '_' + e[1].upper() for e in question_13_data_v2.columns.tolist()],
    )
    question_13_data_v2.reset_index(inplace=True, drop=True)
    features_application[
        'DIFFERENCE_OF_INCOME'
    ] = question_13_data_v2['AMT_INCOME_TOTAL_MEAN'] - df_test['AMT_INCOME_TOTAL']

    features_application.to_csv(
        r'C:\Users\user\Desktop\SHIFT\credit_scoring\data\features_application_test.csv',
        index=False,
    )


if __name__ == '__main__':
    main()
