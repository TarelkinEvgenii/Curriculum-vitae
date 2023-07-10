import pandas as pd


def main() -> None:
    """Функция генерации фич из файла bureau."""
    bureau = pd.read_csv(r'C:\Users\user\Desktop\SHIFT\credit_scoring\data\bureau.csv')

    # Для датафрейма bureau.csv посчитать следующие признаки:
    # 1. Максимальная сумма просрочки
    features_bureau = bureau.groupby('SK_ID_CURR').agg({'AMT_CREDIT_SUM_DEBT': ['max']})
    features_bureau.columns = pd.Index(
        ['BUREAU_' + e[0] + '_' + e[1].upper() for e in features_bureau.columns.tolist()],
    )

    # 2. Минимальная сумма просрочки
    question_2_data = bureau.groupby('SK_ID_CURR').agg({'AMT_CREDIT_SUM_DEBT': ['min']})
    question_2_data.columns = pd.Index(
        ['BUREAU_' + e[0] + '_' + e[1].upper() for e in question_2_data.columns.tolist()],
    )
    features_bureau = pd.concat([features_bureau, question_2_data], axis=1)

    # 3. Какую долю суммы от открытого займа просрочил
    question_3_1 = bureau.loc[bureau.CREDIT_ACTIVE == 'Active']
    question_3_1 = question_3_1.groupby('SK_ID_CURR').agg({
        'AMT_CREDIT_SUM': ['sum'],
        'AMT_CREDIT_SUM_DEBT': ['sum'],
    })
    question_3_1.columns = pd.Index(
        [e[0] for e in question_3_1.columns.tolist()],
    )
    question_3_1['DEBT_RATIO'] = \
        question_3_1['AMT_CREDIT_SUM_DEBT'] / question_3_1['AMT_CREDIT_SUM']

    features_bureau = pd.concat([features_bureau, question_3_1[['DEBT_RATIO']]], axis=1)

    # 4. Кол-во кредитов определенного типа
    question_4_data = pd.get_dummies(
        bureau[['SK_ID_CURR', 'CREDIT_TYPE']],
        prefix=['CREDIT_TYPE'],
        columns=['CREDIT_TYPE'],
    )
    question_4_data = question_4_data.groupby('SK_ID_CURR').sum()
    features_bureau = pd.concat([features_bureau, question_4_data], axis=1)

    # 5. Кол-во просрочек кредитов определенного типа
    question_5_data = pd.pivot_table(
        bureau,
        index='SK_ID_CURR',
        columns=['CREDIT_TYPE'],
        values=['AMT_CREDIT_SUM_DEBT'],
        aggfunc='count',
    )
    question_5_data.columns = pd.Index(
        ['QUESTION_5_' + e[1].upper() for e in question_5_data.columns.tolist()],
    )
    features_bureau = pd.concat([features_bureau, question_5_data], axis=1)

    # 6. Кол-во закрытых кредитов определенного типа
    question_6_data = pd.pivot_table(
        bureau,
        index=['SK_ID_CURR', 'CREDIT_ACTIVE'],
        columns=["CREDIT_TYPE"],
        aggfunc='count',
    )
    question_6_data.columns = pd.Index(
        ['QUESTION_6_' + ec[1].upper() for ec in question_6_data.columns.tolist()],
    )
    question_6_data.reset_index(inplace=True)
    question_6_data = question_6_data[question_6_data['CREDIT_ACTIVE'] == 'Closed']
    question_6_data.drop(columns='CREDIT_ACTIVE', axis=1, inplace=True)
    features_bureau.reset_index(inplace=True)

    features_bureau = pd.merge(
        features_bureau, question_6_data,
        how='left',
        left_on='SK_ID_CURR',
        right_on='SK_ID_CURR',
    )

    # Для датафрейма bureau_balance.csv посчитать следующие признаки:
    # 1. Кол-во открытых кредитов
    # 2. Кол-во закрытых кредитов
    question_1_2_data = pd.get_dummies(
        bureau[['SK_ID_CURR', 'CREDIT_ACTIVE']],
        prefix=['CREDIT_ACTIVE'],
        columns=['CREDIT_ACTIVE'],
    )
    question_1_2_data = question_1_2_data.groupby('SK_ID_CURR').sum()
    question_1_2_data = question_1_2_data[['CREDIT_ACTIVE_Active', 'CREDIT_ACTIVE_Closed']]
    question_1_2_data.reset_index(inplace=True)
    features_bureau = pd.merge(
        features_bureau, question_1_2_data,
        how='left',
        left_on='SK_ID_CURR',
        right_on='SK_ID_CURR',
    )

    # 4. Кол-во кредитов
    question_4_2_data = pd.get_dummies(
        bureau[['SK_ID_CURR', 'CREDIT_ACTIVE']],
        prefix=['CREDIT_ACTIVE'],
        columns=['CREDIT_ACTIVE'],
    )
    question_4_2_data = question_4_2_data.groupby('SK_ID_CURR').sum()
    question_4_2_data = pd.DataFrame(question_4_2_data.sum(axis=1))
    question_4_2_data.rename({'0': 'AMOUNT_OF_CREDITS'})
    question_4_2_data.rename({0: 'AMOUNT_OF_CREDITS'}, axis=1, inplace=True)
    question_4_2_data.reset_index(inplace=True)
    features_bureau = pd.merge(
        features_bureau,
        question_4_2_data,
        how='left',
        left_on='SK_ID_CURR',
        right_on='SK_ID_CURR',
    )

    # 5. Доля закрытых кредитов
    # 6. Доля открытых кредитов
    question_5_2_data = pd.get_dummies(
        bureau[['SK_ID_CURR', 'CREDIT_ACTIVE']],
        prefix=['CREDIT_ACTIVE'], columns=['CREDIT_ACTIVE'],
    )
    question_5_2_data = question_5_2_data.groupby('SK_ID_CURR').sum()
    question_5_2_data.reset_index(inplace=True)
    question_5_2_data = pd.merge(question_5_2_data, question_4_2_data, on='SK_ID_CURR')

    question_5_2_data['RATIO_OF_ACTIVE'] = \
        question_5_2_data['CREDIT_ACTIVE_Active'] / question_5_2_data['AMOUNT_OF_CREDITS']
    question_5_2_data['RATIO_OF_CLOSED'] = \
        question_5_2_data['CREDIT_ACTIVE_Closed'] / question_5_2_data['AMOUNT_OF_CREDITS']

    question_5_2_data = question_5_2_data[['SK_ID_CURR', 'RATIO_OF_ACTIVE', 'RATIO_OF_CLOSED']]
    features_bureau = pd.merge(
        features_bureau, question_5_2_data,
        how='left',
        left_on='SK_ID_CURR',
        right_on='SK_ID_CURR',
    )

    # 3. Кол-во просроченных кредитов по разным дням просрочки
    # (смотреть дни по колонке STATUS)
    bureau_balance = pd.read_csv(
        r'C:\Users\user\Desktop\SHIFT\home-credit-default-risk\bureau_balance.csv',
    )
    bureau_balance_copy = bureau_balance.copy(deep=True)
    bureau_balance_copy.loc[bureau_balance_copy.STATUS == 'C', 'STATUS'] = -1
    bureau_balance_copy.loc[bureau_balance_copy.STATUS == 'X', 'STATUS'] = -2
    bureau_balance_copy['STATUS'] = bureau_balance_copy['STATUS'].astype('int')
    bureau_balance_copy = bureau_balance_copy.groupby('SK_ID_BUREAU').agg({'STATUS': ['max']})
    bureau_balance_copy.columns = pd.Index(
        [
            'BUREAU_BALANCE' + '_' + ec[0] + '_' + ec[1].upper() for ec in
            bureau_balance_copy.columns.tolist()
        ],
    )
    bureau_balance_copy.reset_index(inplace=True)
    question_3_2_data = pd.get_dummies(
        bureau_balance_copy[[
            'SK_ID_BUREAU',
            'BUREAU_BALANCE_STATUS_MAX',
        ]],
        prefix=['STATUS_MAX_'],
        columns=['BUREAU_BALANCE_STATUS_MAX'],
    )

    question_3_2_data = question_3_2_data[[
        'SK_ID_BUREAU',
        'STATUS_MAX__0',
        'STATUS_MAX__1',
        'STATUS_MAX__2',
        'STATUS_MAX__3',
        'STATUS_MAX__4',
        'STATUS_MAX__5',
    ]]
    question_3_2_merge = pd.merge(
        left=bureau,
        right=question_3_2_data,
        how='left',
        on='SK_ID_BUREAU',
    )

    question_3_2_merge = question_3_2_merge.groupby('SK_ID_CURR').agg({
        'STATUS_MAX__0': ['sum'],
        'STATUS_MAX__1': ['sum'],
        'STATUS_MAX__2': ['sum'],
        'STATUS_MAX__3': ['sum'],
        'STATUS_MAX__4': ['sum'],
        'STATUS_MAX__5': ['sum'],
    })
    question_3_2_merge.columns = pd.Index(
        [ec[0] for ec in question_3_2_merge.columns.tolist()],
    )
    question_3_2_merge.reset_index(inplace=True)

    features_bureau = pd.merge(
        features_bureau,
        question_3_2_merge,
        how='left',
        left_on='SK_ID_CURR',
        right_on='SK_ID_CURR',
    )

    # 7. Доля просроченных кредитов по разным дням просрочки (смотреть дни по колонке STATUS)
    question_7_2_merge = pd.merge(
        left=bureau,
        right=question_3_2_data,
        how='left',
        on='SK_ID_BUREAU',
    )
    question_7_2_merge['amount_of_credits'] = 1
    question_7_2_merge = question_7_2_merge.groupby('SK_ID_CURR').agg({
        'STATUS_MAX__0': ['sum'],
        'STATUS_MAX__1': ['sum'],
        'STATUS_MAX__2': ['sum'],
        'STATUS_MAX__3': ['sum'],
        'STATUS_MAX__4': ['sum'],
        'STATUS_MAX__5': ['sum'],
        'amount_of_credits': ['sum'],
    })

    question_7_2_merge.columns = pd.Index(
        [ec[0] for ec in question_7_2_merge.columns.tolist()],
    )

    question_7_2_merge['RATIO_STATUS_0'] = \
        question_7_2_merge['STATUS_MAX__0'] / question_7_2_merge['amount_of_credits']
    question_7_2_merge['RATIO_STATUS_1'] = \
        question_7_2_merge['STATUS_MAX__1'] / question_7_2_merge['amount_of_credits']
    question_7_2_merge['RATIO_STATUS_2'] = \
        question_7_2_merge['STATUS_MAX__2'] / question_7_2_merge['amount_of_credits']
    question_7_2_merge['RATIO_STATUS_3'] = \
        question_7_2_merge['STATUS_MAX__3'] / question_7_2_merge['amount_of_credits']
    question_7_2_merge['RATIO_STATUS_4'] = \
        question_7_2_merge['STATUS_MAX__4'] / question_7_2_merge['amount_of_credits']
    question_7_2_merge['RATIO_STATUS_5'] = \
        question_7_2_merge['STATUS_MAX__5'] / question_7_2_merge['amount_of_credits']

    question_7_2_merge.reset_index(inplace=True)
    question_7_2_merge = question_7_2_merge[[
        'SK_ID_CURR',
        'RATIO_STATUS_0',
        'RATIO_STATUS_1',
        'RATIO_STATUS_2',
        'RATIO_STATUS_3',
        'RATIO_STATUS_4',
        'RATIO_STATUS_5',
    ]]
    features_bureau = pd.merge(
        features_bureau,
        question_7_2_merge,
        how='left',
        left_on='SK_ID_CURR',
        right_on='SK_ID_CURR',
    )

    # 8. Интервал между последним закрытым кредитом и текущей заявкой
    question_8_2_bureau = bureau[
        ['SK_ID_CURR', 'CREDIT_ACTIVE', 'DAYS_CREDIT', 'DAYS_CREDIT_ENDDATE', 'DAYS_ENDDATE_FACT']
    ]

    current_application = question_8_2_bureau.groupby(['SK_ID_CURR', 'CREDIT_ACTIVE']).agg(
        {'DAYS_CREDIT': ['max']},
    )
    current_application.columns = pd.Index(
        [ec[0] + '_' + ec[1].upper() for ec in current_application.columns.tolist()],
    )
    current_application.reset_index(inplace=True)
    current_application = current_application.loc[current_application.CREDIT_ACTIVE == 'Active']
    current_application.drop(columns='CREDIT_ACTIVE', axis=1, inplace=True)

    newest_closed_credit = question_8_2_bureau.groupby(['SK_ID_CURR', 'CREDIT_ACTIVE']).agg(
        {'DAYS_CREDIT_ENDDATE': ['max']},
    )
    newest_closed_credit.columns = pd.Index(
        [ec[0] + '_' + ec[1].upper() for ec in newest_closed_credit.columns.tolist()],
    )
    newest_closed_credit.reset_index(inplace=True)
    newest_closed_credit = newest_closed_credit.loc[newest_closed_credit.CREDIT_ACTIVE == 'Closed']
    newest_closed_credit.drop(columns='CREDIT_ACTIVE', axis=1, inplace=True)

    merged_for_question_8 = pd.merge(
        current_application,
        newest_closed_credit,
        how='inner',
        on='SK_ID_CURR',
    )

    merged_for_question_8['CLOSED_MINUS_CURRENT'] = merged_for_question_8[
                                                        'DAYS_CREDIT_ENDDATE_MAX'
                                                    ] - merged_for_question_8['DAYS_CREDIT_MAX']

    merged_for_question_8 = merged_for_question_8[['SK_ID_CURR', 'CLOSED_MINUS_CURRENT']]

    features_bureau = pd.merge(features_bureau, merged_for_question_8, how='left', on='SK_ID_CURR')

    # 9. Интервал между взятием последнего активного займа и текущей заявкой
    last_active_credit = question_8_2_bureau.groupby(
        ['SK_ID_CURR', 'CREDIT_ACTIVE'],
    ).agg({'DAYS_CREDIT': ['min']})
    last_active_credit.columns = pd.Index(
        [ec[0] + '_' + ec[1].upper() for ec in last_active_credit.columns.tolist()],
    )
    last_active_credit.reset_index(inplace=True)
    last_active_credit = last_active_credit.loc[last_active_credit.CREDIT_ACTIVE == 'Active']
    last_active_credit.drop(columns='CREDIT_ACTIVE', axis=1, inplace=True)

    merged_for_question_9 = pd.merge(
        current_application,
        last_active_credit,
        how='inner',
        on='SK_ID_CURR',
    )

    merged_for_question_9['QUESTION_9_BUREAU'] = \
        merged_for_question_9['DAYS_CREDIT_MIN'] - merged_for_question_9['DAYS_CREDIT_MAX']

    merged_for_question_9 = merged_for_question_9[['SK_ID_CURR', 'QUESTION_9_BUREAU']]

    features_bureau = pd.merge(features_bureau, merged_for_question_9, how='left', on='SK_ID_CURR')

    features_bureau.to_csv(
        r'C:\Users\user\Desktop\SHIFT\credit_scoring\data\features_bureau.csv',
        index=False,
    )


if __name__ == '__main__':
    main()
