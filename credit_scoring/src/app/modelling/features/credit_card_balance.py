import pandas as pd


def main() -> None:
    """Функция генерации фич из файла credit_card_balance."""
    cc = pd.read_csv(
        r'C:\Users\user\Desktop\SHIFT\credit_scoring\data\credit_card_balance.csv',
    )
    cc.drop(['SK_ID_PREV'], axis=1, inplace=True)

    cc_agg = cc.groupby('SK_ID_CURR').agg(['max', 'min', 'mean', 'median', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + ec[0] + '_' + ec[1].upper() for ec in cc_agg.columns.tolist()])

    cc_v2 = cc[cc.MONTHS_BALANCE >= -3]
    cc_agg_v2 = cc_v2.groupby('SK_ID_CURR').agg(['max', 'min', 'mean', 'median', 'sum', 'var'])
    cc_agg_v2.columns = pd.Index(
        ['CC_' + ec[0] + '_' + ec[1].upper() for ec in cc_agg_v2.columns.tolist()],
    )

    cc_agg.reset_index(inplace=True)
    cc_agg_v2.reset_index(inplace=True)
    cc_agg_for_subtraction = cc_agg.loc[cc_agg.SK_ID_CURR.isin(cc_agg_v2['SK_ID_CURR'])]

    difference_between_agg = cc_agg_for_subtraction.set_index('SK_ID_CURR').subtract(
        cc_agg_v2.set_index('SK_ID_CURR'),
    )
    difference_between_agg.columns = pd.Index(
        ['DIFF_' + ec for ec in difference_between_agg.columns.tolist()],
    )

    difference_between_agg.reset_index(inplace=True)

    merged_agg = pd.merge(
        cc_agg,
        difference_between_agg,
        how='left',
        on='SK_ID_CURR',
    )

    merged_agg.to_csv(
        r'C:\Users\user\Desktop\SHIFT\credit_scoring\data\features_credit_card_balance.csv',
        index=False,
    )


if __name__ == '__main__':
    main()
