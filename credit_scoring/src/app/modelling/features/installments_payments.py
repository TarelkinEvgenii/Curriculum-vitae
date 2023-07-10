import pandas as pd


def main() -> None:
    """Функция генерации фич из файла installments_payments."""
    df_install = pd.read_csv(
        r'C:\Users\user\Desktop\SHIFT\credit_scoring\data\installments_payments.csv',
    )
    # Percentage and difference paid in each installment (amount paid and installment value)
    df_install['PAYMENT_PERC'] = df_install['AMT_PAYMENT'] / df_install['AMT_INSTALMENT']
    df_install['PAYMENT_DIFF'] = df_install['AMT_INSTALMENT'] - df_install['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    df_install['DPD'] = df_install['DAYS_ENTRY_PAYMENT'] - df_install['DAYS_INSTALMENT']
    df_install['DBD'] = df_install['DAYS_INSTALMENT'] - df_install['DAYS_ENTRY_PAYMENT']
    df_install['DPD'] = df_install['DPD'].apply(lambda xl: xl if xl > 0 else 0)
    df_install['DBD'] = df_install['DBD'].apply(lambda xl: xl if xl > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum', 'min', 'std'],
        'DBD': ['max', 'mean', 'sum', 'min', 'std'],
        'PAYMENT_PERC': ['max', 'mean', 'var', 'min', 'std'],
        'PAYMENT_DIFF': ['max', 'mean', 'var', 'min', 'std'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum', 'min', 'std'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum', 'std'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum', 'std'],
    }

    install_agg = df_install.groupby('SK_ID_CURR').agg(aggregations)
    install_agg.columns = pd.Index(
        ['INSTALL_' + ec[0] + '_' + ec[1].upper() for ec in install_agg.columns.tolist()],
    )
    # Count installments accounts
    install_agg['INSTALL_COUNT'] = df_install.groupby('SK_ID_CURR').size()
    install_agg.reset_index(inplace=True)
    install_agg.to_csv(
        r'C:\Users\user\Desktop\SHIFT\credit_scoring\data\features_installments_payments.csv',
        index=False,
    )


if __name__ == '__main__':
    main()
