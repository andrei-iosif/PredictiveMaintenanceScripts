def add_time_lags(df_input, lags_list, columns):
    df = df_input.copy()
    for i in lags_list:
        lagged_cols = [col + f'_lag_{i}' for col in columns]
        df[lagged_cols] = df.groupby('unit')[columns].shift(i)

    df.dropna(inplace=True)
    df.drop(['unit'], axis=1, inplace=True)
    return df

