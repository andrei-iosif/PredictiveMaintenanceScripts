def exponential_smoothing(df, columns, alpha):
    df = df.copy()
    df[columns] = df.groupby('unit')[columns].apply(
        lambda x: x.ewm(alpha=alpha, adjust=False).mean())
    return df


def moving_average(df, columns, window):
    df = df.copy()
    df[columns] = df.groupby('unit')[columns].apply(
        lambda x: x.rolling(window, min_periods=1).mean())
    return df


def signal_smoothing(df, columns, filter_type, filter_param):
    if filter_type == 'es':
        return exponential_smoothing(df, columns, alpha=filter_param)
    elif filter_type == 'ma':
        return moving_average(df, columns, window=filter_param)
    else:
        raise RuntimeError("Invalid filter type")
