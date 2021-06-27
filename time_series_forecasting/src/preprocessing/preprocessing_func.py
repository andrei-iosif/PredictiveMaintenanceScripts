import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def read_time_series_data(file_path, start_date, end_date, freq, add_date_index=True):
    """
    Read time series from file, given that values are separated by newline. Optionally, add timestamp index.
    :param file_path: path to the data file
    :param start_date: timestamp of the first observation
    :param end_date: timestamp of the last observation
    :param freq: sampling frequency; can have values D (daily) or M (monthly)
    :param add_date_index: add timestamps to observations
    :return: pd.Dataframe
    """

    series = pd.read_csv(file_path, delimiter="\n", squeeze=True, header=None)

    if add_date_index:
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        series.index = date_range
    return series


def series_to_supervised(data, n_in=1, n_out=1, drop_nan=True):
    """
    Transform a time series into a supervised learning dataset.
    :param data: Sequence of observations as a numpy array.
    :param n_in: Number of lag observations as input (X).
    :param n_out: Number of observations as output (y).
    :param drop_nan: Boolean whether or not to drop rows with NaN values.
    :return: pd.DataFrame of series framed for supervised learning.
    """

    df = pd.DataFrame(data)
    n_vars = len(df.columns)
    cols, names = [], []

    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]

    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]

    dataset = pd.concat(cols, axis=1)
    dataset.columns = names

    if drop_nan:
        dataset.dropna(inplace=True)
    return dataset


def train_test_split(series, l, h):
    """
    Train test split for time series, converted to tabular format.
    :param series: Sequence of observations as a numpy array.
    :param l: sliding window length
    :param h: prediction horizon
    :return: two pd.Dataframe, corresponding to the supervised train set and test set
    """
    dataset = series_to_supervised(series, n_in=l, n_out=1, drop_nan=True).values
    return dataset[:-h, :], dataset[-h:, :]


def scale_time_series(series, test_size):
    """
    Apply normalization to zero mean and unit variance to a series. Fit the scaler on the training set only.
    :param series: Sequence of observations as a numpy array.
    :param test_size: length of the series that would become the supervised test set
    :return: the scaler object and the transformed series
    """
    scaler = StandardScaler()
    train_series = series[:-test_size]
    scaler = scaler.fit(train_series.reshape(len(train_series), 1))
    return scaler, scaler.transform(series.reshape(len(series), 1)).flatten()


def difference_transform(series, interval=1):
    """
    Apply differencing to a time series.
    :param series: sequence of observations as a numpy array
    :param interval: order of differencing
    :return: the differenced time series
    """
    return np.array([series[i] - series[i - interval] for i in range(interval, len(series))])


def inv_diff(series_diff, x_0):
    """
    Revert the differencing operation applied to a time series.
    :param series_diff: the differenced time series
    :param x_0: first element of original, non-differenced time series
    """
    return np.r_[x_0, series_diff].cumsum()[1:]
