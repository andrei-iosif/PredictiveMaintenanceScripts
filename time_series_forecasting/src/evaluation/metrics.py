from numpy import sum, diff, square


def tu_statistic(predictions, naive_predictions, actual):
    num = sum(square(actual - predictions))
    denom = sum(square(actual - naive_predictions))
    return num / denom


def pocid_statistic(predictions, actual):
    h = len(predictions)
    product = diff(predictions) * diff(actual)
    return sum(product > 0) / h * 100
