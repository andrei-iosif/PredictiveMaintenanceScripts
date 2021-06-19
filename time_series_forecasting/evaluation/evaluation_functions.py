import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

from evaluation.metrics import tu_statistic
from preprocessing.preprocessing_func import scale_time_series, train_test_split


def get_predictions(model, series, h, window_size, use_scaling, synthetic_series):
    series_values = series.values
    horizon = int(0.05 * len(series_values)) if synthetic_series else h

    if use_scaling:
        scaler, series_values = scale_time_series(series_values, horizon)
    else:
        scaler = None

    _, test_dataset = train_test_split(series_values, window_size, horizon)
    x_test, y_test = test_dataset[:, :-1], test_dataset[:, -1]
    predictions = model.predict(x_test)

    if use_scaling:
        predictions = scaler.inverse_transform(predictions.reshape(len(predictions), 1)).flatten()
        y_test = scaler.inverse_transform(y_test.reshape(len(y_test), 1)).flatten()
        series_values = scaler.inverse_transform(series_values.reshape(len(series_values), 1)).flatten()

    print(f"Evaluation metrics for model: {model.__class__.__name__}")
    compute_metrics(predictions, y_test, series_values, h)
    return predictions, y_test


def compute_metrics(predictions, actual, series, forecast_horizon):
    naive_predictions = series[-forecast_horizon - 1:-1]
    tu = tu_statistic(predictions, naive_predictions, actual)

    product = np.diff(np.concatenate([[series[-forecast_horizon - 1]], predictions])) * np.diff(
        series[-forecast_horizon - 1:])
    pocid = sum(product > 0) / forecast_horizon * 100

    mse = mean_squared_error(predictions, actual)
    mape = mean_absolute_percentage_error(predictions, actual)
    print("(MSE, TU, POCID, MAPE) = {:.3f}, {:.3f}, {:.3f}, {:.3f}\n".format(mse, tu, pocid, mape))