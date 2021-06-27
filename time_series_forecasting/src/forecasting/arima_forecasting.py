import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from time_series_forecasting.src.evaluation.metrics import tu_statistic
from time_series_forecasting.src.serialization.serialization import save_object, print_msg


def fit_model(series, order):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    return model_fit


def model_selection_arima(series, params):
    p_vals = params["p"]
    d_vals = params["d"]
    q_vals = params["q"]

    best_aic = np.inf
    best_fit = None
    for p in p_vals:
        for d in d_vals:
            for q in q_vals:
                order = (p, d, q)
                print("Fitting ARIMA model with order: ", order)
                fit = fit_model(series, order)

                # aic = - 2 * fit.llf + (np.log(len(series)) + 1) * (p + q + 1)
                aic = fit.aic

                if aic < best_aic:
                    best_aic = aic
                    best_fit = fit

    return best_fit


def forecast_updated_iteration(fitted_model, test_set, apply_log_transform=False):
    predictions = np.array([])
    for val in test_set:
        predictions = np.concatenate([predictions, fitted_model.forecast(1)])
        new_val = [np.log(val)] if apply_log_transform else [val]
        fitted_model = fitted_model.append(new_val)

    return predictions


def evaluate_baseline_model(series, h, synthetic_series=False):
    N = len(series)
    horizon = np.rint(0.05 * N).astype(np.int32) if synthetic_series else h

    test_set = series[-horizon:]
    predictions = series[-horizon - 1:-1]

    product = np.diff(np.concatenate([[series[-horizon - 1]], predictions])) * np.diff(series[-horizon - 1:])
    pocid = sum(product > 0) / horizon * 100
    print("POCID: ", pocid)

    compute_evaluation_metrics_arima(predictions, predictions, test_set)


def get_params_grid(N, h):
    p_range = np.arange(0, np.sqrt(np.log(N - h)))
    d_range = np.array([0, 1, 2])
    q_range = np.arange(0, np.sqrt(np.log(N - h)))
    return {"p": p_range, "d": d_range, "q": q_range}


def compute_evaluation_metrics_arima(predictions, naive_predictions, actual, debug=False):
    tu = tu_statistic(predictions, naive_predictions, actual)
    mse = mean_squared_error(predictions, actual)
    mape = mean_absolute_percentage_error(predictions, actual)

    if debug:
        plt.plot(actual, label='Expected', marker='o')
        plt.plot(predictions, label='Predicted', marker='*')
        plt.legend()
        plt.show()

    return tu, mse, mape


def run(series, h, synthetic_series, debug=False, results_path="", apply_log_transform=False):
    N = len(series)

    horizon = np.rint(0.05 * N).astype(np.int32) if synthetic_series else h

    train_set, test_set = series[:-horizon], series[-horizon:]

    if apply_log_transform:
        train_set = np.log(train_set)

    params = get_params_grid(N, horizon)
    best_model = model_selection_arima(train_set, params)

    save_object(best_model, os.path.join(results_path, "arima_model.pkl"))

    if debug:
        print_msg(best_model.summary(), os.path.join(results_path, "results.txt"))
        best_model.plot_diagnostics(figsize=(15, 12))
        plt.show()

    predictions = forecast_updated_iteration(best_model, test_set, apply_log_transform)

    if apply_log_transform:
        predictions = np.exp(predictions)

    naive_predictions = series[-horizon-1:-1]

    product = np.diff(np.concatenate([[series[-horizon-1]], predictions])) * np.diff(series[-horizon-1:])
    pocid = sum(product > 0) / horizon * 100

    tu, mse, mape = compute_evaluation_metrics_arima(predictions, naive_predictions, test_set)
    print_msg("(MSE, TU, POCID, MAPE) = {}, {}, {}, {}".format(mse, tu, pocid, mape), os.path.join(results_path, "results.txt"))


def test_model(series, order, h, synthetic_series, apply_log_transform=False):
    N = len(series)
    horizon = int(0.05 * N) if synthetic_series else h

    train_set, test_set = series[:-horizon], series[-horizon:]
    if apply_log_transform:
        train_set = np.log(train_set)

    model = fit_model(train_set, order)
    print(model.summary())

    predictions = forecast_updated_iteration(model, test_set, apply_log_transform=apply_log_transform)
    if apply_log_transform:
        predictions = np.exp(predictions)
    naive_predictions = series[-horizon - 1:-1]

    product = np.diff(np.concatenate([[series[-horizon - 1]], predictions])) * np.diff(series[-horizon - 1:])
    pocid = sum(product > 0) / horizon * 100
    print("POCID: ", pocid)
    compute_evaluation_metrics_arima(predictions, naive_predictions, test_set)


def stationarity_test(series):
    print(series.describe())
    result = adfuller(series)
    print('ADF Statistic: ', result[0])
    print('p-value: ', result[1])
    print('Critical Values:' )
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))


def seasonal_decomposition(series):
    from statsmodels.tsa.seasonal import seasonal_decompose

    decomposition = seasonal_decompose(series)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    plt.subplot(411)
    plt.plot(series, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def exploration(series):
    stationarity_test(series)

    plot_acf(series)
    plot_pacf(series)

    plt.figure()
    series.plot()
    plt.show()
