import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold, GridSearchCV, TimeSeriesSplit
from sklearn.utils._testing import ignore_warnings

from evaluation.metrics import tu_statistic
from plotting.plot_func import time_series_plot
from preprocessing.preprocessing_func import train_test_split, scale_time_series, difference_transform
from serialization.serialization import save_object, print_msg

USE_DIFFERENCING = False
USE_LOG_TRANSFORM = False
USE_EXPANDING_WINDOW = False


def grid_search(model, train_data, param_grid, compute_train_error, debug=False):
    X, y = train_data[:, :-1], train_data[:, -1]

    cv = TimeSeriesSplit(n_splits=10) if USE_EXPANDING_WINDOW else KFold(n_splits=10, shuffle=False)

    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    verbose = 2 if debug else 0

    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, verbose=verbose, scoring=scorer, n_jobs=3,
                        return_train_score=compute_train_error)
    grid.fit(X, y)
    return grid


def compute_evaluation_metrics(predictions, actual, series, forecast_horizon, eval_results_path="", debug=False):
    naive_predictions = series[-forecast_horizon - 1:-1]
    tu = tu_statistic(predictions, naive_predictions, actual)

    product = np.diff(np.concatenate([[series[-forecast_horizon - 1]], predictions])) * np.diff(
        series[-forecast_horizon - 1:])
    pocid = sum(product > 0) / forecast_horizon * 100

    mse = mean_squared_error(predictions, actual)
    mape = mean_absolute_percentage_error(predictions, actual)

    if debug:
        print("MSE: ", mse)
        print("TU: ", tu)
        print("POCID: ", pocid)
        print("MAPE:", mape)

        plt.plot(actual, label='Expected', marker='o')
        plt.plot(predictions, label='Predicted', marker='*')
        plt.legend()
        plt.show()
    else:
        print_msg("(MSE, TU, POCID, MAPE) = {}, {}, {}, {}".format(mse, tu, pocid, mape), eval_results_path)


def evaluate_baseline_model(series, h, synthetic_series=False):
    series = series.values
    N = len(series)
    horizon = np.rint(0.05 * N).astype(np.int32) if synthetic_series else h

    test_set = series[-horizon:]
    predictions = series[-horizon - 1:-1]

    compute_evaluation_metrics(predictions, test_set, series, horizon)


def test_fixed_window(series, forecast_model, max_p, h, synthetic_series=False, use_scaling=True, window=11,
                      use_grid_search=False, model_params=None):
    series_values = series.values
    N = len(series_values)
    horizon = int(0.05 * N) if synthetic_series else h

    if use_scaling:
        scaler, series_values = scale_time_series(series_values, horizon + max_p)
    else:
        scaler = None

    train_dataset, test_dataset = train_test_split(series_values, window, horizon)
    x_test, y_test = test_dataset[:, :-1], test_dataset[:, -1]

    if use_grid_search:
        grid_cv = grid_search(forecast_model.get_base_model(), train_dataset, forecast_model.get_params_grid(max_p),
                              compute_train_error=False)
        print('Best parameters found:\n', grid_cv.best_params_)
        predictions = grid_cv.predict(x_test)
    else:
        model = forecast_model.fit_model(train_dataset, model_params)
        predictions = model.predict(x_test)

    if use_scaling:
        predictions = scaler.inverse_transform(predictions.reshape(len(predictions), 1)).flatten()
        y_test = scaler.inverse_transform(y_test.reshape(len(y_test), 1)).flatten()
        series_values = scaler.inverse_transform(series_values.reshape(len(series_values), 1)).flatten()

    compute_evaluation_metrics(predictions, y_test, series_values, horizon)


@ignore_warnings(category=ConvergenceWarning)
def test_variable_window(series, forecast_model, max_p, h=0, synthetic_series=False, use_scaling=True, results_path="", debug=False):
    series_values = series.values
    N = len(series_values)

    horizon = int(0.05 * N) if synthetic_series else h

    if use_scaling:
        scaler, series_values = scale_time_series(series_values, horizon)
    else:
        scaler = None

    min_error = np.inf
    best_l = max_p
    best_grid = None
    for l in range(3, max_p + 1, 2):
        print("\nEstimating models for window size = {}".format(l))
        train_dataset, _ = train_test_split(series_values, l, horizon)

        grid_cv = grid_search(forecast_model.get_base_model(), train_dataset, forecast_model.get_params_grid(max_p),
                              compute_train_error=False, debug=debug)

        error = -grid_cv.best_score_
        if error < min_error:
            min_error = error
            best_grid = grid_cv
            best_l = l

    eval_results_path = os.path.join(results_path, "results.txt")
    print_msg("\nBest window size: {}".format(best_l), eval_results_path)
    print_msg("\nBest parameters found: {}\n".format(best_grid.best_params_), eval_results_path)

    params = best_grid.best_params_
    params["window"] = best_l

    save_object(params, os.path.join(results_path, f"best_params_{forecast_model.name}.pkl"))
    save_object(best_grid, os.path.join(results_path, f"grid_search_{forecast_model.name}.pkl"))
    save_object(best_grid.best_estimator_, os.path.join(results_path, f"{forecast_model.name}_model.pkl"))

    _, test_dataset = train_test_split(series_values, best_l, horizon)
    x_test, y_test = test_dataset[:, :-1], test_dataset[:, -1]

    predictions = best_grid.predict(x_test)
    if use_scaling:
        predictions = scaler.inverse_transform(predictions.reshape(len(predictions), 1)).flatten()
        y_test = scaler.inverse_transform(y_test.reshape(len(y_test), 1)).flatten()
        series_values = scaler.inverse_transform(series_values.reshape(len(series_values), 1)).flatten()

    compute_evaluation_metrics(predictions, y_test, series_values, horizon, eval_results_path, debug=debug)


@ignore_warnings(category=ConvergenceWarning)
def test_variable_window_with_preprocessing(series, forecast_model, max_p, h, synthetic_series=False, use_scaling=True,
                                            use_log_transform=False, use_differencing=False, results_path="", debug=False):
    series_values = series.values
    N = len(series_values)
    horizon = int(0.05 * N) if synthetic_series else h

    series_log = np.log(series) if use_log_transform else series_values
    if debug:
        time_series_plot(series_log)

    series_diff = difference_transform(series_log, interval=1) if use_differencing else series_log
    if debug:
        # stationarity_test(series_diff)
        plt.plot(range(len(series_diff)), series_diff)
        plt.show()

    if use_scaling:
        scaler, series_diff = scale_time_series(series_diff, horizon + max_p)
    else:
        scaler = None

    min_error = np.inf
    best_l = max_p
    best_grid = None
    for l in range(3, max_p + 1, 2):
        print("\nEstimating models for window size = ", l)
        train_dataset, _ = train_test_split(series_diff, l, horizon)

        grid_cv = grid_search(forecast_model.get_base_model(), train_dataset, forecast_model.get_params_grid(max_p),
                              compute_train_error=False, debug=debug)

        error = -grid_cv.best_score_
        if error < min_error:
            min_error = error
            best_grid = grid_cv
            best_l = l

    eval_results_path = os.path.join(results_path, "results.txt")
    print_msg("\nBest window size: {}".format(best_l), eval_results_path)
    print_msg("\nBest parameters found: {}\n".format(best_grid.best_params_), eval_results_path)

    save_object(best_grid, os.path.join(results_path, f"grid_search_{forecast_model.name}.pkl"))
    save_object(best_grid.best_estimator_, os.path.join(results_path, f"{forecast_model.name}_model.pkl"))

    _, test_dataset = train_test_split(series_diff, best_l, horizon)
    x_test, y_test = test_dataset[:, :-1], test_dataset[:, -1]

    predictions = best_grid.predict(x_test)
    if use_scaling:
        predictions = scaler.inverse_transform(predictions.reshape(len(predictions), 1)).flatten()
        y_test = scaler.inverse_transform(y_test.reshape(len(y_test), 1)).flatten()

    def inv_diff(x_diff, x_0):
        return np.r_[x_0, x_diff].cumsum()[1:]

    # Invert difference transform
    if use_differencing:
        first_el = series_log[N - horizon + 1]
        y_test = inv_diff(y_test, first_el)
        predictions = inv_diff(predictions, first_el)

    # Invert power transform
    if use_log_transform:
        predictions = np.exp(predictions)
        y_test = np.exp(y_test)

    compute_evaluation_metrics(predictions, y_test, series_values, horizon)

