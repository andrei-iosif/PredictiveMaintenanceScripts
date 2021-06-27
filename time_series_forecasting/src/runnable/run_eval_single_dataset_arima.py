import configparser
import os
import sys

import numpy as np

from time_series_forecasting.src.forecasting.arima_forecasting import forecast_updated_iteration, compute_evaluation_metrics_arima
from time_series_forecasting.src.plotting.plot_func import plot_forecasting
from time_series_forecasting.src.preprocessing.preprocessing_func import read_time_series_data
from time_series_forecasting.src.serialization.serialization import load_object

DEBUG = True
CONFIG_PATH = r'../../config/config.ini'
RESULTS_PATH = r'../../results'


def evaluate_arima(series, model, h, synthetic_series, apply_log_transform=False):
    N = len(series)
    horizon = int(0.05 * N) if synthetic_series else h

    _, test_set = series[:-horizon], series[-horizon:]
    predictions = forecast_updated_iteration(model, test_set, apply_log_transform=apply_log_transform)
    if apply_log_transform:
        predictions = np.exp(predictions)
    naive_predictions = series[-horizon - 1:-1]

    product = np.diff(np.concatenate([[series[-horizon - 1]], predictions])) * np.diff(series[-horizon - 1:])
    pocid = sum(product > 0) / horizon * 100

    tu, mse, mape = compute_evaluation_metrics_arima(predictions, naive_predictions, test_set, debug=False)
    print(f"Evaluation metrics for ARIMA model")
    print("(MSE, TU, POCID, MAPE) = {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(mse, tu, pocid, mape))

    plot_forecasting(test_set, [predictions], ["ARIMA predictions"],
                     title=f"Forecasting results for {DATASET} dataset",
                     linestyles=['--', '-.', ':'],
                     output_path=None)


if __name__ == "__main__":
    DATASET = sys.argv[1]

    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)

    dataset_path = config[DATASET]['path']
    max_p = config.getint(DATASET, 'max_p')
    add_date_index = config.getboolean(DATASET, 'add_date_index')

    start_date = config[DATASET]['start_date'] if add_date_index else None
    end_date = config[DATASET]['end_date'] if add_date_index else None
    frequency = config[DATASET]['frequency'] if add_date_index else None

    series = read_time_series_data(dataset_path, start_date, end_date, frequency, add_date_index=add_date_index)
    synthetic_series = config.getboolean(DATASET, 'synthetic')
    h = config.getint(DATASET, 'h') if not synthetic_series else 0

    results_path = os.path.join(RESULTS_PATH, DATASET)
    arima_model = load_object(os.path.join(results_path, "arima", "arima_model.pkl"))

    evaluate_arima(series.values, arima_model, h, synthetic_series)
