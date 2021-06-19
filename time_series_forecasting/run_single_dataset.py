import configparser
import os
import sys

from forecasting.ml_forecasting import time_series_plot, test_variable_window, evaluate_baseline_model
from models.svr_forecaster import SVRForecaster
from preprocessing.preprocessing_func import read_time_series_data
from time_series_forecasting.models.random_forest_forecaster import RandomForestForecaster

DEBUG = True
USE_SCALING = True
RESULTS_PATH = r'results'


if __name__ == "__main__":
    DATASET = sys.argv[1]

    forecaster = RandomForestForecaster()
    # forecaster = MLPForecaster()
    # forecaster = SVRForecaster()

    config = configparser.ConfigParser()
    config.read(r'config/config.ini')

    dataset_path = config[DATASET]['path']
    max_p = config.getint(DATASET, 'max_p')
    add_date_index = config.getboolean(DATASET, 'add_date_index')

    if add_date_index:
        start_date = config[DATASET]['start_date']
        end_date = config[DATASET]['end_date']
        frequency = config[DATASET]['frequency']
    else:
        start_date = None
        end_date = None
        frequency = None

    series = read_time_series_data(dataset_path, start_date, end_date, frequency, add_date_index=add_date_index)
    if DEBUG:
        time_series_plot(series.values)

    synthetic_series = config.getboolean(DATASET, 'synthetic')
    if not synthetic_series:
        h = config.getint(DATASET, 'h')
    else:
        h = 0

    results_path = os.path.join(RESULTS_PATH, DATASET, forecaster.name)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    open(os.path.join(results_path, 'results.txt'), 'w').close()

    evaluate_baseline_model(series, h, synthetic_series=synthetic_series)

    # test_variable_window(series, forecaster, max_p, h=h, synthetic_series=synthetic_series, use_scaling=USE_SCALING,
    #                      results_path=results_path, debug=True, dataset_name=DATASET)
