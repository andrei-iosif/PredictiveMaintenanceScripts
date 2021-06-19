import configparser
import os
import sys

from evaluation.evaluation_functions import get_predictions
from plotting.plot_func import plot_forecasting
from preprocessing.preprocessing_func import read_time_series_data
from serialization.serialization import load_object

DEBUG = True
USE_SCALING = True
RESULTS_PATH = r'results'


def load_model(model_str, results_path):
    path = os.path.join(results_path, model_str)
    params = load_object(os.path.join(path, f"best_params_{model_str}.pkl"))
    model = load_object(os.path.join(path, f"{model_str}_model.pkl"))
    return model, params["window"]


if __name__ == "__main__":
    DATASET = sys.argv[1]

    config = configparser.ConfigParser()
    config.read(r'config/config.ini')

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
    svr_model, svr_window = load_model('svr', results_path)
    mlp_model, mlp_window = load_model('mlp', results_path)
    rf_model, rf_window = load_model('rf', results_path)

    predictions_svr, y_test = get_predictions(svr_model, series, h, svr_window, USE_SCALING, synthetic_series)
    predictions_mlp, _ = get_predictions(mlp_model, series, h, mlp_window, USE_SCALING, synthetic_series)
    predictions_rf, _ = get_predictions(rf_model, series, h, rf_window, USE_SCALING, synthetic_series)

    plot_forecasting(y_test, [predictions_svr, predictions_mlp, predictions_rf],
                     ["SVR predictions", "MLP predictions", "RF predictions"],
                     title=f"Forecasting results for {DATASET} dataset",
                     linestyles=['--', '-.', ':'],
                     output_path=os.path.join(results_path, f"forecasting_{DATASET}.png"))
