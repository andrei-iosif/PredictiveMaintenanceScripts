import configparser
import os

from forecasting.ml_forecasting import test_variable_window
from models.svr_forecaster import SVRForecaster
from preprocessing.preprocessing_func import read_time_series_data

RESULTS_PATH = r'results'
USE_SCALING = True
DEBUG = False

DATASET_LIST = ['OZONE', 'STAR', 'ATMOSPHERE_TEMPERATURE', 'PATIENT_DEMAND', 'POLLUTION_CO', 'LASER', 'CBE_ELECTRICITY',
                'FOURIER_A_CONSTANT', 'FOURIER_A_INCREASING', 'CHAOTIC_A']

if __name__ == "__main__":
    # forecaster = RandomForestForecaster()
    # forecaster = MLPForecaster()
    forecaster = SVRForecaster()

    config = configparser.ConfigParser()
    config.read(r'config/config.ini')

    for dataset in DATASET_LIST:
        dataset_path = config[dataset]['path']
        max_p = config.getint(dataset, 'max_p')
        add_date_index = config.getboolean(dataset, 'add_date_index')

        if add_date_index:
            start_date = config[dataset]['start_date']
            end_date = config[dataset]['end_date']
            frequency = config[dataset]['frequency']
        else:
            start_date = None
            end_date = None
            frequency = None

        series = read_time_series_data(dataset_path, start_date, end_date, frequency, add_date_index=add_date_index)

        synthetic_series = config.getboolean(dataset, 'synthetic')
        if not synthetic_series:
            h = config.getint(dataset, 'h')
        else:
            h = 0

        results_path = os.path.join(RESULTS_PATH, dataset, forecaster.name)
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        open(os.path.join(results_path, 'results.txt'), 'w').close()

        test_variable_window(series, forecaster, max_p, h=h, synthetic_series=synthetic_series, use_scaling=USE_SCALING,
                             results_path=results_path, debug=DEBUG)
