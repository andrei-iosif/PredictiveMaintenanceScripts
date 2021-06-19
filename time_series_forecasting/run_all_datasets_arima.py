import configparser
import os

from forecasting.arima_forecasting import run
from preprocessing.preprocessing_func import read_time_series_data

RESULTS_PATH = r'results'
USE_SCALING = True
DEBUG = False

DATASET_LIST = ['OZONE', 'STAR', 'ATMOSPHERE_TEMPERATURE', 'PATIENT_DEMAND', 'POLLUTION_CO', 'LASER', 'CBE_ELECTRICITY',
                'FOURIER_A_CONSTANT', 'FOURIER_A_INCREASING', 'CHAOTIC_A']


if __name__ == "__main__":
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

        results_path = os.path.join(RESULTS_PATH, dataset, "arima")
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        open(os.path.join(results_path, 'results.txt'), 'w').close()

        run(series.values, h, synthetic_series, debug=DEBUG, results_path=results_path)

