import numpy as np

from sklearn.svm import SVR

from time_series_forecasting.models.forecaster import Forecaster


class SVRForecaster(Forecaster):
    def __init__(self):
        self.name = 'svr'

    def get_base_model(self):
        return SVR(kernel='rbf')

    def fit_model(self, train_set, params):
        x_train, y_train = train_set[:, :-1], train_set[:, -1]
        model = SVR(kernel='rbf', gamma=params["gamma"], C=params["C"], epsilon=params["epsilon"])
        model.fit(x_train, y_train)
        return model

    def get_params_grid(self, max_p):
        C_range = 10. ** np.arange(-3, 3)
        gamma_range = 10. ** np.arange(-5, 2)
        epsilon_range = np.array([0.01, 0.05, 0.1, 0.5, 1])
        return dict(gamma=gamma_range, C=C_range, epsilon=epsilon_range)
