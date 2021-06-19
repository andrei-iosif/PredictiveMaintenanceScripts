from sklearn.neural_network import MLPRegressor

from time_series_forecasting.models.forecaster import Forecaster


class MLPForecaster(Forecaster):
    def __init__(self):
        self.name = 'mlp'

    def get_base_model(self):
        return MLPRegressor(
            shuffle=False,
            random_state=1,
            early_stopping=True,
            max_iter=500,
            momentum=0.2,
            learning_rate_init=0.03,
            learning_rate='adaptive')

    def fit_model(self, train_set, params):
        x_train, y_train = train_set[:, :-1], train_set[:, -1]
        model = MLPRegressor(hidden_layer_sizes=params["hidden_layer_sizes"],
                             activation='relu',
                             solver='lbfgs',
                             early_stopping=False,
                             shuffle=False,
                             max_iter=10000,
                             random_state=0)
        model.fit(x_train, y_train)
        return model

    def get_params_grid(self, max_p):
        solver = ['sgd']
        hidden_layer_sizes = [(i,) for i in range(3, max_p + 1, 2)]
        # hidden_layer_sizes = [(i, j, ) for i in range(3, max_p + 1, 2) for j in range(3, max_p + 1, 2)]
        activation = ["logistic", "tanh", "relu"]
        alpha = [0.0001]
        # alpha = 10. ** np.arange(-4, 1)
        return dict(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha)
