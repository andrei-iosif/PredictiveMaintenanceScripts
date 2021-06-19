import numpy as np

from sklearn.ensemble import RandomForestRegressor

from models.forecaster import Forecaster


class RandomForestForecaster(Forecaster):
    def __init__(self):
        self.name = 'rf'

    def get_base_model(self):
        return RandomForestRegressor(random_state=42, bootstrap=True)

    def fit_model(self, train_set, params):
        x_train, y_train = train_set[:, :-1], train_set[:, -1]
        model = RandomForestRegressor(n_estimators=params['n_estimators'], n_jobs=2, random_state=42)
        model.fit(x_train, y_train)
        return model

    def get_params_grid(self, max_p):
        return {'n_estimators': 100 * np.arange(1, 4),
                'max_features': ['auto', 'sqrt', 'log2'],
                # 'max_depth' : [4,5,6,7,8],
                }
