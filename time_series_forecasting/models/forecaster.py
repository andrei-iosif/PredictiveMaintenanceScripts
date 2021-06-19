from abc import ABC, abstractmethod


class Forecaster(ABC):
    @abstractmethod
    def get_base_model(self):
        pass

    @abstractmethod
    def fit_model(self, train_set, params):
        pass

    @abstractmethod
    def get_params_grid(self, max_p):
        pass
