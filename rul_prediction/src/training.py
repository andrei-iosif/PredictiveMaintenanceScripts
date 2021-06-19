import os

import numpy as np

from keras.models import load_model

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.metrics import compute_evaluation_metrics
from src.model import create_mlp_model, get_callbacks
from src.plotting import plot_loss_curves
from src.save_object import save_object
from src.utils import numbers_list_to_string


class MLPConfigParams:
    def __init__(self, layer_sizes=(), activation='tanh', dropout=0.0):
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.dropout = dropout


def train_mlp(model, x_train, y_train, x_val, y_val, weights_file=None, epochs=200, batch_size=512, callbacks=()):
    """ Compile and train model. Optionally, load initial model weights from file. """
    model.compile(loss='mean_squared_error', optimizer='adam')
    if weights_file is not None:
        model.load_weights(weights_file)
    return model.fit(x_train, y_train,
                     validation_data=(x_val, y_val),
                     epochs=epochs,
                     batch_size=batch_size,
                     verbose=1,
                     callbacks=callbacks)


def train_evaluate_mlp(x_train, y_train, x_test, y_test, num_trials,
                       mlp_config_params, results_path, epochs, batch_size, results_file=None):
    """
    Train and evaluate model. Repeat for a number of trials. Used in hyperparameter search.
    :param x_train: training set features
    :param y_train: training set target
    :param x_test: validation set features
    :param y_test: validation set target
    :param num_trials: number of repeats for each configuration
    :param mlp_config_params: model hyperparameters
    :param results_path: path to save model weights and training history
    :param epochs: maximum number of epochs for training
    :param batch_size: batch size for mini-batch SGD
    :param results_file: file to save evaluation metrics
    :return: void
    """

    mse_vals = []
    rmse_vals = []
    cmapss_vals = []

    input_dim = x_train.shape[1]

    for trial_num in range(num_trials):
        # Create results path for current split
        results_path_crr_split = os.path.join(results_path, "split_{}".format(trial_num))
        if not os.path.exists(results_path_crr_split):
            os.makedirs(results_path_crr_split)

        # Train-validation split for early stopping
        x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x_train,
                                                                                  y_train,
                                                                                  test_size=0.1,
                                                                                  random_state=trial_num)

        # Standardization
        scaler_file = os.path.join(results_path_crr_split, 'scaler.pkl')
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train_split)
        x_val_scaled = scaler.transform(x_val_split)
        save_object(scaler, scaler_file)

        weights_file = os.path.join(results_path, 'mlp_initial_weights.h5')
        model_path = os.path.join(results_path_crr_split, 'mlp_model_trained.h5')

        # Initialize weights only in first split
        if trial_num == 0:
            model = create_mlp_model(input_dim,
                                     hidden_layer_sizes=mlp_config_params.layer_sizes,
                                     activation=mlp_config_params.activation,
                                     dropout=mlp_config_params.dropout,
                                     output_weights_file=weights_file)
        else:
            model = create_mlp_model(input_dim,
                                     hidden_layer_sizes=mlp_config_params.layer_sizes,
                                     activation=mlp_config_params.activation,
                                     dropout=mlp_config_params.dropout)
        model.summary()

        # Train model
        history = train_mlp(model,
                            x_train_scaled, y_train_split,
                            x_val_scaled, y_val_split,
                            weights_file=weights_file,
                            batch_size=batch_size,
                            epochs=epochs,
                            callbacks=get_callbacks(model_path))

        history_file = os.path.join(results_path_crr_split, f"history_{trial_num}.pkl")
        plot_loss_curves(history.history)
        save_object(history.history, history_file)

        # Performance evaluation
        x_test_scaled = scaler.transform(x_test)
        loaded_model = load_model(model_path)  # load model saved by checkpoint
        predictions_test = loaded_model.predict(x_test_scaled).flatten()
        mse, rmse, cmapss_score = compute_evaluation_metrics(predictions_test, y_test)

        mse_vals.append(mse)
        rmse_vals.append(rmse)
        cmapss_vals.append(cmapss_score)

    mse_mean = np.mean(mse_vals)
    mse_std = np.std(mse_vals)
    rmse_mean = np.mean(rmse_vals)
    rmse_std = np.std(rmse_vals)
    cmapss_mean = np.mean(cmapss_vals)
    cmapss_std = np.std(cmapss_vals)

    if results_file is not None:
        with open(results_file, "a") as file:
            line_to_write = f"{numbers_list_to_string(mse_vals)}, {numbers_list_to_string(rmse_vals)},"
            line_to_write += f"{numbers_list_to_string(cmapss_vals)}, {mse_mean}, {mse_std}, {rmse_mean},"
            line_to_write += f"{rmse_std}, {cmapss_mean}, {cmapss_std}\n"
            file.write(line_to_write)

    print("MSE: mean = {:.2f}   stddev = {:.2f}".format(mse_mean, mse_std))
    print("RMSE: mean = {:.2f}   stddev = {:.2f}".format(rmse_mean, rmse_std))
    print("CMAPSS: mean = {:.2f}   stddev = {:.2f}".format(cmapss_mean, cmapss_std))

    return mse_vals, rmse_vals, cmapss_vals


