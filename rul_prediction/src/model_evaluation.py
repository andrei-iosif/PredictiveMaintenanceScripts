import os

import numpy as np

from keras.models import load_model

from src.metrics import compute_evaluation_metrics
from src.plotting import plot_loss_curves
from src.save_object import load_object


class EvaluationResults:
    def __init__(self, mse_vals, rmse_vals, cmapss_vals, prediction_vals):
        self.mse_vals = mse_vals
        self.rmse_vals = rmse_vals
        self.cmapss_vals = cmapss_vals
        self.prediction_vals = prediction_vals


def evaluate_mlp(x_test, y_test, train_results_path, plot_loss=False):
    scaler_path = os.path.join(train_results_path, "scaler.pkl")
    model_path = os.path.join(train_results_path, "mlp_model_trained.h5")
    history_path = os.path.join(train_results_path, f"history.pkl")

    # Standardization
    scaler = load_object(scaler_path)
    x_test_scaled = scaler.transform(x_test)

    # Load model and history
    loaded_model = load_model(model_path)
    if plot_loss:
        history = load_object(history_path)
        plot_loss_curves(history)

    # Performance evaluation
    predictions_test = loaded_model.predict(x_test_scaled).flatten()
    mse, rmse, cmapss_score = compute_evaluation_metrics(predictions_test, y_test)

    return mse, rmse, cmapss_score, predictions_test


def evaluate_mlp_multiple_splits(x_test, y_test, num_trials, train_results_path, eval_results_path=None, plot_loss=False):
    mse_vals = []
    rmse_vals = []
    cmapss_vals = []
    prediction_vals = []

    for trial_num in range(num_trials):
        results_path_crr_split = os.path.join(train_results_path, f"split_{trial_num}")

        scaler_path = os.path.join(results_path_crr_split, "scaler.pkl")
        model_path = os.path.join(results_path_crr_split, "mlp_model_trained.h5")
        history_path = os.path.join(results_path_crr_split, f"history_{trial_num}.pkl")

        # Standardization
        scaler = load_object(scaler_path)
        x_test_scaled = scaler.transform(x_test)

        # Load model and history
        loaded_model = load_model(model_path)
        if plot_loss:
            history = load_object(history_path)
            plot_loss_curves(history)

        # Performance evaluation
        predictions_test = loaded_model.predict(x_test_scaled).flatten()
        mse, rmse, cmapss_score = compute_evaluation_metrics(predictions_test, y_test)

        mse_vals.append(mse)
        rmse_vals.append(rmse)
        cmapss_vals.append(cmapss_score)
        prediction_vals.append(predictions_test)

    mse_mean = np.mean(mse_vals)
    mse_std = np.std(mse_vals)
    rmse_mean = np.mean(rmse_vals)
    rmse_std = np.std(rmse_vals)
    cmapss_mean = np.mean(cmapss_vals)
    cmapss_std = np.std(cmapss_vals)

    print("MSE: mean = {:.2f}   stddev = {:.2f}".format(mse_mean, mse_std))
    print("RMSE: mean = {:.2f}   stddev = {:.2f}".format(rmse_mean, rmse_std))
    print("CMAPSS: mean = {:.2f}   stddev = {:.2f}".format(cmapss_mean, cmapss_std))

    return EvaluationResults(mse_vals, rmse_vals, cmapss_vals, prediction_vals)
