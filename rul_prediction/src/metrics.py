import numpy as np

from sklearn.metrics import mean_squared_error


def cmapss_score_function(actual, predictions, normalize=True):
    # diff < 0 -> over-estimation
    # diff > 0 -> under-estimation
    diff = actual - predictions
    alpha = np.full_like(diff, 1 / 13)
    negative_diff_mask = diff < 0
    alpha[negative_diff_mask] = 1 / 10
    score = np.sum(np.exp(alpha * np.abs(diff)))

    if normalize:
        N = len(predictions)
        score /= N
    return score


def compute_evaluation_metrics(actual, predictions, label='Test'):
    mse = mean_squared_error(actual, predictions)
    rmse = np.sqrt(mse)
    cmapss_score = cmapss_score_function(actual, predictions)
    print('{} set:\nMSE: {:.2f}\nRMSE: {:.2f}\nCMAPSS score: {:.2f}\n'.format(label, mse, rmse,
                                                                              cmapss_score))
    return mse, rmse, cmapss_score
