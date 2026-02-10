import numpy as np


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def directional_accuracy(y_true, y_pred):
    return np.mean(
        np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))
    )
