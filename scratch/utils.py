import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    return np.exp(-x) / ((np.exp(-x) + 1) ** 2)


def softmax(x):
    exps = np.exp(x - x.max())
    return exps / np.sum(exps, axis=0)


def softmax_deriv(x):
    exps = np.exp(x - x.max())
    return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_deriv(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

