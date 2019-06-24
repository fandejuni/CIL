# -*- coding: utf-8 -

import numpy as np

def getRCCBCC(Y_pred, Y_truth):
    road_indices = np.where(Y_truth >= 0.5)
    back_indices = np.where(Y_truth < 0.5)

def evaluate(name="base_0"):
    Y_truth = np.load("predictions/" + name + ".npy")
    Y_pred = np.load("predictions/truth_" + name + ".npy")
    RMSE = np.sqrt(np.mean((Y_truth - Y_pred) ** 2))
    return RMSE
