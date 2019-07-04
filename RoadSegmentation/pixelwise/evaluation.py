# -*- coding: utf-8 -

import numpy as np

n_fold = 5
thresholds = [0.3, 0.5, 0.7, 0.9]
patch_size = 16
img_size = 400
n_images_per_fold = 20

def getValues(Y_pred, Y_truth, threshold=0.5):

    road = len(np.where(Y_truth >= 0.5)[0])
    back = len(np.where(Y_truth < 0.5)[0])

    both_road = len(np.where((Y_truth >= 0.5) & (Y_pred >= threshold))[0])
    both_back = len(np.where((Y_truth < 0.5) & (Y_pred < threshold))[0])

    return (road, back, both_road, both_back)

def evaluate(name="predictions/morpho_dn", truth_name="predictions/truth", t=0.5):

    road_total = 0.0
    back_total = 0.0
    both_road_total = 0.0
    both_back_total = 0.0

    squared_RMSE_total = 0.0

    for i in range(n_fold):

        Y_truth = np.load(truth_name + "_" + str(i) + ".npy")
        Y_truth = np.reshape(Y_truth, [-1])
        Y_pred = np.load(name + "_" + str(i) + ".npy")
        Y_pred = np.reshape(Y_pred, [-1])

        squared_RMSE_total += np.mean((Y_truth - Y_pred) ** 2)
        (road, back, both_road, both_back) = getValues(Y_pred, Y_truth, threshold=t)

        road_total += road
        back_total += back
        both_road_total += both_road
        both_back_total += both_back

    RMSE = np.sqrt(squared_RMSE_total / n_fold)
    acc = (both_back_total + both_road_total) / (road_total + back_total)
    RCC = both_road_total / road_total
    BCC = both_back_total / back_total
    # back_percent = back_total / (back_total + road_total)

    return (name.split("/")[-1], "%.3f" % acc, "%.3f" % RMSE, "%.3f" % RCC, "%.3f" % BCC)

if __name__ == "__main__":
    evaluate(name="cnn_predictions/morpho_tta_post", truth_name="cnn_predictions/truth")
