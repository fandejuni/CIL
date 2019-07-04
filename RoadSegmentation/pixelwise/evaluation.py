# -*- coding: utf-8 -

import numpy as np

n_fold = 5
thresholds = [0.3, 0.5, 0.7, 0.9]
patch_size = 16
img_size = 400
n_images_per_fold = 20

def getRCCBCC(Y_pred, Y_truth, threshold=0.5):

    road = len(np.where(Y_truth >= 0.5)[0])
    back = len(np.where(Y_truth < 0.5)[0])

    both_road = len(np.where((Y_truth >= 0.5) & (Y_pred >= threshold))[0])
    both_back = len(np.where((Y_truth < 0.5) & (Y_pred < threshold))[0])

    RCC = both_road / road
    BCC = both_back / back
    acc = (both_back + both_road) / (road + back)

    return (acc, RCC, BCC)

def backPercentage(Y):
    road = len(np.where(Y >= 0.5)[0])
    back = len(np.where(Y < 0.5)[0])
    n = road + back
    return (back / n)

def evaluate(name="predictions/morpho_dn", truth_name="predictions/truth", thresholds=[0.5]):

    print("Evaluation...", name, truth_name)

    for i in range(n_fold):

        print()
        print(i)

        Y_truth = np.load(truth_name + "_" + str(i) + ".npy")
        Y_truth = np.reshape(Y_truth, [-1])
        Y_pred = np.load(name + "_" + str(i) + ".npy")
        Y_pred = np.reshape(Y_pred, [-1])

        RMSE = np.sqrt(np.mean((Y_truth - Y_pred) ** 2))
        bp_truth = backPercentage(Y_truth)
        bp_pred = backPercentage(Y_pred)

        print("Percentages", "truth", bp_truth, "pred", bp_pred)
        print("RMSE", RMSE)

        for t in thresholds:
            (acc, RCC, BCC) = getRCCBCC(Y_pred, Y_truth, threshold=t)
            print("threshold", t, "acc", acc, "RCC", RCC, "BCC", BCC)

if __name__ == "__main__":
    evaluate(name="cnn_predictions/morpho_tta_post", truth_name="cnn_predictions/truth")
