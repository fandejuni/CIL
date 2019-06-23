import numpy as np
import os
import tools as t

threshold = 0.5

def getRoadPixels(X, groundtruth):
    print(X.shape, groundtruth.shape)
    Y = np.where(groundtruth >= threshold)
    n = len(Y[0])
    dim_pixel = X.shape[-1]
    XX = np.zeros((n, dim_pixel))
    for i in range(n):
        XX[i] = X[Y[0][i], Y[1][i], Y[2][i]]
    return XX

#def computeMeanPoint(images_path="../data/preprocessed/base/train",
#                     groundtruth_path="../data/preprocessed/groundtruth"):

def computeMeanPoint(X, groundtruth):
    road_pixels = getRoadPixels(X, groundtruth)
    return np.mean(road_pixels, axis=0)

    #X = t.openFolder(images_path)
    #groundtruth = t.openFolder(groundtruth_path)

def augmentImages(X, mean_point):
    XX = np.zeros(list(X.shape[:-1]) + [X.shape[-1] + 1])
    XX[:, :, :, :X.shape[-1]] = X
    XX[:, :, :, X.shape[-1]] = np.sqrt(np.sum(np.subtract(X, mean_point) ** 2, axis=-1))
    return XX
