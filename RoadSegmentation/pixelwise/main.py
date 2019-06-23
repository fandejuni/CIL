# -*- coding: utf-8 -*-
"""
Created on Thu May 23 21:15:05 2019

@author: Justin Dallant
"""

import os
import numpy as np
import matplotlib.image as mpimg

import tools
import distance
import neighbours

def saveImagesOneByOne(in_path, out_path):
    tools.create_folder(out_path)
    for filename in tools.getImages(in_path, out_path, ".png"):
        print("Saving...", filename)
        image = mpimg.imread(in_path + filename)
        np.save(out_path + filename[:-4], image)

def shuffle(X):
    np.random.seed(42)
    np.random.shuffle(X)

def augmentNeighbours(X):
    XX = np.zeros(list(X.shape[:-1]) + [X.shape[-1] * 9])
    for i in range(X.shape[0]):
        print("Augmenting neighbourhood...", i, X.shape[0])
        XX[i] = neighbours.augmentImage(X[i])
    X = np.reshape(XX, [-1, XX.shape[-1]])
    return X

def reshape(X):
    return np.reshape(X, [-1, X.shape[-1]])

# if __name__ == '__main__':

data_path = "../data/"
preproc_path = data_path + "preprocessed/"
groundtruth_path = preproc_path + "groundtruth/"

saveImagesOneByOne(data_path + "training/groundtruth/", groundtruth_path)

saveImagesOneByOne(data_path + "training/images/",  preproc_path + "base/train/")
saveImagesOneByOne(data_path + "test_images/", preproc_path + "base/test/")

print("Loading base...")

train = tools.openFolder(preproc_path + "base/train/")
test = tools.openFolder(preproc_path + "base/test/")
groundtruth = tools.openFolder(groundtruth_path)

print("Augmenting base...")

mean_point = distance.computeMeanPoint(train, groundtruth)
train_d = distance.augmentImages(train, mean_point)
test_d = distance.augmentImages(test, mean_point)

print("Loading morpho...")

train_morpho = tools.openFolder(preproc_path + "morpho/train/")
test_morpho = tools.openFolder(preproc_path + "morpho/train/")

print("Augmenting morpho")

mean_point_morpho = distance.computeMeanPoint(train_morpho, groundtruth)
train_morpho_d = distance.augmentImages(train_morpho, mean_point_morpho)
test_morpho_d = distance.augmentImages(test_morpho, mean_point_morpho)

# mean_point_base = distance.computeMeanPoint(preproc_path + "base/train/", groundtruth_path)
# augmentDistance(preproc_path + "base/train/", preproc_path + "base_d/train/", mean_point_base)
# augmentDistance(preproc_path + "base/test/", preproc_path + "base_d/test/", mean_point_base)

# mean_point_morpho = distance.computeMeanPoint(preproc_path + "morpho/train/", groundtruth_path)
# augmentDistance(preproc_path + "morpho/train/", preproc_path + "morpho_d/train/", mean_point_morpho)
# augmentDistance(preproc_path + "morpho/test/", preproc_path + "morpho_d/test/", mean_point_morpho)

# groundtruth = np.reshape(groundtruth, [-1])

# np.random.seed(42)
# np.random.shuffle(groundtruth)
