# -*- coding: utf-8 -*-
"""
Created on Thu May 23 21:15:05 2019
"""

# Settings for Google Colab
# local_folder = "/content/gdrive/My Drive/"

# Settings for local use
local_folder = "./"

import os
import numpy as np
import matplotlib.image as mpimg
from tensorflow.keras.models import load_model

import tools
import distance
import neighbours
import patch
import nn
import evaluation

from tabulate import tabulate

def saveImagesOneByOne(in_path, out_path):
    tools.create_folder(out_path)
    for filename in tools.getImages(in_path, out_path, ".png"):
        print("Saving...", filename)
        image = mpimg.imread(in_path + filename)
        np.save(out_path + filename[:-4], image)

def augmentNeighbours(X):
    XX = np.zeros(list(X.shape[:-1]) + [X.shape[-1] * 9])
    for i in range(X.shape[0]):
        print("Augmenting neighbourhood...", i, X.shape[0])
        XX[i] = neighbours.augmentImage(X[i])
    X = np.reshape(XX, [-1, XX.shape[-1]])
    return X

def reshape(X):
    return np.reshape(X, [-1, X.shape[-1]])

def getXX_YY(train, groundtruth, i, n, useDistance, useNeighbours, usePatch, training=True):

    if training:
        XX = np.concatenate([train[:i*n], train[(i+1)*n:]])
        YY = np.concatenate([groundtruth[:i*n], groundtruth[(i+1)*n:]])
    else:
        XX = train[i*n:(i+1)*n]
        YY = groundtruth[i*n:(i+1)*n]

    if useDistance:
        mean_point = distance.computeMeanPoint(XX, YY)
        XX = distance.augmentImages(XX, mean_point)

    if useNeighbours:
        XX = augmentNeighbours(XX)

    if usePatch:
        XX = patch.convertImages(XX)
        YY = patch.convertGroundtruth(YY)

    XX = reshape(XX)
    YY = np.reshape(YY, [-1])

    return (XX, YY)

def trainModels(train, groundtruth, name, useDistance=False, useNeighbours=False, usePatch=False, layers=[10, 5, 1], k=5):

    n = len(train) // k

    for i in range(k):
        out_path = local_folder + "pixelwise/models/" + name + "_" + str(i) + ".h5"
        if not os.path.isfile(out_path):

            print("Training model...", name, i)

            (XX, YY) = getXX_YY(train, groundtruth, i, n, useDistance, useNeighbours, usePatch)

            model = nn.train(XX, YY, layers)
            model.save(out_path)

            del model
            del XX
            del YY

def savePredictions(train, groundtruth, name, useDistance=False, useNeighbours=False, usePatch=False, k=5):

    n = len(train) // k

    for i in range(k):
        model_path = local_folder + "pixelwise/models/" + name + "_" + str(i) + ".h5"
        out_path = local_folder + "pixelwise/predictions/" + name + "_" + str(i) + ".npy"
        if not os.path.isfile(out_path):

            print("Predicting with model...", name, i)

            (XX, YY) = getXX_YY(train, groundtruth, i, n, useDistance, useNeighbours, usePatch, training=False)

            model = load_model(model_path)
            YYY = model.predict(XX, verbose=1)
            np.save(out_path, YYY)
            np.save(local_folder + "pixelwise/predictions/truth_" + name + "_" + str(i), YY)

            del model
            del XX
            del YY
            del YYY

data_path = local_folder + "data/"
preproc_path = data_path + "preprocessed/"
groundtruth_path = preproc_path + "groundtruth/"

base_path = preproc_path + "base"
morpho_path = local_folder + "generated/preprocessed/morphological"

def saveImages():
    saveImagesOneByOne(data_path + "training/groundtruth/", groundtruth_path)
    saveImagesOneByOne(data_path + "training/images/",  preproc_path + "base/train/")
    # saveImagesOneByOne(data_path + "test_images/", preproc_path + "base/test/")

def trainAll(name="base"):

    print("Loading " + name + "...")

    if name == "base":
        folder = base_path
    else:
        folder = morpho_path

    train = tools.openFolder(folder + "/train/")
    # test = tools.openFolder(preproc_path + name + "/test/")
    groundtruth = tools.openFolder(groundtruth_path)

    trainModels(train, groundtruth, name, layers=[10, 5, 1])
    trainModels(train, groundtruth, name + "_d", layers=[10, 5, 1], useDistance=True)
    trainModels(train, groundtruth, name + "_n", layers=[30, 10, 1], useNeighbours=True)
    trainModels(train, groundtruth, name + "_p", layers=[20, 10, 1], usePatch=True)
    trainModels(train, groundtruth, name + "_dn", layers=[30, 10, 1], useDistance=True, useNeighbours=True)

def generatePredictions(name="base"):

    print("Loading " + name + "...")

    if name == "base":
        folder = base_path
    else:
        folder = morpho_path

    train = tools.openFolder(folder + "/train/")
    # test = tools.openFolder(preproc_path + name + "/test/")
    groundtruth = tools.openFolder(groundtruth_path)

    savePredictions(train, groundtruth, name)
    savePredictions(train, groundtruth, name + "_d", useDistance=True)
    savePredictions(train, groundtruth, name + "_p", usePatch=True)
    savePredictions(train, groundtruth, name + "_n", useNeighbours=True)
    savePredictions(train, groundtruth, name + "_dn", useDistance=True, useNeighbours=True)

def evaluateAll():

    print("Evaluation...")

    results = []

    folder_pred = local_folder + "generated/models/patch/"
    # folder_pred = local_folder + "pixelwise/cnn_predictions/"
    l = os.listdir(folder_pred)
    l = [x[:-6] for x in l if x[:6] != "truth_"]
    l = list(set(l))

    truth = folder_pred + "truth"
    for x in l:
        name = folder_pred + x
        results.append(evaluation.evaluate(name=name, truth_name=truth))

    results = [["cnn_" + x[0]] + x[1:] for x in results]


    folder_pred = local_folder + "pixelwise/predictions/"
    l = os.listdir(folder_pred)
    l = [x[:-6] for x in l if x[:6] != "truth_"]
    l = list(set(l))

    for x in l:
        name = folder_pred + x
        truth = folder_pred + "truth_" + x
        results.append(evaluation.evaluate(name=name, truth_name=truth))

    results.sort(key = lambda x: -float(x[1]))
    print(tabulate(results, headers=['Name', 'Acc', 'RMSE', 'RCC', 'BCC'], tablefmt='orgtbl'))


if __name__ == "__main__":
    saveImages()
    trainAll("base")
    trainAll("morpho")
    generatePredictions("base")
    generatePredictions("morpho")
    evaluateAll()
