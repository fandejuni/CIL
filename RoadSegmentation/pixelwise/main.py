# -*- coding: utf-8 -*-
"""
Created on Thu May 23 21:15:05 2019

@author: Justin Dallant
"""

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

def trainModels(train, groundtruth, name, useDistance=False, useNeighbours=False, usePatch=False, layers=[5, 3, 1], k=5):

    # old_shape = train.shape

    # X = reshape(train)
    # Y = np.reshape(groundtruth, [-1])

    n = len(train) // k

    for i in range(k):
        out_path = "models/" + name + "_" + str(i) + ".h5"
        if not os.path.isfile(out_path):

            print("Training model...", name, i)

            XX = np.concatenate([train[:i*n], train[(i+1)*n:]])
            YY = np.concatenate([groundtruth[:i*n], groundtruth[(i+1)*n:]])

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

            model = nn.train(XX, YY, layers)
            model.save(out_path)
            del model

def savePredictions(train, groundtruth, name, useDistance=False, useNeighbours=False, usePatch=False, k=5):

    if useNeighbours:
        train = augmentNeighbours(train)

    old_shape = train.shape

    X = reshape(train)
    Y = np.reshape(groundtruth, [-1])

    n = len(X) // k

    for i in range(k):
        model_path = "models/" + name + "_" + str(i) + ".h5"
        out_path = "predictions/" + name + "_" + str(i) + ".npy"
        if not os.path.isfile(out_path):

            print("Predicting with model...", name, i)

            XX = np.concatenate([X[:i*n], X[(i+1)*n:]])
            YY = np.concatenate([Y[:i*n], Y[(i+1)*n:]])

            if useDistance:

                new_shape = [old_shape[0] * (k - 1) // k] + list(old_shape[1:])
                print(XX.shape, new_shape)
                XX = np.reshape(XX, new_shape)
                YY = np.reshape(YY, new_shape[:-1])
                print(XX.shape)

                mean_point = distance.computeMeanPoint(XX, YY)
                print("Mean point", mean_point)
                XX = distance.augmentImages(XX, mean_point)

                XX = reshape(XX)
                YY = np.reshape(YY, [-1])

            model = load_model(model_path)
            YYY = model.predict(XX, verbose=1)
            np.save(out_path, YYY)
            YY = np.reshape(YY, [-1])
            np.save("predictions/truth_" + name + "_" + str(i), YY)
            del model



data_path = "/content/gdrive/My Drive/data/"
preproc_path = data_path + "preprocessed/"
groundtruth_path = preproc_path + "groundtruth/"

def saveImages():
    saveImagesOneByOne(data_path + "training/groundtruth/", groundtruth_path)
    saveImagesOneByOne(data_path + "training/images/",  preproc_path + "base/train/")
    saveImagesOneByOne(data_path + "test_images/", preproc_path + "base/test/")

def trainAll(name="base", layers=[5, 5, 1]):

    print("Loading " + name + "...")

    train = tools.openFolder(preproc_path + name + "/train/")
    test = tools.openFolder(preproc_path + name + "/test/")
    groundtruth = tools.openFolder(groundtruth_path)

    trainModels(train, groundtruth, name, layers=layers)
    trainModels(train, groundtruth, name + "_d", layers=layers, useDistance=True)
    trainModels(train, groundtruth, name + "_p", layers=layers, usePatch=True)
    trainModels(train, groundtruth, name + "_dn", layers=layers, useNeighbours=True, useDistance=True)
    trainModels(train, groundtruth, name + "_n", layers=layers, useNeighbours=True)

def generatePredictions(name="base"):

    print("Loading base...")

    train = tools.openFolder(preproc_path + name + "/train/")
    test = tools.openFolder(preproc_path + name + "/test/")
    groundtruth = tools.openFolder(groundtruth_path)

    savePredictions(train, groundtruth, name)
    savePredictions(train, groundtruth, name + "_d", useDistance=True)

def evaluateAll():
    print("Evaluation")
    folder = "predictions/"
    l = os.listdir(folder)
    l = [x[:-4] for x in l if x[:6] != "truth_"]
    for x in l:
        print(x, evaluation.evaluate(x))

saveImages()
trainAll("base")
generatePredictions("base")
trainAll("morpho")
generatePredictions("morpho")

evaluateAll()
