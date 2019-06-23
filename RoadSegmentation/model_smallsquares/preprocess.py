# -*- coding: utf-8 -
"""
Created on Thu June 3 21:48:05 2019

@author: Thibault Dardinier
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from common import generate, project_paths

def l1(x):
    return max(abs(x[0]), abs(x[1]))

# L1 norm
def getNeighbourhood(size=1):
    center = (0, 0)
    l = []
    for x in range(-size, size + 1):
        for y in range(-size, size + 1):
            if l1((x, y)) == size:
                l.append((x, y))
    return l

def avg(l):
    n = 0.0
    x = np.zeros(l[0].shape)
    for xx in l:
        x += xx
        n += 1.0
    return x / n

### Data Loading
def augment_neighbours(image_path, save_path):
    image = np.load(open(image_path,"rb"))

    s1 = image.shape[0]
    s2 = image.shape[1]
    n_features = image.shape[2]

    def getPixel(x, y):
        return image[min(max(x, 0), s1 - 1), min(max(y, 0), s2 - 1)]

    max_size = 2

    X = np.zeros([s1, s2, (max_size + 1) * n_features])
    X[:, :, :n_features] = image

    for x in range(s1):
        for y in range(s2):
            for size in range(max_size):
                l = []
                for (j, (xx, yy)) in enumerate(getNeighbourhood(size + 1)):
                    l.append(getPixel(x + xx, y + yy))
                X[x, y, (size + 1)*n_features:(size+2)*n_features] = avg(l)

    np.save(save_path, X)

def main():

    print("Small square preprocessing")

    generate.create_folder(project_paths.PREPROCESSED_PATH / "smallsquares")
    preprocess(project_paths.DATA_PATH / "saved" / "test_images.npy", project_paths.PREPROCESSED_PATH / "smallsquares" / "test.npy")
    preprocess(project_paths.DATA_PATH / "saved" / "train_images.npy", project_paths.PREPROCESSED_PATH / "smallsquares" / "train.npy")

if __name__ == "__main__":
    main()
