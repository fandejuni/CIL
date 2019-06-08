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
def preprocess(images_path, save_path):
    images = np.load(open(images_path,"rb"))

    img_size0 = images.shape[1]
    img_size1 = images.shape[2]

    def getPixel(index, x, y):
        return images[index, min(max(x, 0), img_size0 - 1), min(max(y, 0), img_size1 - 1)]

    max_size = 2

    n = images.shape[0]
    s1 = images.shape[1]
    s2 = images.shape[2]

    X = np.zeros([n, s1, s2, (max_size + 1) * 3])
    X[:, :, :, :3] = images

    for i in range(n):
        print(i + 1, n)
        for x in range(s1):
            for y in range(s2):
                for size in range(max_size):
                    l = []
                    for (j, (xx, yy)) in enumerate(getNeighbourhood(size + 1)):
                        l.append(getPixel(i, x + xx, y + yy))
                    X[i, x, y, (size + 1)*3:(size+2)*3] = avg(l)
        print(X[i][1][1])

    np.save(save_path, X)

def main():

    print("Small square preprocessing")

    generate.create_folder(project_paths.PREPROCESSED_PATH / "smallsquares")
    preprocess(project_paths.DATA_PATH / "saved" / "test_images.npy", project_paths.PREPROCESSED_PATH / "smallsquares" / "test.npy")
    preprocess(project_paths.DATA_PATH / "saved" / "train_images.npy", project_paths.PREPROCESSED_PATH / "smallsquares" / "train.npy")

if __name__ == "__main__":
    main()
