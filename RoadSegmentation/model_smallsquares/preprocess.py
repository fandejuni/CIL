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


### Data Loading
def preprocess(images_path, save_path):
    images = np.load(open(images_path,"rb"))

    img_size0 = images.shape[1]
    img_size1 = images.shape[2]

    def getPixel(index, x, y):
        return images[index, min(max(x, 0), img_size0 - 1), min(max(y, 0), img_size1 - 1)]

    size = 1
    n = images.shape[0]
    s1 = images.shape[1]
    s2 = images.shape[2]

    X = np.zeros([n, s1, s2, 15])
    X[:, :, :, :3] = images

    for i in range(n):
        print(i + 1, n)
        for x in range(s1):
            for y in range(s2):
                for (j, (xx, yy)) in enumerate([(-1, 0), (0, -1), (1, 0), (0, 1)]):
                    X[i, x, y, (j + 1)*3:(j+2)*3] = getPixel(i, x + xx, y + yy)
        print(X[i][1][1])

    np.save(save_path, X)

def main():

    print("Small square preprocessing")

    generate.create_folder(project_paths.PREPROCESSED_PATH / "smallsquares")
    preprocess(project_paths.DATA_PATH / "saved" / "test_images.npy", project_paths.PREPROCESSED_PATH / "smallsquares" / "test.npy")
    preprocess(project_paths.DATA_PATH / "saved" / "train_images.npy", project_paths.PREPROCESSED_PATH / "smallsquares" / "train.npy")

if __name__ == "__main__":
    main()
