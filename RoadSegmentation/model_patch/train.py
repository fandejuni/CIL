# -*- coding: utf-8 -*-
"""
Created on Thu May 23 21:48:05 2019

@author: Justin Dallant
"""

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from common.project_paths import DATA_PATH, MODEL_PATH


### Data Loading
def LoadTrainData(images_path, groundtruth_path, prop = 0.8):
    images = np.load(open(images_path,"rb"))
    groundtruth = np.load(open(groundtruth_path,"rb"))

    size = 5
    rsize = 400
    nsize = int(rsize / size)
    print(nsize)

    patched_images = np.zeros([100, nsize, nsize, 3])
    patched_groundtruth = np.zeros([100, nsize, nsize])

    for im in range(100):
        print("Patching...", str(im + 1) + "/100")
        for x in range(nsize):
            for y in range(nsize):
                for c in range(3):
                    v = np.mean(images[im, x*size:(x+1)*size, y*size:(y+1)*size, c])
                    patched_images[im, x, y, c] = v

                    delta = abs(images[im, x, y, c] - v)

                v = np.mean(groundtruth[im, x*size:(x+1)*size, y*size:(y+1)*size])
                patched_groundtruth[im, x, y] = v

                delta = abs(groundtruth[im, x, y] - v)

    print("KOIS")

    X = np.reshape(patched_images, (-1, 3))
    y = np.reshape(patched_groundtruth, (-1))

    num_tot = X.shape[0]
    num_training = int(num_tot*prop)
    indices = np.random.permutation(num_tot)
    training_idx, test_idx = indices[:num_training], indices[num_training:]

    X_train, X_test = X[training_idx], X[test_idx]
    y_train, y_test = y[training_idx], y[test_idx]

    return (X_train, y_train), (X_test, y_test)


def main():

    print("Patch training")

    (X_train, y_train), (X_test, y_test) = LoadTrainData(DATA_PATH / "saved" / "train_images.npy", 
                                            DATA_PATH / "saved" / "train_groundtruth.npy")

    #### Model
    model = Sequential()
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(5, activation='sigmoid'))
    model.add(Dense(5, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #print(model.summary())

    #### Fitting
    model.fit(X_train, y_train, epochs=2, batch_size=32)

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    model.save(MODEL_PATH / "model.h5")

if __name__ == "__main__":
    main()
