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
def LoadTrainData(images_path, groundtruth_path, prop = 0.8):
    images = np.load(open(images_path,"rb"))
    groundtruth = np.load(open(groundtruth_path,"rb"))

    X = np.reshape(images, (-1, 3))
    y = np.reshape(groundtruth, (-1))

    num_tot = X.shape[0]
    num_training = int(num_tot*prop)
    indices = np.random.permutation(num_tot)
    training_idx, test_idx = indices[:num_training], indices[num_training:]

    X_train, X_test = X[training_idx], X[test_idx]
    y_train, y_test = y[training_idx], y[test_idx]

    return (X_train, y_train), (X_test, y_test)

def main():

    print("Pixelwise training")

    (X_train, y_train), (X_test, y_test) = LoadTrainData(project_paths.DATA_PATH / "saved" / "train_images.npy", 
                                            project_paths.DATA_PATH / "saved" / "train_groundtruth.npy")

    #### Model
    model = Sequential()
    model.add(Dense(5, activation='sigmoid'))
    model.add(Dense(5, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #print(model.summary())

    #### Fitting
    model.fit(X_train, y_train, epochs=1, batch_size=64)

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    generate.create_folder(project_paths.MODELS_PATH / "pixelwise")
    model.save(project_paths.MODELS_PATH / "pixelwise" / "model.h5")

if __name__ == "__main__":
    main()
