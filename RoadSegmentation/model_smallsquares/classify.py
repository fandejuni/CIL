# -*- coding: utf-8 -*-
"""
Created on Thu May 23 22:38:01 2019

@author: Justin Dallant
"""


IM_HEIGHT = 608
IM_WIDTH = 608

import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

import conf

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from common import project_paths, generate

def main():

    model = load_model(project_paths.MODELS_PATH / conf.name / conf.model)

    def get_prediction(img):
        print(img.shape)
        y = model.predict(img)
        y = np.reshape(y, (IM_HEIGHT, IM_WIDTH))
        Y = (y - y.min()) / (y.max() - y.min())
        return Y

    #X, ids = LoadTestData(project_paths.DATA_PATH / "saved" / "test_images.npy")

    #y = model.predict(X, verbose=1)
    #y = np.reshape(y, (-1, IM_HEIGHT, IM_WIDTH))

    #for i in range(len(ids)):
        #I = y[i]
        #I8 = (((I - I.min()) / (I.max() - I.min())) * 255.9).astype(np.uint8)
        #img = Image.fromarray(I8)
        #generate.create_folder(project_paths.RESULTS_TEST_PATH / "pixelwise")
        #img.save(project_paths.RESULTS_TEST_PATH / "pixelwise" / "prediction_{}.png".format(ids[i]))
    if conf.morpho:

        images = np.zeros([0, 608, 608, 7])
        path = project_paths.PREPROCESSED_PATH / "morphological" / "test"
        for f in sorted(os.listdir(path)):
        # for i in range(1, 101):
            # f = "satImage_" + str(i).zfill(3) + ".npy"
            image = np.load(open(path / f, "rb"))
            image = np.reshape(image, [1, 608, 608, 7])
            images = np.append(images, image, axis=0)
            print(f, images.shape)

        images = np.reshape(images, [94, 608, 608, 7])

    else:
        images = np.load(open(project_paths.PREPROCESSED_PATH / "smallsquares" / "test.npy", "rb"))

    generate.run_predictions_test_set(conf.name, get_prediction, images)

if __name__ == "__main__":
    main()
