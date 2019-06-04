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

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from common import project_paths, generate

### Data Loading
def LoadTestData(test_path):
    images = np.load(open(test_path, "rb"))
    images = np.reshape(images, (-1, 3))
    base_path, extension = str(test_path).split('.')
    ids = np.load(open(base_path + "_ids." + extension, "rb"))
    return images, ids

def main():

    model = load_model(project_paths.MODELS_PATH / "pixelwise" / "model.h5")

    def get_prediction(img):
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
    generate.run_predictions_test_set("pixelwise", get_prediction)

if __name__ == "__main__":
    main()
