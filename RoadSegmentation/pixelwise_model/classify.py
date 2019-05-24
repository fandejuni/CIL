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
from project_paths import DATA_PATH, MODEL_PATH


### Data Loading
def LoadTestData(test_path):
    images = np.load(open(test_path, "rb"))
    images = np.reshape(images, (-1, 3))
    base_path, extension = str(test_path).split('.')
    ids = np.load(open(base_path + "_ids." + extension, "rb"))
    return images, ids


model = load_model(MODEL_PATH / "model.h5")

X, ids = LoadTestData(DATA_PATH / "saved" / "test_images.npy")

y = model.predict(X, verbose=1)
y = np.reshape(y, (-1, IM_HEIGHT, IM_WIDTH))

for i in range(len(ids)):
    I = y[i]
    I8 = (((I - I.min()) / (I.max() - I.min())) * 255.9).astype(np.uint8)
    img = Image.fromarray(I8)
    img.save(MODEL_PATH / "prediction" / "prediction_{}.png".format(ids[i]))
    