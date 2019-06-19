# -*- coding: utf-8 -*-
"""
Created on Thu May 23 22:38:01 2019

@author: Justin Dallant
"""


IM_HEIGHT = 608
IM_WIDTH = 608

import os
from os import sys
import re
import numpy as np
import matplotlib.image as mpimg
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, LeakyReLU, Flatten, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, concatenate, Input, UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K
from random import randint

if __name__ == '__main__' and __package__ is None:
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from common import generate, project_paths, tools, mask_to_submission

MODEL_NAME = "patch"
PATCH_SIZE = 8
CONTEXT_SIZE = 96

### Data Loading

def main():
    print("b Classifying with {}...".format(MODEL_NAME))
    model_path = str(project_paths.MODELS_PATH / "patch" / (MODEL_NAME+".h5"))
    
    model = load_model(model_path, custom_objects={})

    image_path = project_paths.TEST_DIR_PATH
    filenames = [filename for filename in os.listdir(image_path) if filename.endswith(".png")]
    full_filenames = [str(image_path/filename) for filename in filenames]
    
    generate.create_folder(project_paths.MASKS_TEST_PATH / "patch")
    generate.create_folder(project_paths.RESULTS_TEST_PATH / "patch")
    generate.create_folder(project_paths.RESULTS_TEST_PATH / "patchogray")
    
    for file_idx,filename in enumerate(full_filenames):
        base_name, _ = filenames[file_idx].split(".")
        print("Generating mask for {}...".format(base_name))
        image = np.load(filename)
        imgheight = image.shape[0]
        imgwidth = image.shape[1]
        image = tools.pad_with_reflection(image, (CONTEXT_SIZE-PATCH_SIZE)//2)
        list_patches = tools.img_crop(image, CONTEXT_SIZE, CONTEXT_SIZE, PATCH_SIZE, PATCH_SIZE)
        list_labels = model.predict(np.array(list_patches), verbose=1)
        mask = np.zeros((imgheight, imgwidth))
        gray_mask = np.zeros((imgheight, imgwidth))
        
        idx = 0
        for i in range(0,imgheight,PATCH_SIZE):
            for j in range(0,imgwidth,PATCH_SIZE):
                label = list_labels[idx]
                idx += 1
                gray_mask[i:i+PATCH_SIZE, j:j+PATCH_SIZE] = label[1]
                if (label[1] > 0.8):
                    mask[i:i+PATCH_SIZE, j:j+PATCH_SIZE] = 1
        
        mask_img = tools.solution_to_img(mask)
        mpimg.imsave(project_paths.MASKS_TEST_PATH / "patch" / (base_name+".png"), mask_img)
        mpimg.imsave(project_paths.RESULTS_TEST_PATH / "patchgray" / (base_name+".png"), gray_mask, cmap="gray")
        
        img = mpimg.imread(str(project_paths.TEST_DIR_PATH / (base_name+".png")))
        overlay = tools.make_img_overlay(img, mask)
        mpimg.imsave(project_paths.RESULTS_TEST_PATH / "patch" / (base_name+".png"), overlay)
    
    
    mask_path = project_paths.MASKS_TEST_PATH / "patch"
    image_filenames =  [str(mask_path / filename) for filename in os.listdir(mask_path) if filename.endswith(".png")]
    mask_to_submission.masks_to_submission(project_paths.SUBMISSIONS_PATH / "patch" / "test0.csv", *image_filenames)

        
if __name__ == "__main__":
    main()
    

