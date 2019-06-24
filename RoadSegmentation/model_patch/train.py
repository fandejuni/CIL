# -*- coding: utf-8 -
"""
Created on Thu June 3 21:48:05 2019

@author: Thibault Dardinier
"""

import os
import numpy as np
import matplotlib.image as mpimg
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, Adadelta
from tensorflow.keras.layers import BatchNormalization, Dense, LeakyReLU, Flatten, Conv2D, Convolution2D, UpSampling2D, Conv2DTranspose, MaxPooling2D, Dropout, concatenate, Input, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K
from random import randint

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from common import generate, project_paths, tools

PATCH_SIZE = 8
CONTEXT_SIZE = 96
BATCH_SIZE = 50
VAL_SIZE = 500
NUM_FEATURES = 3
NUM_EPOCHS = 200
EARLY_STOP_PATIENCE = 100
REDUCE_LR_PATIENCE = 10

### Data Loading
def load_data_with_padding():
    print("Loading data...")
    image_path = project_paths.TRAIN_IMAGE_PATH
    base_filenames = [filename[:-4] for filename in os.listdir(image_path) if filename.endswith(".png")]
    image_filenames = [str(image_path/(filename+".png")) for filename in base_filenames]
    mask_path = project_paths.TRAIN_GROUNDTRUTH_PATH
    mask_filenames = [str(mask_path/(filename+".png")) for filename in base_filenames]
    
    sample_image = mpimg.imread(image_filenames[0])
    num = len(image_filenames)
    h, w, c = sample_image.shape
    padding = (CONTEXT_SIZE-PATCH_SIZE)//2
    images = np.zeros((num,h+2*padding,w+2*padding,c))
    masks = np.zeros((num,h,w))
    print("Loading images...")
    for i,filename in enumerate(image_filenames):
      print(".",end="")
      images[i] = tools.pad_with_reflection(mpimg.imread(filename), padding)
    print("\nLoading masks...")
    for i,filename in enumerate(mask_filenames):
      print(".",end="")
      masks[i] = mpimg.imread(filename)
    print("\nDone loading!")
    return images, masks

def make_data_generator(images, masks, idx_list, batch_size):
    
    _,h, w, c = images.shape
    num = len(idx_list)
    
    batch_x = np.zeros((batch_size,CONTEXT_SIZE,CONTEXT_SIZE, c))
    batch_y = np.zeros((batch_size,2))

    i = 0
    while True:
        idx = idx_list[randint(0,num-1)]
        image = images[idx]
        mask = masks[idx]
    
        for _ in range(5):
            x = randint(0,w-CONTEXT_SIZE-1)
            y = randint(0,h-CONTEXT_SIZE-1)
            batch_y[i,:] = tools.value_to_class(mask[y:y+PATCH_SIZE,x:x+PATCH_SIZE], 0.5)
            if (batch_y[i][1] == 0) and randint(1,10) < 6:
                continue
            batch_x[i] = tools.random_transformation(image[y:y+CONTEXT_SIZE,x:x+CONTEXT_SIZE].copy())
            i += 1
            if i == batch_size:
                yield batch_x.copy(), batch_y.copy()
                i = 0

def BuildModel():

    model = Sequential()

    model.add(Conv2D(32, (5, 5), padding="same", 
                     input_shape=(CONTEXT_SIZE, CONTEXT_SIZE, NUM_FEATURES)))
    model.add(MaxPooling2D((2,2)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(MaxPooling2D((2,2)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(MaxPooling2D((2,2)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(MaxPooling2D((2,2)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(512, (3, 3), padding="same"))
    model.add(MaxPooling2D((2,2)))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=00.1))
    model.add(Dropout(0.25))

    model.add(Dense(2, activation="softmax"))
    
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['binary_accuracy'])
    print(model.summary())
    return model

def FitModel(model):
    print("Training model patch...")
    
    images, masks = load_data_with_padding()
    
    num_images = images.shape[0]
    num_train = int(0.8*num_images)
    
    indices = np.random.permutation(num_images)
    train_idx, val_idx = indices[:num_train], indices[num_train:]
    
    train_batch_generator = make_data_generator(images, masks, train_idx, BATCH_SIZE)
    val_batch_generator =  make_data_generator(images, masks, val_idx, VAL_SIZE)

    generate.create_folder(project_paths.MODELS_PATH / "patch")
    
    model_checkpoint = ModelCheckpoint(str(project_paths.MODELS_PATH / "patch" / ("patch.h5")), save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE, verbose=1)
    
    class_weight = {0: 1., 1: 1.}
    
    model.fit_generator(train_batch_generator,
                        steps_per_epoch = 100,
                        epochs = NUM_EPOCHS,
                        verbose = 1,
                        validation_data = val_batch_generator,
                        validation_steps = 1,
                        callbacks = [early_stopping, model_checkpoint],
                        class_weight = class_weight)

def main():
    model = BuildModel()
    FitModel(model)

if __name__ == "__main__":
    main()
