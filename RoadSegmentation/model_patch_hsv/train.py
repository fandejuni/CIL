# -*- coding: utf-8 -
"""
Created on Thu June 3 21:48:05 2019

@author: Thibault Dardinier
"""

import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, Adadelta
from tensorflow.keras.layers import BatchNormalization, Dense, LeakyReLU, ELU, Flatten, Conv2D, Convolution2D, UpSampling2D, Conv2DTranspose, MaxPooling2D, Dropout, concatenate, Input, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from random import randint
from matplotlib.colors import rgb_to_hsv
import pickle

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from common import generate, project_paths, tools

PATCH_SIZE = 8
CONTEXT_SIZE = 80
BATCH_SIZE = 128
VAL_SIZE = 2000
NUM_FEATURES = 3
NUM_EPOCHS = 200
EARLY_STOP_PATIENCE = 30
REDUCE_LR_PATIENCE = 5

### Data Loading
def load_data_with_padding():
    print("Loading data...")
    image_path = project_paths.TRAIN_IMAGE_PATH
    base_filenames = [filename[:-4] for filename in os.listdir(image_path) if filename.endswith(".png")]
    image_filenames = [str(image_path/(filename+".png")) for filename in base_filenames]
    mask_path = project_paths.TRAIN_GROUNDTRUTH_PATH
    mask_filenames = [str(mask_path/(filename+".png")) for filename in base_filenames]
    
    sample_image = mpimg.imread(image_filenames[0])
    print(sample_image.min(), sample_image.max())
    num = len(image_filenames)
    h, w, c = sample_image.shape
    padding = (CONTEXT_SIZE-PATCH_SIZE)//2
    images = np.zeros((num,h+2*padding,w+2*padding,c))
    masks = np.zeros((num,h,w))
    print("Loading images...")
    for i,filename in enumerate(image_filenames):
        print(".",end="")
        images[i] = rgb_to_hsv(tools.pad_with_reflection(mpimg.imread(filename), padding))
    print("\nLoading masks...")
    for i,filename in enumerate(mask_filenames):
        print(".",end="")
        masks[i] = mpimg.imread(filename)
    print("\nDone loading!")
    return images, masks

def make_data_generator(images, masks, idx_list, batch_size):
    
    """
    datagen = ImageDataGenerator(rotation_range=90, zoom_range=0.1, channel_shift_range=0.05, fill_mode="reflect", 
                                 horizontal_flip=True, vertical_flip=True)
    """
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
            batch_y[i,:] = tools.value_to_class(mask[y:y+PATCH_SIZE,x:x+PATCH_SIZE], 0.25)
            #batch_x[i] = datagen.random_transform(image[y:y+CONTEXT_SIZE,x:x+CONTEXT_SIZE])/255
            batch_x[i] = tools.random_transformation(image[y:y+CONTEXT_SIZE,x:x+CONTEXT_SIZE].copy())
            i += 1
            if i == batch_size:
                yield batch_x.copy(), batch_y.copy()
                i = 0

def BuildModel():

    model = Sequential()

    model.add(Conv2D(64, (5, 5), padding="same", 
                     input_shape=(CONTEXT_SIZE, CONTEXT_SIZE, NUM_FEATURES)))
    model.add(MaxPooling2D((2,2)))
    model.add(ELU())
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(MaxPooling2D((2,2)))
    model.add(ELU())
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(MaxPooling2D((2,2)))
    model.add(ELU())
    model.add(Dropout(0.2))
    
    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(MaxPooling2D((2,2)))
    model.add(ELU())
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(ELU())
    model.add(Dropout(0.4))

    model.add(Dense(2, activation="softmax"))
    
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    print(model.summary())
    return model

def FitModel(model):
    print("Training model patch_hsv...")
    
    images, masks = load_data_with_padding()
    
    num_images = images.shape[0]
    num_train = int(0.8*num_images)
    
    indices = np.random.permutation(num_images)
    train_idx, val_idx = indices[:num_train], indices[num_train:]
    
    train_batch_generator = make_data_generator(images, masks, train_idx, BATCH_SIZE)
    val_batch_generator =  make_data_generator(images, masks, val_idx, VAL_SIZE)

    generate.create_folder(project_paths.MODELS_PATH / "patch_hsv")
    
    model_checkpoint = ModelCheckpoint(str(project_paths.MODELS_PATH / "patch_hsv" / ("patch_hsv_{epoch:02d}.h5")), period=5, verbose=1)
    early_stopping = EarlyStopping(monitor='acc', patience=EARLY_STOP_PATIENCE, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=REDUCE_LR_PATIENCE, verbose=1)
    
    class_weight = {0: 1., 1: 4.}
    
    history = model.fit_generator(train_batch_generator,
                        steps_per_epoch = 500,
                        epochs = NUM_EPOCHS,
                        verbose = 1,
                        validation_data = val_batch_generator,
                        validation_steps = 1,
                        callbacks = [reduce_lr, early_stopping, model_checkpoint],
                        class_weight = class_weight)
    with open(str(project_paths.MODELS_PATH / "patch_hsv" /"trainHistoryDict.pkl"), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
        
        
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    ## Plot loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    ## Plot accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
def main():
    model = BuildModel()
    FitModel(model)

if __name__ == "__main__":
    main()