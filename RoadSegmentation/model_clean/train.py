# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 00:18:49 2019

@author: Justin
"""

import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import  Flatten, Conv2D, UpSampling2D, Conv2DTranspose, MaxPooling2D, Dropout, concatenate, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from random import randint
from skimage.transform import resize
from skimage.filters import gaussian
from skimage.draw import polygon
import pickle

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from common import generate, project_paths, tools

BATCH_SIZE = 64
IM_SIZE = 224
VAL_SIZE = 500
NUM_EPOCHS = 200
EARLY_STOP_PATIENCE = 50
REDUCE_LR_PATIENCE = 20

def get_noise(shape):
    h,w = shape
    noise = np.zeros(shape)
    for _ in range(h//5):
        d = randint(1,h//8)
        x = randint(0, w-d-1)
        y = randint(0, w-d-1)
        noise[y:y+d,x:x+d] += randint(-80,80)/100
    noise = gaussian(np.clip(noise, -1, 1),5)
    return noise 

### Data Loading
def load_data():
    print("Loading data...")
    mask_path = project_paths.TRAIN_GROUNDTRUTH_PATH
    mask_filenames = [str(mask_path/filename) for filename in os.listdir(mask_path) if filename.endswith(".png")]
    mask_path = project_paths.ADD_TRAIN_GROUNDTRUTH_PATH
    mask_filenames += [str(mask_path/filename) for filename in os.listdir(mask_path) if filename.endswith(".png")]
    
    sample_image = mpimg.imread(mask_filenames[0])
    n = len(mask_filenames)
    h,w = sample_image.shape
    masks = np.zeros((n,IM_SIZE,IM_SIZE))
    
    print("\nLoading masks...")
    for i,filename in enumerate(mask_filenames):
        print(".",end="")
        mask = resize(mpimg.imread(filename), (IM_SIZE,IM_SIZE))
        if mask.ndim > 2:
            mask = mask[:,:,0]
        masks[i] = mask
        
    noise_masks = np.zeros((1000,IM_SIZE,IM_SIZE))
    print("\nGenerating noise...")
    for i in range(1000):
        if not (i%50):
            print(".",end="")
        noise_masks[i] = resize(get_noise((h,w)), (IM_SIZE,IM_SIZE))
    return masks, noise_masks

def make_data_generator(masks, noise_masks, idx_list, batch_size):
    _,h, w = masks.shape
    num = len(idx_list)
    num_noise = len(noise_masks)
    
    batch_x = np.zeros((batch_size,IM_SIZE,IM_SIZE,1))
    batch_y = np.zeros((batch_size,IM_SIZE,IM_SIZE,1))
    
    i = 0
    while True:
        mask_idx = idx_list[randint(0, num-1)]
        noise_idx = randint(0, num_noise-1)
        noise_mask = noise_masks[noise_idx].reshape((IM_SIZE,IM_SIZE,1))
        noise_mask = tools.random_transformation(noise_mask)
        batch_y[i] = masks[mask_idx].reshape((IM_SIZE,IM_SIZE,1))
        batch_x[i] = np.clip(batch_y[i] + noise_mask, 0,1)
        i += 1
        if i == batch_size:
            yield batch_x.copy(), batch_y.copy()
            i = 0
            
        poly_mask = np.zeros((h,w))
        for _ in range(randint(1,3)):
            y1, y2, x1, x2 = randint(0,h-1), randint(0,h-1), randint(0,w-1), randint(0,w-1)
            dy, dx = randint(1,5), randint(1,5)
            # (y1,x1) -> (y1+dy, x1+dx) -> (y2+dy, x2+dx) -> (y2,x2)
            r = [y1, y1+dy, y2+dy, y2]
            c = [x1, x1+dx, x2+dx, x2]
            poly_mask[polygon(r,c,shape=(h,w))] = 1
            
        noise_idx = randint(0, num_noise-1)
        noise_mask = noise_masks[noise_idx].reshape((IM_SIZE,IM_SIZE,1))
        noise_mask = tools.random_transformation(noise_mask)
        batch_y[i] = poly_mask.reshape((IM_SIZE,IM_SIZE,1))
        batch_x[i] = np.clip(batch_y[i] + noise_mask, 0,1)
        i += 1
        if i == batch_size:
            yield batch_x.copy(), batch_y.copy()
            i = 0

def BuildModel():
    
    inputs = Input((IM_SIZE,IM_SIZE,1))
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs, conv10)
    
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    print(model.summary())
    return model

def FitModel(model):
    print("Training model clean...")
    
    masks, noise_masks = load_data()
    
    num_masks = masks.shape[0]
    num_train = int(0.8*num_masks)
    
    indices = np.random.permutation(num_masks)
    train_idx, val_idx = indices[:num_train], indices[num_train:]
    
    train_batch_generator = make_data_generator(masks, noise_masks, train_idx, BATCH_SIZE)
    val_batch_generator =  make_data_generator(masks, noise_masks, val_idx, BATCH_SIZE)

    generate.create_folder(project_paths.MODELS_PATH / "clean")
    
    model_checkpoint = ModelCheckpoint(str(project_paths.MODELS_PATH / "clean" / ("clean_{epoch:02d}.h5")), period=5, verbose=1)
    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE, verbose=1)
    reduce_lr = ReduceLROnPlateau(factor=0.5, patience=REDUCE_LR_PATIENCE, verbose=1)
    
    history = model.fit_generator(train_batch_generator,
                        steps_per_epoch = 200,
                        epochs = NUM_EPOCHS,
                        verbose = 2,
                        validation_data = val_batch_generator,
                        validation_steps = (VAL_SIZE//BATCH_SIZE),
                        callbacks = [reduce_lr, early_stopping, model_checkpoint])
    with open(str(project_paths.MODELS_PATH / "clean" /"trainHistoryDict.pkl"), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
        
    # Plotting    
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