import os
import numpy as np
import matplotlib.image as mpimg
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, ELU, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from random import randint
import pickle

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from common import generate, project_paths, tools

PATCH_SIZE = 8
CONTEXT_SIZE = 80
BATCH_SIZE = 512
VAL_SIZE = 1000
NUM_FEATURES = 7
NUM_EPOCHS = 100
REDUCE_LR_PATIENCE = 20

### Data Loading
def load_data_with_padding():
    print("Loading data...")
    image_path = project_paths.PREPROCESSED_PATH / "morphological" / "train"
    base_filenames = [filename[:-4] for filename in os.listdir(image_path) if filename.endswith(".npy")]
    image_filenames = [str(image_path/(filename+".npy")) for filename in base_filenames]
    mask_path = project_paths.TRAIN_GROUNDTRUTH_PATH
    mask_filenames = [str(mask_path/(filename+".png")) for filename in base_filenames]
    
   
    sample_image = np.load(image_filenames[0])
    num = len(image_filenames)
    h, w, c = sample_image.shape
    padding = (CONTEXT_SIZE-PATCH_SIZE)//2
    images = np.zeros((num,h+2*padding,w+2*padding,c))
    masks = np.zeros((num,h,w))
    print("Loading images...")
    for i,filename in enumerate(image_filenames):
        print(".",end="")
        image = np.load(filename)
        images[i] = tools.pad_with_reflection(image, padding)
    print("\nLoading masks...")
    for i,filename in enumerate(mask_filenames):
        print(".",end="")
        mask = mpimg.imread(filename)
        if mask.ndim > 2:
            mask = mask[:,:,0]
        masks[i] = mask
        
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
            batch_y[i,:] = tools.value_to_class(mask[y:y+PATCH_SIZE,x:x+PATCH_SIZE], 0.25)
            batch_x[i] = tools.random_transformation(image[y:y+CONTEXT_SIZE,x:x+CONTEXT_SIZE].copy())
            i += 1
            if i == batch_size:
                yield batch_x.copy(), batch_y.copy()
                i = 0

def BuildModel():

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding="same", 
                     input_shape=(CONTEXT_SIZE, CONTEXT_SIZE, NUM_FEATURES)))
    model.add(MaxPooling2D((2,2)))
    model.add(ELU())
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding="same"))
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
    
    model.add(Flatten())
    model.add(Dense(128))
    model.add(ELU())
    model.add(Dropout(0.4))

    model.add(Dense(2, activation="softmax"))
    
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['binary_accuracy'])
    print(model.summary())
    return model
        
def FitModel(model):
    print("Training model pre_patch...")
    
    images, masks = load_data_with_padding()
    
    num_images = images.shape[0]
    num_train = int(0.95*num_images)
    
    indices = np.random.permutation(num_images)
    train_idx, val_idx = indices[:num_train], indices[num_train:]
    
    train_batch_generator = make_data_generator(images, masks, train_idx, BATCH_SIZE)
    val_batch_generator =  make_data_generator(images, masks, val_idx, BATCH_SIZE)

    generate.create_folder(project_paths.MODELS_PATH / "morpho_patch")
    
    model_checkpoint = ModelCheckpoint(str(project_paths.MODELS_PATH / "morpho_patch" / ("morpho_patch_{epoch:02d}.h5")), period=5, verbose=2)
    reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.5, patience=REDUCE_LR_PATIENCE, verbose=1)
    
    class_weight = {0: 1., 1: 3.}
    
    history = model.fit_generator(train_batch_generator,
                        steps_per_epoch = 50,
                        epochs = NUM_EPOCHS,
                        verbose = 2,
                        validation_data = val_batch_generator,
                        validation_steps = int(VAL_SIZE/BATCH_SIZE+0.5),
                        callbacks = [reduce_lr, model_checkpoint],
                        class_weight = class_weight)
    with open(str(project_paths.MODELS_PATH / "morpho_patch" /"trainHistoryDict.pkl"), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
        
def main():
    model = BuildModel()
    FitModel(model)

if __name__ == "__main__":
    main()