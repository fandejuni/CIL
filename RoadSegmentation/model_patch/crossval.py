import numpy as np
from tensorflow.keras.callbacks import ReduceLROnPlateau
import gc
from keras.preprocessing.image import ImageDataGenerator
from skimage.morphology import disk, binary_closing, binary_opening, binary_erosion, remove_small_objects, remove_small_holes
from train import load_data_with_padding, make_data_generator, BuildModel

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from common import generate, project_paths, tools

PATCH_SIZE = 8
CONTEXT_SIZE = 80
BATCH_SIZE = 512
VAL_SIZE = 1000
NUM_FEATURES = 3
NUM_EPOCHS = 200
REDUCE_LR_PATIENCE = 20

def post_processing(input_mask):
    mask = input_mask.copy()
    mask = (mask>0.6)
    mask = binary_opening(mask, disk(5))
    mask = binary_closing(mask, disk(5))
    remove_small_objects(mask, 8*8*10, in_place=True)
    remove_small_holes(mask, 8*8*3, in_place=True)
    mask = binary_erosion(mask, disk(4))
    return mask.astype(np.uint8)


def CrossVal():
    print("Training model pre_patch...")
    
    images, masks = load_data_with_padding()
    
    num_images = images.shape[0]
    num_val = int(0.2*num_images)
    
    np.random.seed(3)
    indices = np.random.permutation(num_images)

    generate.create_folder(project_paths.MODELS_PATH / "patch")
    
    tta_gen = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        rotation_range=20.,
        fill_mode='reflect',
        width_shift_range = 0.05, 
        height_shift_range = 0.05)
    
    reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.5, patience=REDUCE_LR_PATIENCE, verbose=1)
    class_weight = {0: 1., 1: 3.}
        
    for k in range(5):
        print("Making fold {}...".format(k))
        model = BuildModel()
        val_idx = indices[k*num_val:(k+1)*num_val]
        train_idx = np.concatenate([indices[:k*num_val], indices[(k+1)*num_val:]])
        train_batch_generator = make_data_generator(images, masks, train_idx, BATCH_SIZE)
        
        model.fit_generator(train_batch_generator,
                            steps_per_epoch = 50,
                            epochs = NUM_EPOCHS,
                            verbose = 2,
                            callbacks = [reduce_lr],
                            class_weight = class_weight)
    
    
        val_imgs, val_masks = images[val_idx], masks[val_idx]
        n,h,w, = val_masks.shape
        
        predicted = np.zeros(n*h*w)
        predicted_post = np.zeros(n*h*w)
        predicted_tta = np.zeros(n*h*w)
        predicted_tta_post = np.zeros(n*h*w)
        groundtruth = val_masks.reshape(n*h*w)
        
        for idx, image in enumerate(val_imgs):
            print("Generating mask {}...".format(idx))
            image = tools.pad_with_reflection(image, (CONTEXT_SIZE-PATCH_SIZE)//2)
            list_patches = np.array(tools.img_crop(image, CONTEXT_SIZE, CONTEXT_SIZE, PATCH_SIZE, PATCH_SIZE))

            predictions = []
            for _ in range(5):
                list_labels = model.predict(tta_gen.flow(list_patches, batch_size=len(list_patches), shuffle=False))
                predictions.append(list_labels)
                gc.collect()
            list_labels_tta = np.mean(predictions, axis=0)
            list_labels = model.predict(list_patches)

            mask = np.zeros((h, w))
            gray_mask = np.zeros((h, w))
            mask_tta = np.zeros((h, w))
            gray_mask_tta = np.zeros((h, w))

            patch_idx = 0
            for i in range(0,h,PATCH_SIZE):
                for j in range(0,w,PATCH_SIZE):
                    label = list_labels[patch_idx]
                    label_tta = list_labels_tta[patch_idx]
                    patch_idx += 1
                    gray_mask[i:i+PATCH_SIZE, j:j+PATCH_SIZE] = label[1]
                    gray_mask_tta[i:i+PATCH_SIZE, j:j+PATCH_SIZE] = label_tta[1]
                    if (label[1] > label[0]):
                        mask[i:i+PATCH_SIZE, j:j+PATCH_SIZE] = 1
                    if (label_tta[1] > label_tta[0]):
                        mask_tta[i:i+PATCH_SIZE, j:j+PATCH_SIZE] = 1
            
            mask_post =  post_processing(gray_mask)
            mask_tta_post =  post_processing(gray_mask_tta)
            
            predicted[idx*h*w:(idx+1)*h*w] = mask.reshape(h*w)
            predicted_post[idx*h*w:(idx+1)*h*w] = mask_post.reshape(h*w)
            predicted_tta[idx*h*w:(idx+1)*h*w] = mask_tta.reshape(h*w)
            predicted_tta_post[idx*h*w:(idx+1)*h*w] = mask_tta_post.reshape(h*w)
            groundtruth[idx*h*w:(idx+1)*h*w] = val_masks[idx].reshape(h*w)
            gc.collect()
            
            
        np.save(str(project_paths.MODELS_PATH / "patch" / "base_{}.npy".format(k)), predicted)
        np.save(str(project_paths.MODELS_PATH / "patch" / "base_post_{}.npy".format(k)), predicted_post)
        np.save(str(project_paths.MODELS_PATH / "patch" / "base_tta_{}.npy".format(k)), predicted)
        np.save(str(project_paths.MODELS_PATH / "patch" / "base_tta_post_{}.npy".format(k)), predicted_post)
        np.save(str(project_paths.MODELS_PATH / "patch" / "truth_{}.npy".format(k)), groundtruth)
        
def main():
    CrossVal()

if __name__ == "__main__":
    main()