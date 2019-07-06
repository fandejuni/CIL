IM_HEIGHT = 608
IM_WIDTH = 608

import os
from os import sys
import numpy as np
import matplotlib.image as mpimg
from tensorflow.keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

if __name__ == '__main__' and __package__ is None:
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from common import generate, project_paths, tools, mask_to_submission

MODEL_NAME = "patch_155"
PATCH_SIZE = 8
CONTEXT_SIZE = 80

def main():
    print("b Classifying with {}...".format(MODEL_NAME))
    model_path = str(project_paths.MODELS_PATH / "patch" / (MODEL_NAME+".h5"))
    
    model = load_model(model_path, custom_objects={})

    image_path = project_paths.TEST_DIR_PATH
    filenames = [filename for filename in os.listdir(image_path) if filename.endswith(".png")]
    full_filenames = [str(image_path/filename) for filename in filenames]
    
    generate.create_folder(project_paths.MASKS_TEST_PATH / "patch")
    generate.create_folder(project_paths.RESULTS_TEST_PATH / "patch")
    generate.create_folder(project_paths.RESULTS_TEST_PATH / "patchgray")
    generate.create_folder(project_paths.SUBMISSIONS_PATH / "patch")

    datagen = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        rotation_range=20.,
        fill_mode='reflect',
        width_shift_range = 0.05, 
        height_shift_range = 0.05)
    
    for file_idx,filename in enumerate(full_filenames):
        base_name, _ = filenames[file_idx].split(".")
        print("Generating mask for {}...".format(base_name))
        image = mpimg.imread(filename)
        imgheight = image.shape[0]
        imgwidth = image.shape[1]
        image = tools.pad_with_reflection(image, (CONTEXT_SIZE-PATCH_SIZE)//2)
        list_patches = tools.img_crop(image, CONTEXT_SIZE, CONTEXT_SIZE, PATCH_SIZE, PATCH_SIZE)
        
        predictions = []
        for _ in range(7):
            list_labels = model.predict(datagen.flow(list_patches, batch_size=len(list_patches), shuffle=False))
            predictions.append(list_labels)
        list_labels = np.mean(predictions, axis=0)
    
        mask = np.zeros((imgheight, imgwidth))
        gray_mask = np.zeros((imgheight, imgwidth))
        
        idx = 0
        for i in range(0,imgheight,PATCH_SIZE):
            for j in range(0,imgwidth,PATCH_SIZE):
                label = list_labels[idx]
                idx += 1
                gray_mask[i:i+PATCH_SIZE, j:j+PATCH_SIZE] = label[1]
                if (label[1] > label[0]):
                    mask[i:i+PATCH_SIZE, j:j+PATCH_SIZE] = 1
        
        mask_img = tools.solution_to_img(mask)
        mpimg.imsave(project_paths.MASKS_TEST_PATH / "patch" / (base_name+".png"), mask_img)
        mpimg.imsave(project_paths.RESULTS_TEST_PATH / "patchgray" / (base_name+".png"), gray_mask, cmap="gray")
        
        img = mpimg.imread(str(project_paths.TEST_DIR_PATH / (base_name+".png")))
        overlay = tools.make_img_overlay(img, mask)
        mpimg.imsave(project_paths.RESULTS_TEST_PATH / "patch" / (base_name+".png"), overlay)
    
    
    mask_path = project_paths.MASKS_TEST_PATH / "patch"
    image_filenames =  [str(mask_path / filename) for filename in os.listdir(mask_path) if filename.endswith(".png")]
    mask_to_submission.masks_to_submission(project_paths.SUBMISSIONS_PATH / "patch" / "submission.csv", *image_filenames)

        
if __name__ == "__main__":
    main()

