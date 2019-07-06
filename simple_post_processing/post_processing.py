import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.morphology import disk, binary_erosion, binary_closing, binary_opening, remove_small_objects, remove_small_holes
from PIL import Image

#if __name__ == '__main__' and __package__ is None:
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from common import project_paths, tools, mask_to_submission, generate


def post_processing(input_mask):
    mask = input_mask.copy()
    mask = (mask>0.6)
    mask = binary_opening(mask, disk(5))
    mask = binary_closing(mask, disk(5))
    remove_small_objects(mask, 8*8*10, in_place=True)
    remove_small_holes(mask, 8*8*3, in_place=True)
    mask = binary_erosion(mask, disk(4))
    plt.figure(figsize=(8,8))
    plt.imshow(mask, cmap="gray")
    plt.show()
    
    return mask.astype(np.uint8)
    

def process_masks():
    
    mask_path = project_paths.RESULTS_TEST_PATH / "morpho_patchgray"
    mask_filenames = [filename for filename in os.listdir(mask_path) if filename.endswith(".png")]

    generate.create_folder(project_paths.MASKS_TEST_PATH / "morpho_patch_post")
    generate.create_folder(project_paths.RESULTS_TEST_PATH / "morpho_patch_post")
    
    for i in range(len(mask_filenames)):
        print("Processing {}...".format(mask_filenames[i]))
        mask = mpimg.imread(mask_filenames[i])[:,:,0]
        
        new_mask = post_processing(mask)
        
        Image.fromarray((new_mask*255).astype(np.uint8)).save(project_paths.MASKS_TEST_PATH / "morpho_patch_post" / mask_filenames[i])
        
        img = mpimg.imread(str(project_paths.TEST_DIR_PATH / mask_filenames[i]))
        overlay = tools.make_img_overlay(img, new_mask)
        mpimg.imsave(project_paths.RESULTS_TEST_PATH / "morpho_patch_post" / mask_filenames[i], overlay)
        plt.figure(figsize=(8,8))
        plt.imshow(overlay)
        plt.show()
     
    generate.create_folder(project_paths.SUBMISSIONS_PATH / "morpho_patch_post")
    submission_filename = project_paths.SUBMISSIONS_PATH / "morpho_patch_post" / "submission.csv"
    mask_to_submission.generate(submission_filename, project_paths.MASKS_TEST_PATH / "morpho_patch_post", 0.70)
     
if __name__ == '__main__':
    process_masks()

    
  
