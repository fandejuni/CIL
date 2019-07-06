# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 22:11:09 2019
"""

import os
import numpy as np
import matplotlib.image as mpimg
from scipy import ndimage
from scipy.ndimage.morphology import distance_transform_edt, binary_fill_holes
from skimage.morphology import disk, binary_closing
from skimage.transform import resize
from skimage.segmentation import felzenszwalb
from skimage.morphology import skeletonize
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from common import project_paths
from morphological_feature_augmentation import skelgraph

NUM_NEW_FEATURES = 4

def getSegmentFeatures(segment):
    side = segment.shape[0]
    
    segment[0,0] = 0
    
    smooth_seg = binary_fill_holes(segment)
    smooth_seg = binary_closing(smooth_seg, disk(side//60))
    dist_map = distance_transform_edt(smooth_seg)
    
    skel = skeletonize(smooth_seg)
    sG = skelgraph.SkelGraph(skel)
    sG.trim(side//12)
    skel = sG.mask
    num_endpoints = sG.get_num_endpoints()
    
    length = skel.sum()
    width_map = skel*dist_map
    mean_width = width_map.sum()/length
    std_width = (((width_map)**2).sum()/length - mean_width**2)**0.5
    
    elongation = length*length/segment.sum()
    
    features = [mean_width/side, std_width/side, elongation/side, num_endpoints]
    
    return features
    

def featureAugmentation(input_image):
    
    image = input_image.copy()
    
    image[:,:,0] = ndimage.median_filter(image[:,:,0], 5)
    image[:,:,1] = ndimage.median_filter(image[:,:,1], 5)
    image[:,:,2] = ndimage.median_filter(image[:,:,2], 5)
    
    image = resize(image, (304,304,3), anti_aliasing=True, preserve_range=True)
     
    image = rgb_to_hsv(image)
    
    side = image.shape[0]
    
    #Oversegmentation
    segments = felzenszwalb(image, scale=int(side/2.5), sigma=0.5, min_size=side*side//1000)
    num_segments = segments.max() + 1
    
    h,w,c = image.shape
    
    features = np.zeros((h,w,NUM_NEW_FEATURES))
    
    for segment_id in range(num_segments):
        segment = (segments == segment_id)
        features[segment] = getSegmentFeatures(segment)
    
    for i in range(NUM_NEW_FEATURES):
        features[:,:,i] = (features[:,:,i] - features[:,:,i].min())/(features[:,:,i].max() - features[:,:,i].min())
      
    h,w,c = input_image.shape
    features = resize(features, (h,w,NUM_NEW_FEATURES), anti_aliasing=False, preserve_range=True)
    
    augmented = np.zeros((h,w,c+NUM_NEW_FEATURES))
    augmented[:,:,:c] = rgb_to_hsv(input_image)
    augmented[:,:,c:] = features
    
    
    
    return augmented


def augment_images(dir_path, output_path):
    image_files = [filename for filename in os.listdir(dir_path) if filename.endswith(".png")] 
    num_images = len(image_files)

    if num_images == 0:
        return

    shape = list(mpimg.imread(str(dir_path/image_files[0])).shape)+[1]
    shape = shape[:3]
    new_shape = shape[:]
    new_shape[2] += NUM_NEW_FEATURES

    for i, filename in enumerate(image_files):
        print("Augmenting {}...".format(filename))
        image = mpimg.imread(str(dir_path/filename))
        image = featureAugmentation(image)

        base_name, extension = filename.split(".")
        with open(output_path / (base_name + ".npy"), "wb") as np_out:
            np.save(np_out, image)

if __name__ == '__main__':
    dir_path = project_paths.TRAIN_IMAGE_PATH
    output_path = project_paths.PREPROCESSED_PATH / "morphological" / "train"
    augment_images(dir_path, output_path)
    
    dir_path = project_paths.TEST_DIR_PATH
    output_path = project_paths.PREPROCESSED_PATH / "morphological" / "test"
    augment_images(dir_path, output_path)


