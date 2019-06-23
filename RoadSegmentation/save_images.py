# -*- coding: utf-8 -*-
"""
Created on Thu May 23 21:15:05 2019

@author: Justin Dallant
"""

import os
import re
import numpy as np
import matplotlib.image as mpimg

def save_images(dir_path, output_path):
    image_files = [filename for filename in os.listdir(dir_path) if filename.endswith(".png")] 
    image_files.sort()
    num_images = len(image_files)

    if num_images == 0:
        return

    shape = list(mpimg.imread(dir_path + "/" + image_files[0]).shape)

    images = np.zeros([num_images] + shape)
    ids = np.zeros(num_images, np.int)

    for i, filename in enumerate(image_files):
        images[i] = mpimg.imread(dir_path + "/" + filename)
        ids[i] = int(re.search(r"\d+", filename).group(0))

    with open(output_path, "wb") as np_out:
        np.save(np_out, images)


    base_path, extension = output_path.split('.')
    with open(base_path + "_ids." + extension, "wb") as np_out:
        np.save(np_out, ids)


if __name__ == '__main__':
    save_images("data/training/groundtruth/", "data/saved/train_groundtruth.npy")
    save_images("data/training/images/", "data/saved/train_images.npy")
    save_images("data/test_images/", "data/saved/test_images.npy")
