#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.image as mpimg
import re

n_zeros = 0.0
n_ones = 0.0

# assign a label to a patch
def patch_to_label(patch, threshold = 0.7):
    global n_zeros, n_ones

    df = np.mean(patch)
    if df > threshold:
        n_ones += 1.
        return 1
    else:
        n_zeros += 1.
        return 0


def mask_to_submission_strings(image_filename, threshold = 0.7):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch, threshold)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames, threshold = 0.7):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn, threshold))

def generate(submission_filename="../submissions/dummy_submission.csv", dir_path="../cnn/predictions_test/",  threshold = 0.7):
    global n_zeros, n_ones
    n_zeros = 0.0
    n_ones = 0.0
    image_filenames =  [str(dir_path / filename) for filename in os.listdir(dir_path) if filename.endswith(".png")]
    """
    for i in range(1, 51):
        image_filename = 'training/groundtruth/satImage_' + '%.3d' % i + '.png'
        print image_filename
        image_filenames.append(image_filename)
    """
    masks_to_submission(submission_filename, *image_filenames, threshold=threshold)
    print(n_ones / (n_zeros + n_ones))
