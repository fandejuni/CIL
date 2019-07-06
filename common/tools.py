"""
Baseline for CIL project on road segmentation.
This simple baseline consits of a CNN with two convolutional+pooling layers with a soft-max loss
"""

import gzip
import os
import sys
import urllib
import matplotlib.image as mpimg
from PIL import Image
from random import randint
import numpy as np
import common.project_paths
from skimage.util import random_noise

def solution_to_img(sol):
    (xx, yy) = sol.shape
    img = np.zeros([xx, yy, 3])
    img[:, :, 0] = sol[:, :]
    img[:, :, 1] = sol[:, :]
    img[:, :, 2] = sol[:, :]
    img *= 255
    # for x in range(xx):
        # for y in range(yy):
            # for c in range(3):
                # img[x, y, c] = sol[x, y] * 255
    return img.astype(np.uint8)

# Extract patches from a given image
def img_crop(im, w, h, stride_x=None, stride_y=None):
    if stride_x == None:
        stride_x = w
    if stride_y == None:
        stride_y = h
    list_patches = []
    imgheight = im.shape[0]
    imgwidth = im.shape[1]
    is_2d = im.ndim < 3
    for i in range(0,imgheight-h+stride_y,stride_y):
        for j in range(0,imgwidth-w+stride_x,stride_x):
            if is_2d:
                im_patch = im[i:i+w, j:j+h]
            else:
                im_patch = im[i:i+w, j:j+h, :]
            list_patches.append(im_patch)
    return list_patches


def extract_data(filename, num_images, IMG_PATCH_SIZE):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename / (imageid + ".png")
        if os.path.isfile(image_filename):
            print ('Loading', image_filename)
            img = mpimg.imread(str(image_filename))
            imgs.append(img)
        else:
            print ('File ' + str(image_filename) + ' does not exist')

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH/IMG_PATCH_SIZE)*(IMG_HEIGHT/IMG_PATCH_SIZE)

    img_patches = [img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

    return np.asarray(data)
        
# Assign a label to a patch v
def value_to_class(v, foreground_threshold = 0.25):
    df = np.mean(v)
    if df > foreground_threshold:
        return [0, 1]
    else:
        return [1, 0]

# Extract label images
def extract_labels(filename, num_images, IMG_PATCH_SIZE):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename / (imageid + ".png")
        if os.path.isfile(image_filename):
            print ('Loading', image_filename)
            img = mpimg.imread(str(image_filename))
            gt_imgs.append(img)
        else:
            print ('File ' + image_filename + ' does not exist')

    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = np.asarray([value_to_class(np.mean(data[i])) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(np.float32)


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
        100.0 *
        np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
        predictions.shape[0])

# Write predictions from neural network to a file
def write_predictions_to_file(predictions, labels, filename):
    max_labels = np.argmax(labels, 1)
    max_predictions = np.argmax(predictions, 1)
    file = open(filename, "w")
    n = predictions.shape[0]
    for i in range(0, n):
        file.write(max_labels(i) + ' ' + max_predictions(i))
    file.close()

# Print predictions from neural network
def print_predictions(predictions, labels):
    max_labels = np.argmax(labels, 1)
    max_predictions = np.argmax(predictions, 1)
    print (str(max_labels) + ' ' + str(max_predictions))

# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    array_labels = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if labels[idx][0] > 0.5:
                l = 1
            else:
                l = 0
            array_labels[j:j+w, i:i+h] = l
            idx = idx + 1
    return array_labels

def img_float_to_uint8(img, PIXEL_DEPTH=255):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
    return rimg

def concatenate_images(img, gt_img, PIXEL_DEPTH=255):

    print(img.shape, gt_img.shape)

    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img, PIXEL_DEPTH)
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img, PIXEL_DEPTH)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def make_img_overlay(img, predicted_img, PIXEL_DEPTH=255):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*PIXEL_DEPTH

    img8 = img_float_to_uint8(img, PIXEL_DEPTH)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img
	
def pad_with_reflection(img, padding):
    npad = [(padding,padding),(padding,padding)] + [(0,0)]*(img.ndim-2)
    return np.pad(img, npad, mode = 'reflect')
	
def random_transformation(img):
    r = randint(0,3)
    img = np.rot90(img,r)
    if randint(0,1):
        img = np.flipud(img)
    r = randint(-5,5)/100
    img[:,:,randint(0,img.shape[2]-1)] += r
    r = 0.95 + randint(0,10)/100
    img[:,:,randint(0,img.shape[2]-1)] *= r
    return random_noise(img)
