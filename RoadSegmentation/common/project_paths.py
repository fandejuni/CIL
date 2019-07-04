# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:32:16 2019

@author: Justin Dallant
"""

from pathlib import Path

PARENT_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = PARENT_PATH / "data"

TEST_DIR_PATH = DATA_PATH / "test_images"

TRAIN_DIR_PATH = DATA_PATH / "training"
TRAIN_IMAGE_PATH = TRAIN_DIR_PATH / "images"
TRAIN_GROUNDTRUTH_PATH = TRAIN_DIR_PATH / "groundtruth"

ADD_TRAIN_IMAGE_PATH = TRAIN_DIR_PATH / "add_images"
ADD_TRAIN_GROUNDTRUTH_PATH = TRAIN_DIR_PATH / "add_groundtruth"



GENERATED_PATH = PARENT_PATH / "generated"
MODELS_PATH = GENERATED_PATH / "models"
MASKS_TEST_PATH = GENERATED_PATH / "masks_test"
RESULTS_TEST_PATH = GENERATED_PATH / "results_test"
MASKS_TRAIN_PATH = GENERATED_PATH / "masks_train"
SUBMISSIONS_PATH = GENERATED_PATH / "submissions"
PREPROCESSED_PATH = GENERATED_PATH / "preprocessed"
