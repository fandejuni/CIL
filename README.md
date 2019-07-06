# 1. Setting up the data

The data has to be put in the *data* folder.

- The test images in *data/test_images*
- The training data in *data/training/images* and *data/training/groundtruth*

Then run *python save_images.py* to save the data in the relevant folders in *.npy* format.

# 2. Augment the data with morphological features

Once the data is set up (1):
Run *python morphological_feature_augmentation/feature_augmentation.py*

# 3. Get the Kaggle submission

The data has to have been augmented (1 and 2).

1. Run *python model_morpho_patch/train.py*
2. Run *python model_morpho_patch/classify.py*
3. Run *python simple_post_processing/post_processing.py*

The submission should then be located at *generated/submissions/morpho_patch_post/submission.csv*

# 4. Get the comparison results

The baselines are all implemented in the *pixelwise* folder.
The evaluation is also implemented in the *pixelwise* folder.

The data has to have been augmented (1 and 2).
If you want to get the comparison data from our novel approach, run *python python model_patch/crossval.py*.

Then simply run *python pixelwise/main.py*.
This will train the baselines on the subsets defined the 5-fold cross-validation, generate the predictions of the baselines,
and use the predictions from the baselines (and from our novel approach) to show a table of summarized results similar (and with similar values) to the one in the report.
