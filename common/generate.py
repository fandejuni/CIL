from PIL import Image
from common import project_paths, tools, mask_to_submission
import os
import matplotlib.image as mpimg

def create_folder(folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)

# Get a concatenation of the prediction and groundtruth for given input file
def get_prediction_with_groundtruth(filename, image_idx, get_prediction):

    imageid = "satImage_%.3d" % image_idx
    image_filename = filename / (imageid + ".png")
    img = mpimg.imread(str(image_filename))

    img_prediction = get_prediction(img)
    cimg = tools.concatenate_images(img, img_prediction)

    return cimg

# Get prediction overlaid on the original image for given input file
def get_prediction_with_overlay(filename, image_idx, get_prediction):

    imageid = "satImage_%.3d" % image_idx
    image_filename = filename / (imageid + ".png")
    img = mpimg.imread(str(image_filename))

    img_prediction = get_prediction(img)
    oimg = tools.make_img_overlay(img, img_prediction)

    return oimg


def run_predictions_training_set(name, get_prediction, TRAINING_SIZE=100):

    data_dir = project_paths.DATA_PATH / "training"
    train_data_filename = data_dir / 'images'

    print ("Running prediction on training set")
    prediction_training_dir = project_paths.MASKS_TRAIN_PATH

    create_folder(prediction_training_dir)

    for i in range(1, TRAINING_SIZE+1):
        print("Predicting", i, TRAINING_SIZE)
        pimg = get_prediction_with_groundtruth(train_data_filename, i, get_prediction)
        Image.fromarray(pimg).save(str(prediction_training_dir / ("prediction_" + str(i) + ".png")))
        oimg = get_prediction_with_overlay(train_data_filename, i, get_prediction)
        oimg.save(str(prediction_training_dir / ("overlay_" + str(i) + ".png")))



def run_predictions_test_set(name, get_prediction, preprocessed=None):

    print ("Running prediction on test set")
    #mask_test_dir = "masks_test/"
    masks_test_dir = project_paths.MASKS_TEST_PATH / name
    #prediction_test_dir = "predictions_test/"
    results_test_dir = project_paths.RESULTS_TEST_PATH / name
    #test_dir = "../data/test_images/"
    test_dir = project_paths.TEST_DIR_PATH

    create_folder(masks_test_dir)
    create_folder(results_test_dir)

    if not (preprocessed is None):
        print(preprocessed.shape)

    l = [x for x in os.listdir(test_dir) if x[-4:] == ".png"]
    l.sort()
    for i in range(len(l)):

        f = l[i]
        print("Predicting " + f + "...", str(i+1) + "/" + str(len(l)))

        img = mpimg.imread(str(test_dir / f))

        if not (preprocessed is None):
            img = preprocessed[i]

        img_pred = get_prediction(img)
        img_prediction = tools.solution_to_img(img_pred)

        img = mpimg.imread(str(test_dir / l[i]))

        Image.fromarray(img_prediction).save(str(results_test_dir / f))

        pimg = tools.concatenate_images(img, img_pred)
        Image.fromarray(pimg).save(str(masks_test_dir / ("concat_" + f)))

        oimg = tools.make_img_overlay(img, img_pred)
        oimg.save(str(masks_test_dir / ("overlay_" + f)))

    submission_filename = str(project_paths.SUBMISSIONS_PATH / (name + ".csv"))

    print("Generating submission file...")
    mask_to_submission.generate(submission_filename, results_test_dir)
    print("Done!")
