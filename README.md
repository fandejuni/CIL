# Get started

## Get the data

Download data from https://inclass.kaggle.com/c/cil-road-segmentation-2019/data

Put the "training" and "test_images" folders in "RoadSegmentation/data"

# Organisation

generated:

* masks_test
* masks_train
* models
* results_test
* submissions: csv files to submit

# Conseils

## Imports

Mettre

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

en début de fichier Python avant d'importer common.whatever

## Utiliser les fonctions pour générer des résultats

* generate.run_predictions_training_set("cnn", get_prediction, TRAINING_SIZE)
* generate.run_predictions_test_set("cnn", get_prediction)
