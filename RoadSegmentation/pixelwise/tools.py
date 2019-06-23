import os
import numpy as np

def create_folder(folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)

def getImages(in_path, out_path=None, extension=".npy"):

    l = [filename for filename in os.listdir(in_path) if filename.endswith(extension)]
    l.sort()

    if out_path is not None:
        l = [f for f in l if not os.path.isfile(out_path + "/" + f[:-4] + ".npy")]

    return l

def load(name):
    return np.load(open(name, "rb"))

def openFolder(path):
    l = getImages(path)
    X = np.zeros([len(l)] + list(load(path + "/" + l[0]).shape))
    for i, f in enumerate(l):
        X[i] = np.load(open(path + "/" + f, "rb"))
    return X
