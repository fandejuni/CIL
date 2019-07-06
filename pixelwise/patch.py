import numpy as np

size = 8

def simplify(X):
    Y = np.reshape(X, [-1, X.shape[-1]])
    return np.concatenate([np.mean(Y, axis=0), np.std(Y, axis=0)])

def convertImage(X):
    XX = np.zeros([X.shape[0] // size, X.shape[1] // size, 2 * X.shape[-1]])
    for x in range(X.shape[0] // size):
        for y in range(X.shape[1] // size):
            Y = X[x*size:(x+1)*size, y*size:(y+1)*size]
            XX[x, y] = np.reshape(simplify(Y), [-1])
    return XX

def convertImages(X):
    XX = np.zeros([X.shape[0], X.shape[1] // size, X.shape[2] // size, 2 * X.shape[-1]])
    for i in range(X.shape[0]):
        print("Patching...", i + 1, X.shape[0])
        XX[i] = convertImage(X[i])
    return XX

def convertGroundtruth(Y):
    YY = np.zeros([Y.shape[0], Y.shape[1] // size, Y.shape[2] // size])
    for i in range(Y.shape[0]):
        for x in range(Y.shape[1] // size):
            for y in range(Y.shape[2] // size):
                YY[i, x, y] = np.mean(Y[i, x*size:(x+1)*size, y*size:(y+1)*size])
    return YY
