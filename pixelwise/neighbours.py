# -*- coding: utf-8 -
"""
Created on Thu June 3 21:48:05 2019
"""

import numpy as np

def augmentImage(X):

    a = X.shape[0]
    b = X.shape[1]
    n = X.shape[2]

    XX = np.zeros((a, b, 9 * n))

    def getBounds(x, a):
        return (1 + x, a - 1 + x)

    l = [(x, y) for x in range(-1, 2) for y in range(-1, 2)]
    for i, (x, y) in enumerate(l):
        (a1, b1) = getBounds(x, a)
        (a2, b2) = getBounds(y, b)
        XX[1:-1, 1:-1, i*n:(i+1)*n] = X[a1:b1, a2:b2]

    def getXX(x, y):
        XXX = np.zeros((9 * n))
        for i, (dx, dy) in enumerate(l):
            xx = max(min(a - 1, x + dx), 0)
            yy = max(min(b - 1, y + dy), 0)
            XXX[i*n:(i+1)*n] = X[xx, yy]
        return XXX

    # Bounds
    for x in range(a):
        XX[x, 0] = getXX(x, 0)
        XX[x, b-1] = getXX(x, b-1)

    for y in range(b):
        XX[0, y] = getXX(0, y)
        XX[a-1, y] = getXX(a-1, y)

    return XX
