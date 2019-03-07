import numpy as np
import matplotlib.pyplot as plt

n = 10000 # users
m = 1000 # movies

mean_per_movies = False

def initData():

    data = np.zeros([n, m])
    f = open("data_train.csv")

    for line in f.readlines()[1:]:
        a = line.split(",")
        b = a[0].split("_")
        row = int(b[0][1:])
        column = int(b[1][1:])
        value = int(a[1])
        data[row - 1, column - 1] = value

    print("Ready")
    missing = (data == 0)

    if mean_per_movies:
        for j in range(m):
            s = 0
            number = 0
            for i in range(n):
                if data[i, j] != 0:
                    number += 1
                    s += data[i, j]
            mu = s / number 
            print(mu)
            for i in range(n):
                if data[i, j] == 0:
                    data[i, j] = mu
    else:
        mu = data.sum() / (data != 0).sum()
        print(mu)
        data[data == 0] = mu

    return data, missing

def computeRep(data):

    print("Computing SVD...")
    U, s, Vt = np.linalg.svd(data, full_matrices=True)
    print("Done!")

    print(U.shape, s.shape, Vt.shape)

    S = np.zeros((10000, 1000))
    S[:1000, :1000] = np.diag(s)

    np.allclose(data, U.dot(S).dot(Vt)) # check

    return U, S, Vt

def predict(U, data, k=30):
    pred = U[:, :k].dot((U[:, :k].T).dot(data))
    return pred

def evaluate(pred, data, missing):
    Z = np.abs(pred - data)
    Z[missing] = 0
    return Z.mean()

def writeResults(data, pred, missing):
    ### Write results

    f = open("sampleSubmission.csv", "r")
    r = open("predictions/pred_" + str(k) + ".csv", "w")
    r.write("Id,Prediction\n")

    for line in f.readlines()[1:]:
        a = line.split(",")[0].split("_")
        i = int(a[0][1:]) - 1
        j = int(a[1][1:]) - 1
        value = data[i, j]
        if missing[i, j]:
            value = pred[i, j]
        s = "r"
        s += str(i + 1)
        s += "_c"
        s += str(j + 1)
        s += ","
        value = round(value)
        value = min(5, max(1, value))
        s += str(int(value))
        s += "\n"
        r.write(s)

    r.close()

data, missing = initData()
U, S, Vt = computeRep(data)

ks = [1, 2, 3, 5, 10, 20, 30, 50, 75, 100, 200, 500, 1000]
errors = []

for k in ks:
    print("Predicting k=" + str(k) + "...")
    pred = predict(U, data, k)
    error = evaluate(pred, data, missing)
    errors.append(error)
    print(error)
    writeResults(data, pred, missing)

plt.plot(ks, errors)
plt.show()
