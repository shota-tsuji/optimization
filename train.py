import numpy as np
from numpy import linalg as la
import pandas as pd
import math
from scipy.optimize import fmin_bfgs
import sys

def main():
    test_csv = sys.argv[1]
    df = pd.read_csv(test_csv, header=None)
    labels = df[0]
    pos_class = labels.unique()[1]
    mask = labels == pos_class
    data = df.iloc[:, 1:]
    # number of feature parameters
    n = 123
    # number of train data
    l = len(df)
    print(n)
    features = np.zeros((l,n))
    for (i, row) in data.iterrows():
        for j in row:
            features[i, j-1] = 1.0
    y_bin = np.ones(labels.shape, dtype=features.dtype)
    y_bin[~mask] = 0.0

    w = np.zeros(n)
    print(f(w, y_bin, features))
    print(phi(w, y_bin, features))
    print(la.norm(phi(w, y_bin, features)))

    (xopt, fopt, gopt, _, _, _, _) = fmin_bfgs(f, w, phi, args=(y_bin, features), maxiter=1000, full_output=True)
    print(fopt)


def f(w, y, X):
    sum = 0.0
    l = X.shape[0]
    for i in range(l):
        sum += (math.log(1.0 + math.exp(w @ X[i])) - y[i] * w @ X[i])

    return sum


def phi(w, y, X):
    l = X.shape[0]
    n = X.shape[1]
    g = np.zeros(n)

    for i in range(l):
        for j in range(n):
            p = 1.0 / (1.0 + math.exp(- w @ X[i]))
            g[j] += X[i, j] * (p - y[i])

    return g


if __name__ == "__main__":
    main()
