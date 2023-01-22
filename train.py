import numpy as np
from numpy import linalg as la
import pandas as pd
import math
from scipy.optimize import fmin_bfgs

def main():
    df = pd.read_csv('a1a.csv', header=None)
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
    print(y_bin)

    w = np.zeros(n)
    print(f(w, y_bin, features))
    print(la.norm(phi(w, y_bin, features)))

    (xopt, fopt, gopt, _, _, _, _) = fmin_bfgs(f, w, phi, args=(y_bin, features), maxiter=100, full_output=True)
    print(fopt)
    print(gopt)

def f(w, y, features):
    sum = 0.0
    i = 0
    for x in features:
        sum += (math.log(1.0 + math.exp(w @ x)) - y[i] * w @ x)
        i += 1

    return sum

def phi(w, y, features):
    l = features.shape[0]
    n = features.shape[1]
    g = np.zeros(n)
    i = 0

    for x in features:
        for j in range(n):
            #g[j] += - y[i] * features[i, j] / (1.0 + math.exp(y[i] * w @ x))
            #p = math.exp(w @ x) / 1 + math.exp(w @ x)
            p = 1.0 / (1.0 + math.exp(- w @ x))
            g[j] += features[i, j] * (p - y[i])
        i += 1

    return g

if __name__ == "__main__":
    main()
