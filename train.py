import numpy as np
from numpy import linalg as la
import pandas as pd
import math
from scipy.optimize import fmin_bfgs

def main():
    df = pd.read_csv('a1a.csv', header=None)
    labels = df[0]
    #print(labels)
    data = df.iloc[:, 1:]
    #print(data)
    # number of feature parameters
    n = 123
    # number of train data
    l = len(df)
    print(n)
    features = np.zeros((l,n))
    for (i, row) in data.iterrows():
        for j in row:
            features[i, j-1] = 1.0

    w = np.zeros(n)
    print(f(w, labels, features))
    print(phi(w, labels, features))
    print(la.norm(phi(w, labels, features)))

    x0 = np.array([-3, -4])
    cost_weight = np.diag([1., 10.])
    #fmin_bfgs(quadratic_cost, x0, args=(cost_weight,))

def f(w, y, features):
    sum = 0.0
    i = 0
    for x in features:
        sum += math.log(1.0 + math.exp(- y[i] * w @ x))
        i += 1

    return sum

def phi(w, y, features):
    l = features.shape[0]
    n = features.shape[1]
    g = np.zeros(n)
    i = 0

    for x in features:
        for j in range(n):
            g[j] += - y[i] * features[i, j] / (1.0 + math.exp(y[i] * w @ x))
        i += 1

    return g

if __name__ == "__main__":
    main()
