import math
import numpy as np

__author__ = 'Dmitrii'

"""
 assume that for input we have to ndarrays of points - X_1 and X_2
 the points are supposed to be arranged by columns (each column - one 3D datapoint)
"""


def CalculateTransformation(X_1, X_2):
    mean_X1 = np.mean(X_1, axis=1, keepdims=True)
    mean_X2 = np.mean(X_2, axis=1, keepdims=True)
    r_1 = X_1 - mean_X1
    r_2 = X_2 - mean_X2
    r_1_square = r_1 * r_1
    r_2_square = r_2 * r_2

    # sum of norms of r_1i is just the sum of all elements of r_1_square
    s = math.sqrt(np.sum(r_1_square) / math.sqrt(np.sum(r_2_square)))
    M = np.dot(r_2, r_1.T)
    Q = np.dot(M.T, M)
    w, V = np.linalg.eig(Q)
    egiens = np.diag(w)

    # 1 / sqrt(lambda_i)
    modified_eigens = np.linalg.inv(np.sqrt(egiens))

    # Q ^(-.05)
    Q_1 = np.dot(np.dot(V, modified_eigens), V.T)
    R = np.dot(M, Q_1)
    T = mean_X2 - s * np.dot(R, mean_X1)
    return s, R, T


### test to check that it doesn't produce any errors
X_1 = np.asarray([[4.0, 2.0, 3.0], [8.0, 3.1, 4.0]]).T
X_2 = np.asarray([[2.0, 3.0, 6.0], [3.0, 5.0, 2.0]]).T
CalculateTransformation(X_1, X_2)
