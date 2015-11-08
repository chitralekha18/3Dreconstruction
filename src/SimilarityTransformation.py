import math

import numpy as np

__author__ = 'Dmitrii'


def calculate_transform(QPoints, linear_params, quadratic_params):
    # x and y
    inputs = np.asarray(QPoints[:, 0:2])
    n = inputs.shape[0]
    ones = np.ones((n, 1))

    squares = inputs * inputs
    xy = inputs[:, 0:1] * inputs[:, 1:2]

    D_linear = np.empty((n, 3))
    D_linear[:, 0:2] = inputs
    D_linear[:, 2:3] = ones

    D_quadratic = np.empty((n, 6))
    D_quadratic[:, 0:2] = squares
    D_quadratic[:, 2:3] = xy
    D_quadratic[:, 3:6] = D_linear

    linear_surface_z = D_linear * linear_params
    quadratic_surface_z = D_quadratic * quadratic_params
    X_1 = np.empty((n, 3))
    X_1[:, 0:2] = inputs
    X_1[:, 2:3] = linear_surface_z

    X_2 = np.empty((n, 3))
    X_2[:, 0:2] = inputs
    X_2[:, 2:3] = quadratic_surface_z
    s, R, T = __CalculateTransformation(X_1.T, X_2.T)

    # print X_1 - (s  * np.dot(X_1, R.T) + T.T )
    # if each point is presented by a row, the formula will be x_2 = s * x_1 * R.T + T.T
    return s, R, T


def transform_surface(surface, s, R, T):
    number_of_points = surface.shape[0]
    # 3D coordinates component
    points = surface[:, 0:3]
    result = s * np.dot(points, R) + T
    return result


"""
 assume that for input we have to ndarrays of points - X_1 and X_2
 the points are supposed to be arranged by columns (each column - one 3D datapoint)
"""


def __CalculateTransformation(X_1, X_2):
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
__CalculateTransformation(X_1, X_2)
