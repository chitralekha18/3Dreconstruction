__author__ = 'abhinavkashyap'
import numpy as np
from plyfile import PlyData
import matplotlib.pyplot as plt


class LineFitting():
    def __init__(self, points):
        """

        :param points: x,y,z points where the z coordinates are set to 0 and the line is fit for x and y values
        :return:
        """
        self.data = points

    def getDesignMatrix(self):
        """
        :return: D, v Da = v
        """
        data=  self.data
        # Considering only the x and the y values
        x = np.matrix(data[:, 0]).reshape(-1, 1)
        y = np.matrix(data[:, 1]).reshape(-1,1)
        ones = np.matrix(np.ones((x.shape[0],1)))
        D = np.hstack((x, ones))
        return D, y

    def solveLinearFitting(self, D, v):
        x, residuals, rank, singularvalues = np.linalg.lstsq(D, v)
        print residuals
        return x, residuals

    def get_lineparameters(self):
        D, y = self.getDesignMatrix()
        return self.solveLinearFitting(D, y)

if __name__ == "__main__":

    data = np.genfromtxt("./left_line/left_line1.xyz")
    x = np.matrix(data[:, 0]).reshape(-1, 1)
    y = np.matrix(data[:, 1]).reshape(-1, 1)
    z = np.matrix(data[:, 2]).reshape(-1, 1)
    lineFitting = LineFitting(data)
    params, error = lineFitting.get_lineparameters()
    ones = np.matrix(np.ones((x.shape[0],1)))
    new_y = np.hstack((x, ones)) * params
    plt.scatter(x, y)
    plt.plot(x, new_y)
    plt.show()
    np.savetxt("./left_line/left_line_new.xyz", np.hstack((x, new_y.reshape(-1,1), z)))





