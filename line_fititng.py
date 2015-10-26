__author__ = 'abhinavkashyap'
import numpy as np
# The xyz file contains the (x, y, z) of the line that is manually drawn between the center and the left images


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
        x = np.matrix(data[:, 0])
        y = np.matrix(data[:, 1])
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
    #Choosing the line1-4 because that gives the minimal error
    data = np.matrix(np.genfromtxt("./left-line1/line1-4.xyz"))
    lineFitting = LineFitting(data)
    print lineFitting.get_lineparameters()





