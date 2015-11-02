__author__ = 'abhinavkashyap'
import numpy as np
from line_fititng import LineFitting
from RobustSurfaceFitting import LinSurfFit
from RobustSurfaceFitting import QuadSurfFit
from plyfile import PlyData
from utils import *

class RobustSurfaceSplitting():
    def __init__(self, point_cloud, initial_line_points_file):
        """
            point_cloud: The set of points that needs to be split
            Point cloud contains the points that needs to be split
            Initially the front curved surface and the planar wall to its left are considered(For testing)
        """
        self.pointcloud = np.matrix(point_cloud)
        self.lineFitting = LineFitting(np.matrix(np.genfromtxt(initial_line_points_file)))
        self.line_params, initial_error = self.lineFitting.get_lineparameters()  # [theta0, theta1] shape 2,1
        print "initial error", initial_error
        self.Q = 50  # Q is the number of points close to the splitting line
        self.number_iterations = 2
        self.p1, self.p2 = self.__splitPointCloud()
        P1 = np.array(self.p1)
        P2 = np.array(self.p2)
        self.param_linear_surface, self.DLinearSurface = LinSurfFit(P1[:, 0], P1[:, 1], P1[:, 2])
        self.param_curved_surface, self.DQuadraticSurface = QuadSurfFit(P2[:, 0], P2[:, 1], P2[:, 2])


    def __splitPointCloud(self):
        """
            the point cloud is split into two according to the sign of l(X)
        """
        x = self.pointcloud[:, 0]
        y = self.pointcloud[:, 1]
        ones = np.matrix(np.ones(x.shape[0])).T
        values = (np.hstack((x, ones)) * self.line_params) - y
        # Return two sets of points based on the sign
        values = np.array(values).ravel()
        P1 = np.where(values < 0)
        P2 = np.where(values > 0)
        return self.pointcloud[P1], self.pointcloud[P2]

    def split(self):
        # Step 1: Split the points in the points cloud into P1 and P2
        # Step 2: Fit appropriate surfaces to P1 and P2
        # Step 3: Find the Distance D = | S1(x, y) - S2(x, y) |
        # Step 4: Find the subset of Points Q that have least distance
        # Step 5: Fit a line to this set of points. Repeat
        for i in xrange(self.number_iterations-1):

            x = self.pointcloud[:, 0]
            y = self.pointcloud[:, 1]
            x2 = np.square(x)
            y2 = np.square(y)
            xy = np.multiply(x, y)
            ones = np.ones((x.shape[0], 1))
            S1xy = np.hstack((x,y, ones)) * self.param_linear_surface
            S2xy = np.hstack((x2, y2, xy, x, y, ones)) * self.param_curved_surface
            D = np.abs(np.subtract(S1xy, S2xy))
            D = np.array(D).ravel()
            minimumQPoints = np.argsort(D)[:50]
            QPoints = self.pointcloud[minimumQPoints]
            self.lineFitting = LineFitting(QPoints)
            self.line_params, error = self.lineFitting.get_lineparameters()
            print "error: ", error
            self.p1, self.P2 = self.__splitPointCloud()
            P1 = np.array(self.p1)
            P2 = np.array(self.p2)
            self.param_linear_surface, self.DLinearSurface = LinSurfFit(P1[:, 0], P1[:, 1], P1[:, 2])
            self.param_curved_surface, self.DQuadraticSurface = QuadSurfFit(P2[:, 0], P2[:, 1], P2[:, 2])

        return self.p1, self.p2, self.param_linear_surface, self.param_curved_surface, self.line_params


if __name__ == "__main__":
    left_linear_surface_ply = PlyData.read('./left-linear-surface/left-linear-surface.ply')
    front_curved_surface_ply = PlyData.read('./front-curved-surface/front-curved-surface.ply')
    left_linear_surface_points = np.hstack((left_linear_surface_ply['vertex']['x'].reshape(-1,1), left_linear_surface_ply['vertex']['y'].reshape(-1, 1), left_linear_surface_ply['vertex']['z'].reshape(-1, 1)))
    front_curved_surface_points = np.hstack((front_curved_surface_ply['vertex']['x'].reshape(-1,1), front_curved_surface_ply['vertex']['y'].reshape(-1,1), front_curved_surface_ply['vertex']['z'].reshape(-1,1)))
    cloud_points = np.vstack((left_linear_surface_points, front_curved_surface_points))
    robust_surface_splitting = RobustSurfaceSplitting(cloud_points, "./left-line1/line1-4.xyz")
    P1, P2, parameters_lin_surface, parameters_curved_surface, line_params = robust_surface_splitting.split()
    p1_pickle = open('./surface_split_parameter_files/p1_pickle.txt', 'w')
    p2_pickle = open('./surface_split_parameter_files/p2_pickle.txt', 'w')
    param_lin_surface_pickle = open('./surface_split_parameter_files/lin_surface_pickle.txt', 'w')
    parameters_curved_surface_pickle = open('./surface_split_parameter_files/curved_surface_pickle.txt', 'w')
    final_split_line_pickle = open('./surface_split_parameter_files/final_line_pickle.txt', 'w')
    pickledump(P1, p1_pickle)
    pickledump(P2, p2_pickle)
    pickledump(parameters_lin_surface, param_lin_surface_pickle)
    pickledump(parameters_curved_surface,parameters_curved_surface_pickle)
    pickledump(line_params, final_split_line_pickle)


