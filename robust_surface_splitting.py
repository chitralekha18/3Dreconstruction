__author__ = 'abhinavkashyap'
import numpy as np
from line_fititng import LineFitting
from RobustSurfaceFitting import LinSurfFit
from RobustSurfaceFitting import QuadSurfFit
from plyfile import PlyData

class RobustSurfaceSplitting():
    def __init__(self, point_cloud, initial_line_points_file):
        """
            point_cloud: The set of points that needs to be split
            Point cloud contains the points that needs to be split
            Initially the front curved surface and the planar wall to its left are considered(For testing)
        """
        self.pointcloud = np.matrix(point_cloud)
        self.lineFitting = LineFitting(np.matrix(np.genfromtxt(initial_line_points_file)))
        self.line_params = self.lineFitting.get_lineparameters()[0]  # [theta0, theta1] shape 2,1
        self.Q = 50  # Q is the number of points close to the splitting line
        self.number_iterations = 2
        # self.fig = plt.figure()
        # self.axis = self.fig.add_subplot(111, projection="3d")

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
        count = self.number_iterations
        P1, P2 = self.__splitPointCloud()  # P1 P2 contain the points that are split according to the sign
        P1_array = np.array(P1)
        P2_array = np.array(P2)
        parameters_lin_surface, DLinear = LinSurfFit(P1_array[:, 0], P1_array[:, 1], P1_array[:, 2])
        parameters_curved_surface, DQuadratic = QuadSurfFit(P2_array[:, 0], P2_array[:, 1], P2_array[:, 2])
        P = np.vstack((P1, P2))
        D = []
        for eachRow in P:
            x = eachRow.item(0, 0)
            y = eachRow.item(0, 1)
            S1xy = np.matrix([[x, y, 1]]) * parameters_lin_surface
            S2xy = np.matrix([[x ** 2, y ** 2, x * y, x, y, 1]]) * parameters_curved_surface
            D.append(np.abs(S1xy - S2xy).item(0, 0))
        D = np.array(D)
        minimumQPoints = np.argsort(D)[:50]
        QPoints = P[minimumQPoints]
        self.lineFitting = LineFitting(QPoints)
        self.line_params, error = self.lineFitting.get_lineparameters()
        print 'count: ', count
        count = count - 1
        self.number_iterations = count
        if count != 0:
            self.split()

        return P1, P2, parameters_lin_surface, parameters_curved_surface, self.line_params


if __name__ == "__main__":
    left_linear_surface_ply = PlyData.read('./left-linear-surface/left-linear-surface.ply')
    front_curved_surface_ply = PlyData.read('./front-curved-surface/front-curved-surface.ply')
    left_linear_surface_points = np.hstack((left_linear_surface_ply['vertex']['x'].reshape(-1,1), left_linear_surface_ply['vertex']['y'].reshape(-1, 1), left_linear_surface_ply['vertex']['z'].reshape(-1, 1)))
    front_curved_surface_points = np.hstack((front_curved_surface_ply['vertex']['x'].reshape(-1,1), front_curved_surface_ply['vertex']['y'].reshape(-1,1), front_curved_surface_ply['vertex']['z'].reshape(-1,1)))
    cloud_points = np.vstack((left_linear_surface_points, front_curved_surface_points))
    robust_surface_splitting = RobustSurfaceSplitting(cloud_points, "./left-line1/line1-4.xyz")
    P1, P2, parameters_lin_surface, parameters_curved_surface, line_params = robust_surface_splitting.split()
    print P1
