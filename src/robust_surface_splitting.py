from src.SimilarityTransformation import calculate_transform, transform_surface

__author__ = 'abhinavkashyap'
from src.line_fititng import LineFitting
from RobustSurfaceFitting import LinSurfFit
from RobustSurfaceFitting import QuadSurfFit
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
        self.number_iterations = 1

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
        P1 = np.where(values < 0)[0]
        P2 = np.where(values > 0)[0]
        return P1, P2

    def split(self):
        # Step 1: Split the points in the points cloud into P1 and P2
        # Step 2: Fit appropriate surfaces to P1 and P2
        # Step 3: Find the Distance D = | S1(x, y) - S2(x, y) |
        # Step 4: Find the subset of Points Q that have least distance
        # Step 5: Fit a line to this set of points. Repeat
        for i in xrange(self.number_iterations):
            self.p1_indices, self.p2_indices = self.__splitPointCloud()
            self.p1, self.p2 = self.pointcloud[self.p1_indices], self.pointcloud[self.p2_indices]
            P1 = np.array(self.p1)
            P2 = np.array(self.p2)
            self.param_linear_surface, self.DLinearSurface, self.indices_chosen_lin = LinSurfFit(P1[:, 0], P1[:, 1],
                                                                                                 P1[:, 2])
            self.param_curved_surface, self.DQuadraticSurface, self.indices_chosen_quad = QuadSurfFit(P2[:, 0],
                                                                                                      P2[:, 1],
                                                                                                      P2[:, 2])
            x = self.pointcloud[:, 0]
            y = self.pointcloud[:, 1]
            x2 = np.square(x)
            y2 = np.square(y)
            xy = np.multiply(x, y)
            ones = np.ones((x.shape[0], 1))
            S1xy = np.hstack((x, y, ones)) * self.param_linear_surface
            S2xy = np.hstack((x2, y2, xy, x, y, ones)) * self.param_curved_surface
            D = np.abs(np.subtract(S1xy, S2xy))
            D = np.array(D).ravel()
            minimumQPoints = np.argsort(D)[:50]
            QPoints = self.pointcloud[minimumQPoints]
            self.lineFitting = LineFitting(QPoints)
            self.line_params, error = self.lineFitting.get_lineparameters()
            print "error after fitting again: ", error
        return self.p1_indices, self.p2_indices, self.param_linear_surface, self.param_curved_surface, \
               self.DLinearSurface, self.DQuadraticSurface, self.line_params, self.indices_chosen_lin, \
               self.indices_chosen_quad, QPoints, minimumQPoints


def robust_splitting_main(front, left):
    FRONT_LEFT_PICKLE = "./pickled_files/fl_surface.pkl"
    LEFTLINE = "./left_line/left_line1.xyz"
    front_left_surface = pickleload(FRONT_LEFT_PICKLE)
    surfaceSplitting = RobustSurfaceSplitting(front_left_surface, LEFTLINE)
    p1_indices, p2_indices, param_line_surfaces, param_curved_surface, \
    DLinear, DCurved, \
    line_params, indices_lin, indices_quad, QPoints, indexes_of_QPoints = surfaceSplitting.split()
    # left_surface = front_left_surface[p1_indices]
    # front_surface = front_left_surface[p2_indices]


    left_surface = front_left_surface[p1_indices[indices_lin]]
    front_surface = front_left_surface[p2_indices[indices_quad]]

    s, R, T = calculate_transform(QPoints, param_line_surfaces, param_curved_surface)
    # the formula will be x_2 = s * x_1 * R + T
    # see comment in the method

    left_surface = get_new_surface(left_surface, DLinear, param_line_surfaces)
    transformed_surface = transform_surface(left_surface, s, R.T, T.T)
    left_surface[:, 0:3] = transformed_surface

    front_surface = get_new_surface(front_surface, DCurved, param_curved_surface)

    write_ply_file(left_surface, left)
    write_ply_file(front_surface, front)


if __name__ == "__main__":
    front_ply_path = "./front_left_surface/front_surface.ply"
    left_ply_path = "./front_left_surface/left_surface.ply"
    robust_splitting_main(front_ply_path, left_ply_path)
