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
        self.pointcloud = point_cloud
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

            #Fit new surfaces to the line
            P1 = np.array(self.p1)
            P2 = np.array(self.p2)
            self.linearSurfaceParameters, self.DLinearSurface, self.indicesLinearSurface = LinSurfFit(P1[:, 0], P1[:, 1], P1[:, 2])
            chosenPointsLinear = self.p1[self.indicesLinearSurface]
            chosenPointsLinear = get_new_surface(chosenPointsLinear, self.DLinearSurface, self.linearSurfaceParameters)
            self.quadraticSurfaceParameters, self.DQuadratic, self.indicesQuadraticSurface = QuadSurfFit(P2[:, 0], P2[:, 1], P2[:, 2])
            chosenPointsQuadratic = self.p2[self.indicesQuadraticSurface]
            chosenPointsQuadratic = get_new_surface(chosenPointsQuadratic, self.DQuadratic, self.quadraticSurfaceParameters)
            write_ply_file(chosenPointsLinear, "../plyfiles/iteration"+str(i+1)+"LinearSurface.ply")
            write_ply_file(chosenPointsQuadratic, "../plyfiles/iteration"+str(i+1)+"QuadraticSurface.ply")
            write_ply_file(chosenPointsLinear, "../front_left_surface/left_surface.ply")
            write_ply_file(chosenPointsQuadratic, "../front_left_surface/front_surface.ply")

            self.pointcloud = np.vstack((chosenPointsLinear, chosenPointsQuadratic))
            write_ply_file(self.pointcloud, "../plyfiles/iteration"+str(i+1)+"newpointcloud.ply")





def robust_splitting_main(front, left):
    FRONT_LEFT_PICKLE = "../pickled_files/fl_surface.pkl"
    LEFTLINE = "../left_line/left_line1.xyz"
    front_left_surface = pickleload(FRONT_LEFT_PICKLE)
    surfaceSplitting = RobustSurfaceSplitting(front_left_surface, LEFTLINE)
    surfaceSplitting.split()



if __name__ == "__main__":
    front_ply_path = "../front_left_surface/front_surface.ply"
    left_ply_path = "../front_left_surface/left_surface.ply"
    robust_splitting_main(front_ply_path, left_ply_path)
