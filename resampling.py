import numpy as np
from plyfile import PlyData, PlyElement
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class Resampling(object):
    def __init__(self,point_cloud,color,projection_mat=None):
        '''
        TODO: Need to find a way to get projection_mat from PLY file
        '''
        self.point_cloud = point_cloud
        self.color = color
        self.projection_mat = projection_mat
    
    def __str__(self):
        pass
        
    def get_proj_xy(self):
        '''3D point clouds are projected onto X-Y plane
        returns: vector of [x y 1] 
        ''' 
        
        x = self.point_cloud[0,:]
        y = self.point_cloud[1,:]
        one = np.ones(len(x))
        self.projected_pts = np.vstack((x,y,one)) # following column major order 
        return self.projected_pts

    def plot_3D(self):
        '''
        For plotting the pointcloud.
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.point_cloud[0,:], self.point_cloud[1,:], self.point_cloud[2,:], c=self.color.T/255, marker='.')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()
        
    def plot_projection(self):
        '''
        For plotting the projected points
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.point_cloud[0,:], self.point_cloud[1,:], np.zeros(len(self.point_cloud[0,:])), c=self.color.T/255, marker='.') 
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()
        
    def do_resampling(self):
        ''' 
        TODO: 
        1. Sample 2D points at regular intervals from the sampling region. Get the corresponding Z value from the curve params. i.e Get the corresponding 3D point 
        2. If the point has
        3. Do bilinear interpolation for the point based on 4 nearest neighbour (the paper instructs otherwise)
        '''
        self.plot_projection()  # simple projection into X-Y plane (putting Z=0) won't work. We need to take Perspective Projection. 
        # self.plot_3D()

if __name__ == "__main__":
        PLY_FILENAME = "frontalpoints_small.ply"
        plydata = PlyData.read(PLY_FILENAME)
        x = (plydata['vertex']['x'])
        y = (plydata['vertex']['y'])
        z = (plydata['vertex']['z'])
        red = (plydata['vertex']['red'])
        green = (plydata['vertex']['green'])
        blue = (plydata['vertex']['blue'])
        alpha = (plydata['vertex']['alpha'])
        normal_x = (plydata['vertex']['nx'])
        normal_y = (plydata['vertex']['ny'])
        normal_z = (plydata['vertex']['nz'])
        
        point_cloud = np.matrix(np.empty([len(x), 4])) # [x,y,z,1] constructing the point cloud in homogenous representation
        point_cloud = np.vstack((x,y,z,np.ones(len(x)))) #Filling in with values, i follow column major order [ 3Dpoint1 3Dpoint2 3Dpoint3 ... ]
        
        print point_cloud[:,0].shape # [x y z 1] vector
        color_matrix = np.vstack((red,green,blue,alpha)) # constructing the color matrix
        #print color_matrix
        normal_matrix =  np.vstack((normal_x,normal_y,normal_z)) # constructing the normal matrix
        
        resamp_obj = Resampling(point_cloud,color_matrix)
        resamp_obj.get_proj_xy()
        resamp_obj.do_resampling()
        
    