import numpy as np
from plyfile import PlyData, PlyElement
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from RobustSurfaceFitting import QuadSurfFit

class Resampling(object):
    def __init__(self,point_cloud,color,projection_mat=None):
        '''
        TODO: Need to find a way to get projection_mat from PLY file
        '''
        self.point_cloud = point_cloud
        self.color = {tuple(point_cloud[:,i]): color[:,i] for i in range(point_cloud.shape[1])}
        self.projection_mat = projection_mat
        self.uniform_pointcloud = None #uniformpointCloud, will be written to ply file later
        self.color_uniform_pointcloud = {}
        
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

    def plot_3D(self,points=None):
        '''
        For plotting the pointcloud.
        '''
        if points == None:
            points = self.point_cloud
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(points[0,:], points[1,:], points[2,:], marker='.')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()
        
    def write_pointsPLY(self,plyfilename):
        '''Write the uniformly sampled point from the curved surface in a ply file'''
        
        header  =  \
'''ply
format binary_little_endian 1.0
comment VCGLIB generated
'''
        header_mid = 'element vertex '+str(int(self.uniform_pointcloud.shape[1]))
        headerend = \
'''
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar alpha
element face 0
property list uchar int vertex_indices
end_header
'''
        points = self.uniform_pointcloud
        
        with open(plyfilename,'w') as f:
            f.write(header+header_mid+headerend)
            for j in range(self.uniform_pointcloud.shape[1]):
                for i in range(3):
                    f.write(str(self.uniform_pointcloud[i,j])+' ')

                f.write(' '.join([ str(x) for x in self.color_uniform_pointcloud[tuple(self.uniform_pointcloud[:,j])] ])+' ')

                f.write('\n')

    '''
    Don't need this function now
    def plot_projection(self):
        #For plotting the projected points
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.point_cloud[0,:], self.point_cloud[1,:], np.zeros(len(self.point_cloud[0,:])), c=self.color.T/255, marker='.') 
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()
        '''         
          
    def do_resampling(self):
        ''' 
        TODO: 
        1. Sample 2D points at regular intervals from the sampling region. Get the corresponding Z value from the curve params. i.e Get the corresponding 3D point (Done)
        2. Fill the empty space with default color (Done)
        3. Do linear interpolation for the point based on 4 nearest neighbour (although the paper instructs otherwise)
        '''
       
        param_curved_surface, points_without_outliers = QuadSurfFit(self.point_cloud[0, :], self.point_cloud[1, :], self.point_cloud[2, :])
        
        x =  points_without_outliers[:,3]
        y = points_without_outliers[:,4]
       
        z =  points_without_outliers*param_curved_surface
		
        resolution = 80
        
        xpoints = np.linspace(float(min(x)),float(max(x)),resolution)
        ypoints = np.linspace(float(min(y)),float(max(y)),resolution)
        xv,yv = np.meshgrid(xpoints,ypoints)
        a = param_curved_surface
        zv = float(a[0])*xv**2 + float(a[1])*yv**2 + float(a[2])*xv*yv + float(a[3])*xv + float(a[4])*yv + float(a[5])
        flattenx = np.hstack(np.array(xv))
        flatteny = np.hstack(np.array(yv))
        flattenz = np.hstack(np.array(zv))
        self.uniform_pointcloud = np.vstack((flattenx,flatteny,flattenz))
        
        #using default color filling
        self.color_uniform_pointcloud = {tuple(self.uniform_pointcloud[:,i]): np.array([255,255,0]) for i in range(self.uniform_pointcloud.shape[1])} 
        #print self.uniform_pointcloud.shape
        #print self.point_cloud.shape
        
        # TO DO: code for nearest neighbour color filling here
        
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
        
        point_cloud = np.matrix(np.empty([len(x), 3])) # [x,y,z,1] constructing the point cloud in homogenous representation
        point_cloud = np.vstack((x,y,z)) #Filling in with values, i follow column major order [ 3Dpoint1 3Dpoint2 3Dpoint3 ... ]
        
        print point_cloud[:,0].shape # [x y z 1] vector
        color_matrix = np.vstack((red,green,blue,alpha)) # constructing the color matrix
        #print color_matrix
        #normal_matrix =  np.vstack((normal_x,normal_y,normal_z)) # constructing the normal matrix
        
        resamp_obj = Resampling(point_cloud,color_matrix)

        resamp_obj.do_resampling()
        resamp_obj.write_pointsPLY('output.ply')
        resamp_obj.plot_3D(resamp_obj.uniform_pointcloud)
    
