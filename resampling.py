import numpy as np
from plyfile import PlyData, PlyElement

class Resampling():
    def __init__(self,point_cloud,color,projection_mat=None):
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
        
    def do_resampling(self):
        ''' 
        TODO: 1. Sample 3D points at regular intervals. 
        2. Do bilinear interpolation
        '''
    

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
        point_cloud = np.matrix(np.empty([len(x), 4])) # [x,y,z,1]
        point_cloud = np.vstack((x,y,z,np.ones(len(x)))) #Filling in with values, i follow column major order [ 3Dpoint1 3Dpoint2 3Dpoint3 ... ]
        
        print point_cloud[:,0].shape # [x y z 1] vector
        color_matrix = np.vstack((red,green,blue))
        #print color_matrix
        resamp_obj = Resampling(point_cloud,color_matrix)
        resamp_obj.get_proj_xy()
        resamp_obj.do_resampling()
        
    