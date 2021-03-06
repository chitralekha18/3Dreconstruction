import numpy as np
from plyfile import PlyData, PlyElement
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from RobustSurfaceFitting import QuadSurfFit
from RobustSurfaceFitting import LinSurfFit
from sklearn.neighbors import NearestNeighbors

class Resampling(object):
    def __init__(self,point_cloud,color,resolution = 150):
        '''
        TODO: Need to find a way to get projection_mat from PLY file
        '''
        self.point_cloud = point_cloud
        self.color = {tuple(point_cloud[:,i]): color[:,i] for i in range(point_cloud.shape[1])}
        self.resolution = resolution
        self.uniform_pointcloud = None #uniformpointCloud, will be written to ply file later
        self.color_uniform_pointcloud = {}
        self.numfacestowrite = 0;
        
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
        print plyfilename
        header  =  \
'''ply
format ascii 1.0
'''
        header_mid = 'element vertex '+str(int(self.uniform_pointcloud.shape[1]))
        header_beforeface = \
'''
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar alpha
'''
        self.numfacestowrite = 2*(self.resolution-1)*(self.resolution-1)
        header_face = 'element face '+str(int(self.numfacestowrite))
        header_end = \
'''
property list uchar int vertex_indices
end_header
'''
        # Now write Points
        points = self.uniform_pointcloud
        
        with open(plyfilename,'w') as f:
            f.write(header+header_mid+header_beforeface+header_face+header_end)
            for j in range(self.uniform_pointcloud.shape[1]):
                for i in range(3):
                    f.write(str(self.uniform_pointcloud[i,j])+' ')

                f.write(' '.join([ str(x) for x in self.color_uniform_pointcloud[tuple(self.uniform_pointcloud[:,j])] ])+' ')

                f.write('\n')

        #Now write faces
        with open(plyfilename,'a') as f:
            i = 0
            j = 0
            
            for row in range(self.resolution-1):
            
                for col in range(self.resolution-1):
                    i = col
                    f.write('3 '+str(self.resolution*row+i)+' '+str(self.resolution*row+i+1)+' '+str(self.resolution*(row+1)+i)+'\n')
                    f.write('3 '+str(self.resolution*row+i+1)+' '+str(self.resolution*(row+1)+i)+' '+str(self.resolution*(row+1)+i+1)+'\n')                    
                    j+=1
            
                               

    def do_resampling(self,fitplane = 0):
        ''' 
        1. Sample 2D points at regular intervals from the sampling region. Get the corresponding Z value from the curve params. i.e Get the corresponding 3D point (Done)
        2. Fill the empty space with default color (Done)
        3. Do linear interpolation for the point based on 4 nearest neighbour (although the paper instructs otherwise)
        '''
        if fitplane:
            param, points_without_outliers,whatever = LinSurfFit(self.point_cloud[0, :], self.point_cloud[1, :], self.point_cloud[2, :])
            x =  points_without_outliers[:,0]
            y = points_without_outliers[:,1]
        else:
            param, points_without_outliers,whatever = QuadSurfFit(self.point_cloud[0, :], self.point_cloud[1, :], self.point_cloud[2, :])
            x =  points_without_outliers[:,3]
            y = points_without_outliers[:,4]
       
        z =  points_without_outliers*param
		
        
        
        xpoints = np.linspace(float(min(x)),float(max(x)),self.resolution)
        ypoints = np.linspace(float(min(y)),float(max(y)),self.resolution)
        xv,yv = np.meshgrid(xpoints,ypoints)
        
        
        a = param
        if fitplane:
            zv =  float(a[0])*xv + float(a[1])*yv + float(a[2])
        else:
            zv = float(a[0])*xv**2 + float(a[1])*yv**2 + float(a[2])*xv*yv + float(a[3])*xv + float(a[4])*yv + float(a[5])
        
        #print (len(xv),len(yv),len(zv))
        
        flattenx = np.hstack(np.array(xv))
        flatteny = np.hstack(np.array(yv))
        flattenz = np.hstack(np.array(zv))
        print 'number of uniform points: ',np.vstack((flattenx,flatteny,flattenz)).shape[1],'\n'
        self.gridx = xv;
        self.gridy = yv;
        self.gridz = zv;
        self.uniform_pointcloud = np.vstack((flattenx,flatteny,flattenz))
     
        '''
        #using default color filling
        self.color_uniform_pointcloud = {tuple(self.uniform_pointcloud[:,i]): np.array([255,0,0,255]) for i in range(self.uniform_pointcloud.shape[1])} 
        print self.uniform_pointcloud.shape
        print self.point_cloud.shape
        '''
        
        #code for nearest neighbour color filling here
        num_neighbours = 4
        nbrs = NearestNeighbors(n_neighbors=num_neighbours, algorithm='ball_tree').fit(self.point_cloud.T)
        distances, indices = nbrs.kneighbors(self.uniform_pointcloud.T)
        #print indices.shape
        for idx in range(indices.shape[0]):
            #print 'index: ',idx
            avg_color = np.zeros(4,dtype = int)
            for i in indices[idx,:]:
                color = self.color[tuple(self.point_cloud.T[i,:])]
                #print color
                avg_color += color
                
            self.color_uniform_pointcloud[ tuple(self.uniform_pointcloud[:,idx]) ] = avg_color/num_neighbours
            
            
class PLYLoader(object):
    def __init__(self,filename):
        self.plydata = PlyData.read(filename)
        
    def get_points(self):
        x = (self.plydata['vertex']['x'])
        y = (self.plydata['vertex']['y'])
        z = (self.plydata['vertex']['z'])
        return np.vstack((x,y,z))
        
    def get_colors(self):
        red = (self.plydata['vertex']['red'])
        green = (self.plydata['vertex']['green'])
        blue = (self.plydata['vertex']['blue'])
        alpha = (self.plydata['vertex']['alpha'])
        return np.vstack((red,green,blue,alpha))
        
    def get_normals(self):
        normal_x = (self.plydata['vertex']['nx'])
        normal_y = (self.plydata['vertex']['ny'])
        normal_z = (self.plydata['vertex']['nz'])
        return np.vstack((normal_x,normal_y,normal_z))
        
def resampling_main(resolution,front, left, out_path):
        
        #PLY_PlaneFile = 'front_left_surface/left_surface.ply'
        #PLY_FILENAME = "front_left_surface/front_surface.ply"
        PLY_PlaneFile = left
        PLY_FILENAME = front 
        
        plyloader = PLYLoader(PLY_FILENAME)
        
        point_cloud = plyloader.get_points()  #Filling in with values, i follow column major order [ 3Dpoint1 3Dpoint2 3Dpoint3 ... ]
        
        #print point_cloud.shape # [x y z 1] vector
        color_matrix = plyloader.get_colors() # constructing the color matrix
        #print color_matrix
        #normal_matrix =  np.vstack((normal_x,normal_y,normal_z)) # constructing the normal matrix
        
        resamp_obj = Resampling(point_cloud,color_matrix,resolution)

        resamp_obj.do_resampling()
        resamp_obj.write_pointsPLY(out_path + 'front_surface_resampled.ply')
        #resamp_obj.plot_3D(resamp_obj.uniform_pointcloud)
        
        
        
        #Now Construct Planer Surface
        plyloader = PLYLoader(PLY_PlaneFile)
        point_cloud = plyloader.get_points()
        color_matrix = plyloader.get_colors()
        resamp_obj = Resampling(point_cloud,color_matrix)
        resamp_obj.do_resampling(fitplane=1)
        
        resamp_obj.write_pointsPLY(out_path + 'left_surface_resampled.ply')
        
if __name__ == "__main__":
    path_to_front_surface = './front_left_surface/front_surface.ply'
    path_to_left_surface = './front_left_surface/left_surface.ply'
    path_to_output = "./output/"
    resolution = 150
    resampling_main(resolution,path_to_front_surface, path_to_left_surface, path_to_output)
        