import numpy as np
import pdb
from line_fititng import LineFitting
from RobustSurfaceFitting import LinSurfFit
from RobustSurfaceFitting import QuadSurfFit
from plyfile import PlyData, PlyElement
import pickle

class PLYLoader(object):
    def __init__(self,filename):
        self.plydata = PlyData.read(filename)
        
    def get_points(self):
        x = (self.plydata['vertex']['x'])
        y = (self.plydata['vertex']['y'])
        z = (self.plydata['vertex']['z'])
        return np.vstack((x.T,y.T,z.T))
        
    def get_colors(self):
        red = (self.plydata['vertex']['diffuse_red'])
        green = (self.plydata['vertex']['diffuse_green'])
        blue = (self.plydata['vertex']['diffuse_blue'])
        return np.vstack((red,green,blue))
        
    def get_normals(self):
        normal_x = (self.plydata['vertex']['nx'])
        normal_y = (self.plydata['vertex']['ny'])
        normal_z = (self.plydata['vertex']['nz'])
        return np.vstack((normal_x,normal_y,normal_z))


def CreatePLY(P1,parameters,filename,surf_type,originalPLYfile):
        x1 = P1[:,0]
        y1 = P1[:,1]
        z1 = P1[:,2]

        if surf_type == 'lin':
            z1new = np.hstack((x1,y1,np.matrix(np.ones(np.size(P1,0))).T))*parameters
        else:
            z1new = np.hstack((np.multiply(x1,x1),np.multiply(y1,y1), np.multiply(x1,y1), x1, y1,np.matrix(np.ones(np.size(P1,0))).T))*parameters

        originalPLY = PLYLoader(originalPLYfile)
        point_cloud = (originalPLY.get_points()).T
        color_matrix = (originalPLY.get_colors()).T
        normals_matrix = (originalPLY.get_normals()).T

        x1 = np.array(x1)
        y1 = np.array(y1)
        z1new = np.array(z1new)

        x = point_cloud[:,0]
    	y = point_cloud[:,1]
    	z = point_cloud[:,2]

    	new_pointcloud = np.zeros((np.size(P1,0), 9))
    	pdb.set_trace()
    	
    	for Prow in range(len(x1)):
    		print 'Prow = ' 
    		print Prow
    		for originalPLY_row in range(len(x)):
    			if ((x[originalPLY_row] == x1[Prow]) and (y[originalPLY_row] == y1[Prow])):
    				print 'found'
    				print Prow
    				#point_cloud[originalPLY_row,2] = z1new[Prow]
    				new_pointcloud[Prow,0:2] = (P1[Prow,0:2])
    				new_pointcloud[Prow,2] = (z1new[Prow])
    				new_pointcloud[Prow,3:6] = (normals_matrix[originalPLY_row,:])
    				new_pointcloud[Prow,6:9] = color_matrix[originalPLY_row,:]

    				


    	pdb.set_trace()
    	#vertex = point_cloud

    	#new_mat = np.concatenate((vertex,normals_matrix,color_matrix),axis=1)
    	new_mat = new_pointcloud
    	vertex_new = map(tuple, new_mat)
    	vertex_new = np.array(vertex_new, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),('nx','f4'),('ny','f4'),('nz','f4'),('diffuse_red', 'u1'), ('diffuse_green', 'u1'),('diffuse_blue', 'u1')])
    	el = PlyElement.describe(vertex_new, 'vertex')

    	
    	PlyData([el], text=True).write(filename)
    	

if __name__ == "__main__":
	P1 = pickle.load( open( "./surface_split_parameter_files/p1_pickle.txt", "rb" ) )
	P2 = pickle.load( open( "./surface_split_parameter_files/p2_pickle.txt", "rb" ) )
	parameters_lin_surface = pickle.load( open( "./surface_split_parameter_files/lin_surface_pickle.txt", "rb" ) )
	parameters_curved_surface = pickle.load( open( "./surface_split_parameter_files/curved_surface_pickle.txt", "rb" ) )
	
	CreatePLY(P1,parameters_lin_surface,'surf1.ply','lin','example-1.ply')
	CreatePLY(P2,parameters_curved_surface,'surf2.ply','quad','example-1.ply')
