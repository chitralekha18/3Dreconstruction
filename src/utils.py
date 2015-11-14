__author__ = 'abhinavkashyap'
import pickle
from plyfile import PlyData
import numpy as np


def get_ply_array(plydata):
    x =  np.matrix(plydata['vertex']['x']).reshape(-1, 1)
    y = np.matrix(plydata['vertex']['y']).reshape(-1, 1)
    z = np.matrix(plydata['vertex']['z']).reshape(-1, 1)
    nx = np.matrix(plydata['vertex']['nx']).reshape(-1, 1)
    ny = np.matrix(plydata['vertex']['ny']).reshape(-1, 1)
    nz = np.matrix(plydata['vertex']['nz']).reshape(-1, 1)
    red = np.matrix(plydata['vertex']['red']).reshape(-1, 1)
    green = np.matrix(plydata['vertex']['green']).reshape(-1, 1)
    blue = np.matrix(plydata['vertex']['blue']).reshape(-1, 1)
    alpha = np.matrix(plydata['vertex']['alpha']).reshape(-1, 1)

    return np.hstack((x, y, z, nx, ny, nz, red, green, blue, alpha))

def get_new_surface(pointcloud, designmatrix, params):
    """
    This method gets the fitted  surface
    :param pointcloud: point cloud representing the surface
    :param designmatrix: The D matrix that is used for fitting
    :param params: The parameters representing the linear surface
    :return: point cloud representing the planar surface
    """
    new_z = designmatrix * params
    pointcloud[:, 2] = new_z
    # pointcloud[:, 2] = np.zeros((pointcloud.shape[0], 1))
    return pointcloud

def get_new_line(pointcloud, designmatrix, params):
    new_y = designmatrix * params;
    pointcloud[:, 1] = new_y
    return pointcloud

def pickledump(obj, file):
    """

    :param obj: object that has to be pickled
    :param file: file is an open file object for writing
    :return:
    """
    pickle.dump(obj, file)

def pickleload(filename):
    return pickle.load(open(filename, 'r'))


def pickle_front_left_surface(filename):
    pickle_file = open("../pickled_files/fl_surface.pkl", "w")
    fl_file = open(filename)
    plydata = PlyData.read(fl_file)
    plyarray = get_ply_array(plydata)
    pickledump(plyarray, pickle_file)

def write_ply_file(pointcloud, filelocation):
    """
    This function writes the plyfile
    :param pointcloud:
    :return: None
    """
    file = open(filelocation, "w")
    header  =  \
'''ply
format ascii 1.0
'''
    header_mid = 'element vertex '+str(pointcloud.shape[0])
    header_end = \
'''
property float x
property float y
property float z
property float nx
property float ny
property float nz
property uchar red
property uchar green
property uchar blue
property uchar alpha
end_header
'''
    file.write(header + header_mid + header_end)
    for eachRow in pointcloud:
        x = "{:.6f}".format(eachRow.item(0,0))
        y = "{:.6f}".format(eachRow.item(0,1))
        z = "{:.6f}".format(eachRow.item(0,2))
        nx = "{:.6f}".format(eachRow.item(0,3))
        ny = "{:.6f}".format(eachRow.item(0,4))
        nz = "{:.6f}".format(eachRow.item(0,5))
        red = int(eachRow.item(0,6))
        green = int(eachRow.item(0,7))
        blue = int(eachRow.item(0,8))
        alpha = int((eachRow.item(0,9)))
        file.write(str(x) + " " + str(y) +  " " + str(z) + " " + str(nx) + " " + str(ny) + " " +
                   str(nz) + " " + str(red) + " " + str(green) + " " + str(blue) + " " + str(alpha) + "\n")
    file.close()
if __name__ == "__main__":
    pickle_front_left_surface("../front_left_surface/fl_surface.ply")


