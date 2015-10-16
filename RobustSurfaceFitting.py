import numpy as np 
import numpy.linalg as la
import pdb
import plyfile
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sys, getopt

def LinSurfFit(plydata):
	x_orig = plydata['vertex']['x']
	y_orig = plydata['vertex']['y']
	z_orig = plydata['vertex']['z']

	x = x_orig
	y = y_orig
	z = z_orig

	
	# Robust Fitting (Linear)
	tau_s = 5E-7
	print 'tau_s = '+str(tau_s)
	x = x_orig
	y = y_orig
	z = z_orig

	n = len(x)
	D = np.matrix(np.empty([n,3]))
	D[:,0] = np.matrix(x).T
	D[:,1] = np.matrix(y).T
	D[:,2] = np.matrix(np.ones([n,1]))

	v = np.matrix(z).T

	# Solve for least square solution
	a,e,r,s = la.lstsq(D, v)
	print "Linear surface fitting"
	
	Z_linearFit = D*a

	#fig = plt.figure(figsize=plt.figaspect(3.))
	#ax = fig.add_subplot(3, 1, 1, projection='3d')
	'''
	fig = plt.figure()
	ax = fig.gca(projection='3d')

	
	ax.plot_trisurf(x, y, np.array(Z_linearFit).flatten(), cmap=cm.jet, linewidth=0.2)
	ax.set_title('Initial Estimate')
	plt.show()

	fig = plt.figure()
	ax = fig.gca(projection='3d')

	ax.plot_trisurf(x, y, np.array(v).flatten(), cmap=cm.jet, linewidth=0.2)
	ax.set_title('Initial Actual')

	plt.show()
	'''
	r = np.array(v-Z_linearFit)*np.array(v-Z_linearFit)
	E = np.median(r)

	iter = 0
	while E>tau_s:
		iter = iter+1
		
		v = (v[r<E]).T
		x = x[np.array((r<E).T).ravel()]
		y = y[np.array((r<E).T).ravel()]
		n = len(x)
		D = np.matrix(np.empty([n,3]))
		D[:,0] = np.matrix(x).T
		D[:,1] = np.matrix(y).T
		D[:,2] = np.matrix(np.ones([n,1]))

		# Solve for least square solution
		a,e,r,s = la.lstsq(D, v)
		Z_linearFit = D*a
		r = np.array(v-Z_linearFit)*np.array(v-Z_linearFit)
		E = np.median(r)
		print 'At iter = '+str(iter)+' error E is = ' + str(E)

	'''
	#ax = fig.add_subplot(3, 1, 2, projection='3d')
	fig = plt.figure()
	ax = fig.gca(projection='3d')

	ax.plot_trisurf(x, y, np.array(Z_linearFit).flatten(), cmap=cm.jet, linewidth=0.2)
	ax.set_title('Final Estimate')
	plt.show()

	#ax = fig.add_subplot(3, 1, 3, projection='3d')
	fig = plt.figure()
	ax = fig.gca(projection='3d')

	ax.plot_trisurf(x, y, np.array(v).flatten(), cmap=cm.jet, linewidth=0.2)
	ax.set_title('Final Actual')
	plt.show()
	'''
	return a

def QuadSurfFit(plydata):
	x_orig = plydata['vertex']['x']
	y_orig = plydata['vertex']['y']
	z_orig = plydata['vertex']['z']

	x = x_orig
	y = y_orig
	z = z_orig

	#Quadratic Surface Fitting
	n = len(x)
	D = np.matrix(np.empty([n,6]))
	D[:,0] = np.matrix(x*x).T
	D[:,1] = np.matrix(y*y).T
	D[:,2] = np.matrix(x*y).T
	D[:,3] = np.matrix(x).T
	D[:,4] = np.matrix(y).T
	D[:,5] = np.matrix(np.ones([n,1]))

	v = np.matrix(z).T

	# Solve for least square solution
	a,e,r,s = la.lstsq(D, v)
	print "Quadratic surface fitting"
	
	Z_quadFit = D*a
	'''
	fig = plt.figure()
	ax = fig.gca(projection='3d')

	ax.plot_trisurf(x, y, np.array(Z_quadFit).flatten(), cmap=cm.jet, linewidth=0.2)
	ax.set_title('Initial Estimate')
	plt.show()

	fig = plt.figure()
	ax = fig.gca(projection='3d')

	ax.plot_trisurf(x, y, np.array(v).flatten(), cmap=cm.jet, linewidth=0.2)
	ax.set_title('Initial Actual')

	plt.show()

	'''
	# Robust Fitting (Quadratic)
	tau_s = 5E-7
	print 'tau_s = '+str(tau_s)
	r = np.array(v-Z_quadFit)*np.array(v-Z_quadFit)
	E = np.median(r)

	iter = 0
	while E>tau_s:
		iter = iter+1
		
		v = (v[r<E]).T
		x = x[np.array((r<E).T).ravel()]
		y = y[np.array((r<E).T).ravel()]
		n = len(x)
		D = np.matrix(np.empty([n,6]))
		D[:,0] = np.matrix(x*x).T
		D[:,1] = np.matrix(y*y).T
		D[:,2] = np.matrix(x*y).T
		D[:,3] = np.matrix(x).T
		D[:,4] = np.matrix(y).T
		D[:,5] = np.matrix(np.ones([n,1]))

		# Solve for least square solution
		a,e,r,s = la.lstsq(D, v)
		Z_quadFit = D*a
		r = np.array(v-Z_quadFit)*np.array(v-Z_quadFit)
		E = np.median(r)
		print 'At iter = '+str(iter)+' error E is = ' + str(E)
	'''
	fig = plt.figure()
	ax = fig.gca(projection='3d')

	ax.plot_trisurf(x, y, np.array(Z_quadFit).flatten(), cmap=cm.jet, linewidth=0.2)
	ax.set_title('Final Estimate')

	plt.show()

	fig = plt.figure()
	ax = fig.gca(projection='3d')

	ax.plot_trisurf(x, y, np.array(v).flatten(), cmap=cm.jet, linewidth=0.2)
	ax.set_title('Actual')

	plt.show()
	'''
	return a

def main(argv):
	opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])

   # Read data in input file, which is in ply format.
	plydata = PlyData.read(args[0])

	if (args[1] == 'quad'):
		a = QuadSurfFit(plydata)

	elif (args[1] == 'lin'):
		a = LinSurfFit(plydata)

	return a
	
	


if __name__ == "__main__":
	a = main(sys.argv[1:])
	print 'final a = '
	print a
