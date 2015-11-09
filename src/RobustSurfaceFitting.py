import numpy as np
import numpy.linalg as la
from plyfile import PlyData


def LinSurfFit(x_orig, y_orig, z_orig):
    x = x_orig
    y = y_orig
    z = z_orig


    # Robust Fitting (Linear)
    tau_s = 5E-2 #5E-7
    print 'tau_s = ' + str(tau_s)
    x = x_orig
    y = y_orig
    z = z_orig

    n = len(x)
    D = np.matrix(np.empty([n, 3]))
    D[:, 0] = np.matrix(x).T
    D[:, 1] = np.matrix(y).T
    D[:, 2] = np.matrix(np.ones([n, 1]))

    v = np.matrix(z).T
    indices_chosen = np.array(range(len(x)))

    # Solve for least square solution
    a, e, r, s = la.lstsq(D, v)
    print "Linear surface fitting"

    Z_linearFit = D * a

    # fig = plt.figure(figsize=plt.figaspect(3.))
    # ax = fig.add_subplot(3, 1, 1, projection='3d')
    r = np.array(v - Z_linearFit) * np.array(v - Z_linearFit)
    E = np.median(r)
    
    iter = 0
    while E > tau_s:
        iter = iter + 1

        v = (v[r < E]).T
        x = x[np.array((r < E).T).ravel()]
        y = y[np.array((r < E).T).ravel()]
        n = len(x)
        D = np.matrix(np.empty([n, 3]))
        D[:, 0] = np.matrix(x).T
        D[:, 1] = np.matrix(y).T
        D[:, 2] = np.matrix(np.ones([n, 1]))
        
        indices_chosen = indices_chosen[np.array((r < E).T).ravel()]
        
        # Solve for least square solution
        a, e, r, s = la.lstsq(D, v)
        Z_linearFit = D * a
        r = np.array(v - Z_linearFit) * np.array(v - Z_linearFit)
        E = np.median(r)
        print 'At iter = ' + str(iter) + ' error E is = ' + str(E)

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot_trisurf(x, y, np.array(Z_linearFit).flatten(), cmap=cm.jet, linewidth=0.2)
    # ax.set_title('Final Estimate')
    # plt.draw()
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot_trisurf(x, y, np.array(v).flatten(), cmap=cm.jet, linewidth=0.2)
    # ax.set_title('Final Actual')
    # plt.draw()
    # plt.show()
    return a, D, indices_chosen


def QuadSurfFit(x_orig, y_orig, z_orig):
    x = x_orig
    y = y_orig
    z = z_orig

    # Quadratic Surface Fitting
    n = len(x)
    D = np.matrix(np.empty([n, 6]))
    D[:, 0] = np.matrix(x * x).T
    D[:, 1] = np.matrix(y * y).T
    D[:, 2] = np.matrix(x * y).T
    D[:, 3] = np.matrix(x).T
    D[:, 4] = np.matrix(y).T
    D[:, 5] = np.matrix(np.ones([n, 1]))

    v = np.matrix(z).T
    indices_chosen = np.array(range(len(x)))

    # Solve for least square solution
    a, e, r, s = la.lstsq(D, v)
    print "Quadratic surface fitting"

    Z_quadFit = D * a

    # Robust Fitting (Quadratic)
    tau_s = 5E-2
    print 'tau_s = ' + str(tau_s)
    r = np.array(v - Z_quadFit) * np.array(v - Z_quadFit)
    E = np.median(r)

    iter = 0
    while E > tau_s:
        iter = iter + 1

        v = (v[r < E]).T
        x = x[np.array((r < E).T).ravel()]
        y = y[np.array((r < E).T).ravel()]
        n = len(x)
        D = np.matrix(np.empty([n, 6]))
        D[:, 0] = np.matrix(x * x).T
        D[:, 1] = np.matrix(y * y).T
        D[:, 2] = np.matrix(x * y).T
        D[:, 3] = np.matrix(x).T
        D[:, 4] = np.matrix(y).T
        D[:, 5] = np.matrix(np.ones([n, 1]))

        indices_chosen = indices_chosen[np.array((r < E).T).ravel()]
        
        # Solve for least square solution
        a, e, r, s = la.lstsq(D, v)
        Z_quadFit = D * a
        r = np.array(v - Z_quadFit) * np.array(v - Z_quadFit)
        E = np.median(r)
        print 'At iter = ' + str(iter) + ' error E is = ' + str(E)


    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot_trisurf(x, y, np.array(Z_quadFit).flatten(), cmap=cm.jet, linewidth=0.2)
    # ax.set_title('Final Estimate')
    # plt.draw()
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot_trisurf(x, y, np.array(v).flatten(), cmap=cm.jet, linewidth=0.2)
    # ax.set_title('Actual')
    # plt.draw()
    # plt.show()

    return a, D, indices_chosen


def main(x, y, z, ftype):
    if (ftype == 'quad'):
        a = QuadSurfFit(x, y, z)

    elif (ftype == 'lin'):
        a = LinSurfFit(x, y, z)

    return a


if __name__ == "__main__":
    plydata = PlyData.read('example-1.ply')
    x = (plydata['vertex']['x'])
    y = (plydata['vertex']['y'])
    z = (plydata['vertex']['z'])

    a = main(x, y, z, 'quad')
    print 'final a = '
    print a
