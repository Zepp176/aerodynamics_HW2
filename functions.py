import numpy as np
from vortxl import vortxl
import matplotlib.pyplot as plt

def lin_tap(ksi, c0, L):
    return (1 - (1 - L)*ksi)*c0

def elliptical(ksi, c0):
    return np.sqrt(1 - (ksi*0.98)**2)*c0

def get_flat_wing(c_fun, m, n, b):
    x = np.zeros((3, (n+1)*(m+1)))
    
    c = c_fun(np.abs(np.linspace(-1, 1, n+1)))
    
    for i in range(1, n+2):
        for j in range(1, m+2):
            x[0, -1 + j + (i-1)*(m+1)] = (-1/4 + (4*j-3)/(4*m))*c[i-1]
            x[1, -1 + j + (i-1)*(m+1)] = (-1/2 + (i-1)/n)*b
            x[2, -1 + j + (i-1)*(m+1)] = 0.0
    
    return x

def get_winglets(c_fun, m, n, b, r, beta):
    x = np.zeros((3, (n+1)*(m+1)))
    
    c = c_fun(np.abs(np.linspace(-r, r, n+1)))
    
    for i in range(1, n+2):
        for j in range(1, m+2):
            if np.abs((-1/2 + (i-1)/n)*b*r) > b/2:
                d = np.abs((-1/2 + (i-1)/n)*b*r) - b/2
                x[0, -1 + j + (i-1)*(m+1)] = (-1/4 + (4*j-3)/(4*m))*c[i-1]
                if (-1/2 + (i-1)/n) > 0:
                    x[1, -1 + j + (i-1)*(m+1)] = np.cos(beta) * d + b/2
                else:
                    x[1, -1 + j + (i-1)*(m+1)] = -np.cos(beta) * d - b/2
                x[2, -1 + j + (i-1)*(m+1)] = np.sin(beta) * d
            else:
                x[0, -1 + j + (i-1)*(m+1)] = (-1/4 + (4*j-3)/(4*m))*c[i-1]
                x[1, -1 + j + (i-1)*(m+1)] = (-1/2 + (i-1)/n)*b*r
                x[2, -1 + j + (i-1)*(m+1)] = 0.0
            
    
    
    return x

def center(points):
    return np.sum(points, axis=1)/4

def get_centers(x, m, n):
    centers = np.zeros((3, m*n))
    points = np.zeros((3, 4))
    for i in range(1, n+1):
        for j in range(1, m+1):
            points[:,0] = x[:, j - 1 + (i-1)*(m+1)]
            points[:,1] = x[:, j     + (i-1)*(m+1)]
            points[:,2] = x[:, j     + (i  )*(m+1)]
            points[:,3] = x[:, j - 1 + (i  )*(m+1)]
            centers[:, j-1+(i-1)*m] = center(points)
    return centers

def normal_v(points):
    n = np.cross(points[:,1]-points[:,0], points[:,2]-points[:,0])
    return n / np.linalg.norm(n)

def get_normal_vectors(x, m, n):
    normals = np.zeros((3, m*n))
    points = np.zeros((3, 4))
    for i in range(1, n+1):
        for j in range(1, m+1):
            points[:,0] = x[:, j - 1 + (i-1)*(m+1)]
            points[:,1] = x[:, j     + (i-1)*(m+1)]
            points[:,2] = x[:, j     + (i  )*(m+1)]
            points[:,3] = x[:, j - 1 + (i  )*(m+1)]
            normals[:, j-1+(i-1)*m] = normal_v(points)
    return normals

def draw_wing(x, m, n):
    for i in range(1, n+1):
        for j in range(1, m+1):
            segments = np.zeros((3, 5))
            segments[:, 0] = x[:, j - 1 + (i-1)*(m+1)]
            segments[:, 1] = x[:, j     + (i-1)*(m+1)]
            segments[:, 2] = x[:, j     + (i  )*(m+1)]
            segments[:, 3] = x[:, j - 1 + (i  )*(m+1)]
            segments[:, 4] = segments[:, 0]
            xs = segments[0, :]
            ys = segments[1, :]
            plt.plot(xs, ys, 'k')

def compute_circulation(mesh, m, n, U, alpha):
    
    U_inf = np.array([np.cos(alpha)*U, 0, np.sin(alpha)*U])
    x1, x2 = get_segments(mesh, m, n, U_inf)
    x = np.empty((3, 4*m*n + 3*n))
    
    x_center = get_centers(mesh, m, n)
    normals = get_normal_vectors(mesh, m, n)
    
    A = np.zeros((n*m, n*m))
    B = np.empty(n*m)
    
    for i in range(n):
        for j in range(m):
            
            x[0, :].fill(x_center[0, j + i*m])
            x[1, :].fill(x_center[1, j + i*m])
            x[2, :].fill(x_center[2, j + i*m])
            w = np.dot(vortxl(x, x1, x2).T, normals[:, i*m + j])
            
            B[i*m + j] = np.dot(normals[:, i*m + j], U_inf)
            
            for k in range(n):
                for l in range(m):
                    A[i*m + j, k*m + l] += w[l*4 + k*(4*m + 3)    ]
                    A[i*m + j, k*m + l] += w[l*4 + k*(4*m + 3) + 1]
                    A[i*m + j, k*m + l] += w[l*4 + k*(4*m + 3) + 2]
                    A[i*m + j, k*m + l] += w[l*4 + k*(4*m + 3) + 3]
                
                A[i*m + j, k*m + m-1] += w[m*4 + k*(4*m + 3)    ]
                A[i*m + j, k*m + m-1] += w[m*4 + k*(4*m + 3) + 1]
                A[i*m + j, k*m + m-1] += w[m*4 + k*(4*m + 3) + 2]
                
    return np.linalg.solve(A, -B)

def get_segments(x, m, n, U_inf):
    
    x1 = np.empty((3, 4*m*n + 3*n))
    x2 = np.empty((3, 4*m*n + 3*n))
    
    for i in range(n):
        for j in range(m):
            x1[:,     4*(j + i*m) + 3*i] = x[:, j   + i*(m+1)    ]
            x1[:, 1 + 4*(j + i*m) + 3*i] = x[:, j+1 + i*(m+1)    ]
            x1[:, 2 + 4*(j + i*m) + 3*i] = x[:, j+1 + (i+1)*(m+1)]
            x1[:, 3 + 4*(j + i*m) + 3*i] = x[:, j   + (i+1)*(m+1)]
            
            x2[:,     4*(j + i*m) + 3*i] = x[:, j+1 + i*(m+1)    ]
            x2[:, 1 + 4*(j + i*m) + 3*i] = x[:, j+1 + (i+1)*(m+1)]
            x2[:, 2 + 4*(j + i*m) + 3*i] = x[:, j   + (i+1)*(m+1)]
            x2[:, 3 + 4*(j + i*m) + 3*i] = x[:, j   + i*(m+1)    ]
        
        x1[:,     4*m + i*(4*m + 3)] = x[:, m + (i+1)*(m+1)] + 60*U_inf
        x1[:, 1 + 4*m + i*(4*m + 3)] = x[:, m + (i+1)*(m+1)]
        x1[:, 2 + 4*m + i*(4*m + 3)] = x[:, m + i*(m+1)    ]
        
        x2[:,     4*m + i*(4*m + 3)] = x[:, m + (i+1)*(m+1)]
        x2[:, 1 + 4*m + i*(4*m + 3)] = x[:, m + i*(m+1)    ]
        x2[:, 2 + 4*m + i*(4*m + 3)] = x[:, m + i*(m+1)    ] + 60*U_inf
    
    return x1, x2

def get_gamma_distribution(gamma, m, n):
    dis = np.zeros(n)
    for i in range(n):
        dis[i] = -gamma[m-1 + i*m]
    return dis

def get_xc(x, m, n):
    xc  = np.empty((3, n*m))
    xc2 = np.empty((3, (n+1)*m))
    
    for i in range(n):
        for j in range(m):
            xc[:, j + i*m] = (x[:, j + i*(m+1)] + x[:, j + (i+1)*(m+1)])/2
    
    for i in range(n+1):
        for j in range(m):
            xc2[:, j + i*m] = (x[:, j + i*(m+1)] + x[:, j+1 + i*(m+1)])/2
    
    return xc, xc2

def get_u(xs, mesh, Gamma, m, n, U, alpha):
    
    U_inf = np.array([np.cos(alpha)*U, 0, np.sin(alpha)*U])
    x1, x2 = get_segments(mesh, m, n, U_inf)
    x = np.empty((3, 4*m*n + 3*n))
    
    Gamma_seg = np.empty(4*m*n + 3*n)
    for i in range(n):
        for j in range(m):
            Gamma_seg[4*j + (4*m + 3)*i    ] = Gamma[j + i*m]
            Gamma_seg[4*j + (4*m + 3)*i + 1] = Gamma[j + i*m]
            Gamma_seg[4*j + (4*m + 3)*i + 2] = Gamma[j + i*m]
            Gamma_seg[4*j + (4*m + 3)*i + 3] = Gamma[j + i*m]
        
        Gamma_seg[4*m + (4*m + 3)*i    ] = Gamma[m-1 + i*m]
        Gamma_seg[4*m + (4*m + 3)*i + 1] = Gamma[m-1 + i*m]
        Gamma_seg[4*m + (4*m + 3)*i + 2] = Gamma[m-1 + i*m]
    
    u = np.empty((3, len(xs[0])))
    
    for i in range(len(xs[0])):
        
        x[0, :].fill(xs[0, i])
        x[1, :].fill(xs[1, i])
        x[2, :].fill(xs[2, i])
        
        u[:, i] = np.sum(vortxl(x, x1, x2, Gamma_seg), axis=1) + U_inf
    
    return u

def get_F(mesh, Gamma, m, n, rho, U, alpha):
    
    xc, xc2 = get_xc(mesh, m, n)
    
    us = get_u(xc, mesh, Gamma, m, n, U, alpha)
    
    F = np.zeros(3)
    
    for i in range(n):
        for j in range(m-1):
            segment = mesh[:, i*(m+1) + j] - mesh[:, (i+1)*(m+1) + j]
            F += rho * np.cross(us[:, i*m + j], segment) * Gamma[i*m + j]
            
            segment = mesh[:, (i+1)*(m+1) + j+1] - mesh[:, i*(m+1) + j+1]
            F += rho * np.cross(us[:, i*m + j+1], segment) * Gamma[i*m + j]
        
        segment = mesh[:, i*(m+1) + m-1] - mesh[:, (i+1)*(m+1) + m-1]
        F += rho * np.cross(us[:, i*m + m-1], segment) * Gamma[i*m + m-1]
    
    return F

def get_coefs(F, alpha, rho, S, U):
    
    L = np.cos(alpha) * F[2] - np.sin(alpha) * F[0]
    D = np.cos(alpha) * F[0] + np.sin(alpha) * F[2]
    return L/(0.5*rho*U**2*S), D/(0.5*rho*U**2*S)

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


