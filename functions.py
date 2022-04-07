import numpy as np
from vortxl import vortxl
import matplotlib.pyplot as plt

def tapered_wing(m, n, c0, ct, b):
    x = np.zeros((3, (n+1)*(m+1)))
    
    c = np.zeros(n+1)
    c[:(n//2)+1] = np.linspace(ct, c0, n//2 + 1)
    c[n//2:] = np.linspace(c0, ct, n//2 + 1)
    
    for i in range(1, n+2):
        for j in range(1, m+2):
            x[0, -1 + j + (i-1)*(m+1)] = (-1/4 + (4*j-3)/(4*m))*c[i-1]
            x[1, -1 + j + (i-1)*(m+1)] = (-1/2 + (i-1)/n)*b
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
    
def ind_vel(x, i, j, m, n, xs, U_inf):
    
    segments = np.zeros((3, 4))
    segments[:, 0] = xs[:, j - 1 + (i-1)*(m+1)]
    segments[:, 1] = xs[:, j     + (i-1)*(m+1)]
    segments[:, 2] = xs[:, j     + (i  )*(m+1)]
    segments[:, 3] = xs[:, j - 1 + (i  )*(m+1)]
    
    if j == m:
        # On est dans le cas d'un horseshoe
        res =  vortxl(x, segments[:,2] + U_inf*60, segments[:,2])
        res += vortxl(x, segments[:,2], segments[:,3])
        res += vortxl(x, segments[:,3], segments[:,0])
        res += vortxl(x, segments[:,0], segments[:,1])
        res += vortxl(x, segments[:,1], segments[:,1] + U_inf*60)
        return res
    
    else:
        # On est pas dans un horseshoe
        res =  vortxl(x, segments[:,0], segments[:,1])
        res += vortxl(x, segments[:,1], segments[:,2])
        res += vortxl(x, segments[:,2], segments[:,3])
        res += vortxl(x, segments[:,3], segments[:,0])
        return res

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

def compute_circulation(x, m, n, U_inf):
    x_center = get_centers(x, m, n)
    normals = get_normal_vectors(x, m, n)
    
    A = np.zeros((n*m, n*m))
    B = np.zeros(n*m)
    
    for k in range(1, n+1):    # Quel point on considère (le point i)
        for l in range(1, m+1):
            
            idx_cell_to   = l+(k-1)*m
            
            normal = normals[:, idx_cell_to-1]
            B[idx_cell_to-1] = np.dot(normal, U_inf)
            
            for i in range(1, n+1):        # Quel point influence (le point j)
                for j in range(1, m+1):
                    
                    idx_cell_from = j+(i-1)*m
                    
                    ind_velocity = ind_vel(x_center[:, idx_cell_to-1], i, j, m, n, x, U_inf)
                    normal = normals[:, idx_cell_to-1]
                    A[idx_cell_to-1, idx_cell_from-1] = np.dot(normal, ind_velocity)
                    
    return np.linalg.solve(A, B)