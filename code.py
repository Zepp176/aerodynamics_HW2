from vortxl import vortxl
from functions import *
import numpy as np
import matplotlib.pyplot as plt

n = 8*3
m = 5

U_inf = np.array([10, 0, 1])
b = 3
L = 0.5
c0 = 0.5
ct = L*c0
x = tapered_wing(m, n, c0, ct, b)

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
                
                #print("influence of cell {} on cell {}: ".format(idx_cell_from, idx_cell_to), end='')
                
                ind_velocity = ind_vel(x_center[:, idx_cell_to-1], i, j, m, n, x, U_inf)
                
                #print("{:.3f} m/s".format(ind_velocity[2]))
                
                normal = normals[:, idx_cell_to-1]
                A[idx_cell_to-1, idx_cell_from-1] = np.dot(normal, ind_velocity)

gamma = np.linalg.solve(A, B)
print(gamma)

xs = x[0,:]
ys = x[1,:]
xs_c = x_center[0,:]
ys_c = x_center[1,:]

plt.figure(figsize=(4,8))
draw_wing(x/b, m, n)
plt.plot(xs/b, ys/b, 'ok')
plt.plot(xs_c/b, ys_c/b, 'or')
plt.ylabel("y/b")
plt.xlabel("x/b")
plt.axis('equal')
plt.show()
