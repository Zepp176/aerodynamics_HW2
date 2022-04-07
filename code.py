from vortxl import vortxl
from functions import *
import numpy as np
import matplotlib.pyplot as plt

n = 8
m = 4

U_inf = np.array([10, 0, 0])
c0 = 0.4
ct = 0.2
x = tapered_wing(n, m, c0, ct, 2)

x_center = get_centers(x, m, n)
normals = get_normal_vectors(x, m, n)

A = np.zeros((n*m, n*m))

for i in range(m*n):        # Quel point on considère
    for j in range(m*n):    # Quel point influence
        A[i, j] = 0

xs = x[0,:]
ys = x[1,:]
xs_c = x_center[0,:]
ys_c = x_center[1,:]

plt.figure(figsize=(4,8))
draw_wing(x, m, n)
plt.plot(xs, ys, 'o')
plt.plot(xs_c, ys_c, 'o')
plt.axis('equal')
plt.show()
