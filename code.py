from vortxl import vortxl
from functions import *
import numpy as np
import matplotlib.pyplot as plt

n = 8*2
m = 4

U_inf = np.array([10, 0, 2])
b = 5
L = 0.4
c0 = 0.5
ct = L*c0
x = tapered_wing(m, n, c0, ct, b)

gamma = compute_circulation(x, m, n, U_inf)

l = []
for i in range(n):
    l.append(gamma[m-1 + i*m])

plt.plot(l)
plt.show()

x_center = get_centers(x, m, n)
normals = get_normal_vectors(x, m, n)

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