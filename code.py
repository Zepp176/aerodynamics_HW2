from functions import *
import numpy as np
import matplotlib.pyplot as plt

n = 10
m = 3

U = 100
alpha = np.pi/36
b = 1
rho = 1.225
beta = 90*np.pi/180
r = 1.4

AR = 8
L = 0.5
c0 = b/AR * 2/(1+L)
S = b**2/AR

chord = lambda ksi : lin_tap(ksi, c0, L)
#mesh = get_winglets(chord, m, n, b, r, beta)
mesh = get_flat_wing(chord, m, n, b)

plt.figure(figsize=(4,8))
draw_wing(mesh, m, n)
plt.ylabel("y")
plt.xlabel("x")
plt.grid()
plt.axis('equal')
plt.show()

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(mesh[0,:], mesh[1,:], mesh[2,:])
set_axes_equal(ax)
plt.show()

gamma = compute_circulation(mesh, m, n, U, alpha)
F = get_F(mesh, gamma, m, n, rho, U, alpha)
C_L, C_D = get_coefs(F, alpha, rho, S, U)
print(C_L, C_D)
print(C_L/C_D)