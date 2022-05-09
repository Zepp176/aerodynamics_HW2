from vortxl import vortxl
from functions import *
import numpy as np
import matplotlib.pyplot as plt

n = 50
m = 4

U = 10
alpha = np.pi/36
b = 1
rho = 1.225

AR = 5
L = 0.5
c0 = b/AR * 2/(1+L)
S = b**2/AR

chord = lambda ksi : lin_tap(ksi, c0, L)
mesh = get_flat_wing(chord, m, n, b)

plt.figure(figsize=(4,8))
draw_wing(mesh, m, n)
plt.ylabel("y")
plt.xlabel("x")
plt.grid()
plt.axis('equal')
plt.show()

C_D_arr = []
C_L_arr = []

for alpha in np.linspace(0, np.pi/180*45, 20):
    
    print(alpha)
    
    gamma = compute_circulation(mesh, m, n, U, alpha)
    circ = get_gamma_distribution(gamma, m, n)
    ksi = np.linspace(-1, 1, len(circ))
    
    F = get_F(mesh, gamma, m, n, rho, U, alpha)
    
    L = np.cos(alpha) * F[2] - np.sin(alpha) * F[0]
    D = np.cos(alpha) * F[0] + np.sin(alpha) * F[2]
    C_L = L / (0.5*rho*U**2*S)
    C_D = D / (0.5*rho*U**2*S)
    
    print(C_L, C_D)
    
    C_L_arr.append(C_L)
    C_D_arr.append(C_D)

plt.figure(figsize=(7,5))
plt.grid()
plt.plot(np.linspace(0, 45, 20), C_L_arr)
plt.plot(np.linspace(0, 45, 20), C_D_arr)
plt.legend(["C_L", "C_D"])
plt.show()

