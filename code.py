from vortxl import vortxl
from functions import *
import numpy as np
import matplotlib.pyplot as plt

n = 64
m = 8

U = 10
alpha = np.pi/36
b = 1

AR = 5
L = 1
c0 = b/AR * 2/(1+L)
        
chord = lambda ksi : lin_tap(ksi, c0, L)
mesh = get_flat_wing(chord, m, n, b)
xc = get_xc(mesh, m, n)[0]

plt.figure(figsize=(4,8))
draw_wing(mesh, m, n)
plt.ylabel("y")
plt.xlabel("x")
plt.grid()
plt.axis('equal')
plt.show()

gamma = compute_circulation(mesh, m, n, U, alpha)
circ = get_gamma_distribution(gamma, m, n)
ksi = np.linspace(-1, 1, len(circ))
plt.plot(ksi, circ/(U*b*alpha))
plt.grid()
plt.xlabel("$\\xi$")
plt.ylabel("$\\Gamma$")
plt.show()