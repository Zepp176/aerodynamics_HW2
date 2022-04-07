from vortxl import vortxl
from functions import *
import numpy as np
import matplotlib.pyplot as plt

n = 30
m = 8

U = 10
alpha = np.pi/36
U_inf = np.array([np.cos(alpha)*U, 0, np.sin(alpha)*U])
b = 1

plt.figure(figsize=(6, 4))

ARs = [3, 4, 5, 7, 9, 11]
for AR in ARs:
    L = 0.4
    c0 = b/AR * 2/(1+L)
    
    chord = lambda x : lin_tap(x, c0, L)
    x = get_flat_wing(chord, m, n, b)
    
    gamma = compute_circulation(x, m, n, U_inf)
    circ_distr = get_gamma_distribution(gamma, m, n)
    ksi = np.linspace(-b/2 + b/(2*n), b/2 - b/(2*n), n)*2/b
    
    plt.plot(ksi[n//2-1:], circ_distr[n//2-1:]/(U*b*alpha))
    print("AR={} finish".format(AR))

leg = []
for AR in ARs:
    leg.append("AR = {}".format(AR))

plt.grid()
plt.title('$\\Lambda = {}$'.format(L))
plt.xlabel('$\\xi$')
plt.ylabel('$\\Gamma/(U_{\\infty} b \\alpha)$')
plt.legend(leg)
plt.savefig("figures/fig1.png", dpi=300)

x_center = get_centers(x, m, n)
normals = get_normal_vectors(x, m, n)

xs = x[0,:]
ys = x[1,:]
xs_c = x_center[0,:]
ys_c = x_center[1,:]

plt.figure(figsize=(4,8))
draw_wing(x, m, n)
plt.ylabel("y")
plt.xlabel("x")
plt.axis('equal')
plt.show()