from functions import *
import numpy as np
import matplotlib.pyplot as plt

# mesh
n = 2000
m = 5

# airflow
U = 100
alpha = np.pi/36
rho = 1.225

# wing
b = 1
AR = 8
L = 0.2
c0 = b/AR * 2/(1+L)
S = b**2/AR

mesh = get_flat_wing(lambda ksi : lin_tap(ksi, c0, L), m, n, b)
gamma = compute_circulation(mesh, m, n, U, alpha)

l = get_l(mesh, gamma, m, n, b, rho, U, alpha)
circ = get_gamma_distribution(gamma, m, n)

#%%

plt.figure(figsize=(7, 5))
plt.plot(np.linspace(-1, 1, n), l/(circ*rho*U))
plt.xlabel("$\\xi$")
plt.ylabel("$\\frac{l(\\xi)}{\\Gamma U_{\\infty} \\rho}$")
plt.title("AR = {}, $\Lambda = {}$, n = {}".format(AR, L, n))
plt.xlim(left=0, right=1)
plt.ylim(top=1, bottom=0.997)
plt.grid()
plt.tight_layout()
plt.savefig("figures/part1_ratio.png", dpi=300)

#%%

m = 8
n = 100

mesh = get_flat_wing(lambda ksi : lin_tap(ksi, c0, L), m, n, b)
gamma = compute_circulation(mesh, m, n, U, alpha)

F = get_F(mesh, gamma, m, n, rho, U, alpha)
print("F_x = {}".format(F[0]))
print("F_x = {}".format(F[1]))
print("F_x = {}".format(F[2]))

C_L, C_D = get_coefs(F, alpha, rho, S, U)