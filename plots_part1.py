from functions import *
import numpy as np
import matplotlib.pyplot as plt

# mesh
n = 50
m = 8

# airflow
U = 100
alpha = np.pi/36
rho = 1.225

# wing
b = 1
AR = 8
L = 0.5
c0 = b/AR * 2/(1+L)
S = b**2/AR

mesh = get_flat_wing(lambda ksi : lin_tap(ksi, c0, L), m, n, b)
gamma = compute_circulation(mesh, m, n, U, alpha)

l = get_l(mesh, gamma, m, n, b, rho, U, alpha)
circ = get_gamma_distribution(gamma, m, n)

plt.figure(figsize=(3, 6))
draw_wing(mesh, m, n)
plt.ylabel("y")
plt.xlabel("x")
plt.grid()
plt.axis('equal')
plt.tight_layout()
plt.savefig("figures/wing_2D.png", dpi=300)

for i in range(n):
    print("{}".format(i+1), end=" & ")
    for j in range(m-1):
        print("{:.4f}".format(gamma[j+i*m]), end=" & ")
    print("{:.4f}".format(gamma[(i+1)*m - 1]), end=" \\\\\n")

#%%

plt.figure(figsize=(7, 5))
plt.plot(np.linspace(-1, 1, n), l/(circ*rho*U))
plt.xlabel("$\\xi$")
plt.ylabel("$\\frac{l(\\xi)}{\\Gamma(\\xi) U_{\\infty} \\rho}$")
plt.legend(["AR = {}, $\Lambda = {}$, n = {}".format(AR, L, n)])
plt.xlim(left=0, right=1)
plt.ylim(top=1, bottom=0.996)
plt.grid()
plt.tight_layout()
plt.show()
#plt.savefig("figures/part1_ratio.png", dpi=300)

#%%

m = 5
n = 50

mesh = get_flat_wing(lambda ksi : lin_tap(ksi, c0, L), m, n, b)
gamma = compute_circulation(mesh, m, n, U, alpha)

F = get_F(mesh, gamma, m, n, rho, U, alpha)
print("F_x = {:.2f} N".format(F[0]))
print("F_y = {:.4f} N".format(F[1]))
print("F_z = {:.2f} N".format(F[2]))
print("||F|| = {:.2f} N".format(np.linalg.norm(F)))

C_L, C_D = get_coefs(F, alpha, rho, S, U)
print("L = {:.2f} N".format(C_L*(0.5*rho*U**2*S)))
print("D = {:.2f} N".format(C_D*(0.5*rho*U**2*S)))
print("C_L = {:.4f}".format(C_L))
print("C_D = {:.5f}".format(C_D))

#%% A0 and k

alphas = np.linspace(0, 10*np.pi/180, 50)
C_L = np.empty(len(alphas))
C_D = np.empty(len(alphas))

for i, alpha in enumerate(alphas):
    gamma = compute_circulation(mesh, m, n, U, alpha)
    
    C_L[i], C_D[i] = get_coefs(get_F(mesh, gamma, m, n, rho, U, alpha), alpha, rho, S, U)

    print(i)

k = np.mean(C_D[1:]/C_L[1:]**2)
A0 = (C_L[-1] - C_L[0]) / alphas[-1]

#%%

print(k)
print(A0)

plt.figure(figsize=(8, 4))

plt.subplot(121)
plt.plot(alphas*180/np.pi, C_L)
plt.grid()
plt.xlabel("angle of attack [°]")
plt.ylabel("$C_L$")

plt.subplot(122)
plt.plot(alphas*180/np.pi, C_D)
plt.grid()
plt.xlabel("angle of attack [°]")
plt.ylabel("$C_D$")

plt.tight_layout()
plt.savefig("figures/part1_CLCD.png", dpi=300)
