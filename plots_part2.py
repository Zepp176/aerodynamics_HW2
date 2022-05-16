from functions import *
import numpy as np
import matplotlib.pyplot as plt

# mesh
n = 50
m = 10

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

# winglets
r = 1.4
beta = np.pi/2

mesh_winglet = get_winglets(lambda ksi : lin_tap(ksi, c0, L), m, n, b, r, beta)
mesh = get_flat_wing(lambda ksi : lin_tap(ksi, c0, L), m, n, b)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(mesh_winglet[0,:], mesh_winglet[1,:], mesh_winglet[2,:])
set_axes_equal(ax)
plt.savefig("figures/wing_3D.png", dpi=300)

gamma_winglet = compute_circulation(mesh_winglet, m, n, U, alpha)
circ_winglet = get_gamma_distribution(gamma_winglet, m, n)
circ_winglet /= U*b*alpha

gamma = compute_circulation(mesh, m, n, U, alpha)
circ = get_gamma_distribution(gamma, m, n)
circ /= U*b*alpha

#%%

plt.figure(figsize=(7, 5))
plt.plot(np.linspace(-r, r, len(circ_winglet)), circ_winglet)
plt.plot(np.linspace(-1, 1, len(circ)), circ)
plt.grid()
plt.xlabel("$\\xi$")
plt.ylabel("$\\frac{\\Gamma}{U_{\\infty} b \\alpha}$")
plt.legend(["Wing with winglet", "Wing without winglet"])
plt.title("dimensionless circulation distribution with and without winglets")
plt.tight_layout()
plt.savefig("figures/comparaison_winglets.png", dpi=300)

#%% C_L and C_D

C_L, C_D = get_coefs(get_F(mesh, gamma, m, n, rho, U, alpha), alpha, rho, S, U)
C_L_w, C_D_w = get_coefs(get_F(mesh_winglet, gamma_winglet, m, n, rho, U, alpha), alpha, rho, S, U)

print("Without winglets :\n    C_L = {:.3f}, C_D = {:.5f}, L/D = {:.1f}".format(C_L, C_D, C_L/C_D))
print("With winglets :\n    C_L = {:.3f}, C_D = {:.5f}, L/D = {:.1f}".format(C_L_w, C_D_w, C_L_w/C_D_w))

#%% A0 and k

n = 50
m = 4

mesh_winglet = get_winglets(lambda ksi : lin_tap(ksi, c0, L), m, n, b, r, beta)
mesh = get_flat_wing(lambda ksi : lin_tap(ksi, c0, L), m, n, b)

alphas = np.linspace(0, 10*np.pi/180, 50)
C_L = np.empty(len(alphas))
C_D = np.empty(len(alphas))
C_L_w = np.empty(len(alphas))
C_D_w = np.empty(len(alphas))

for i, alpha in enumerate(alphas):
    gamma_winglet = compute_circulation(mesh_winglet, m, n, U, alpha)
    gamma = compute_circulation(mesh, m, n, U, alpha)
    
    C_L[i], C_D[i] = get_coefs(get_F(mesh, gamma, m, n, rho, U, alpha), alpha, rho, S, U)
    C_L_w[i], C_D_w[i] = get_coefs(get_F(mesh_winglet, gamma_winglet, m, n, rho, U, alpha), alpha, rho, S, U)

    print(i)

#%%

A0 = (C_L[-1] - C_L[0]) / alphas[-1]
A0_w = (C_L_w[-1] - C_L_w[0]) / alphas[-1]

print("Without winglets : A0 = {:.2f}".format(A0))
print("With    winglets : A0 = {:.2f}".format(A0_w))

plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.plot(alphas*180/np.pi, C_L)
plt.plot(alphas*180/np.pi, C_L_w)
plt.grid()
plt.xlabel("angle of attack [°]")
plt.ylabel("$C_L$")
plt.legend(["Without winglets", "With winglets"])

plt.subplot(122)
plt.plot(alphas*180/np.pi, C_D)
plt.plot(alphas*180/np.pi, C_D_w)
plt.grid()
plt.xlabel("angle of attack [°]")
plt.ylabel("$C_D$")
plt.legend(["Without winglets", "With winglets"])

plt.tight_layout()
plt.savefig("figures/comparaison_CL_CD.png", dpi=300)

#%%

k = np.mean(C_D[1:]/C_L[1:]**2)
k_w = np.mean(C_D_w[1:]/C_L_w[1:]**2)
print("Without winglets : k = {:.4f}".format(k))
print("With    winglets : k = {:.4f}".format(k_w))

plt.figure(figsize=(7, 5))
plt.plot(C_D, C_L)
plt.grid()
plt.xlabel("$C_D$")
plt.ylabel("$C_L$")
plt.title("Polar")
plt.show()

