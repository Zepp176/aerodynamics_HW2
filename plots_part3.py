from functions import *
import numpy as np
import matplotlib.pyplot as plt

# mesh
n = 200
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
betas = np.array([np.pi/3, np.pi/6, 0])

mesh = []
for i in range(3):
    mesh.append(get_winglets(lambda ksi : lin_tap(ksi, c0, L), m, n, b, r, betas[i]))

for i in range(3):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(mesh[i][0,:], mesh[i][1,:], mesh[i][2,:])
    set_axes_equal(ax)
    plt.show()

circ = []
for i in range(3):
    gamma = compute_circulation(mesh[i], m, n, U, alpha)
    circ.append(get_gamma_distribution(gamma, m, n))
    circ[i] /= U*b*alpha
    
    print("done ", i)

#%%

plt.figure(figsize=(7, 5))
for i in range(3):
    plt.plot(np.linspace(-r, r, len(circ[i])), circ[i])
plt.grid()
plt.xlabel("$\\xi$")
plt.ylabel("$\\frac{\\Gamma}{U_{\\infty} b \\alpha}$")
plt.legend(["$\\beta = \\pi/3$", "$\\beta = \\pi/6$", "$\\beta = 0$"])
plt.title("dimensionless circulation distribution for different $\\beta$ (AR = {}, $\\Lambda = {}$)".format(AR, L))
plt.tight_layout()
plt.savefig("figures/part3_comparaison.png", dpi=300)

#%% A0 and k

n = 50
m = 4

mesh = []
for i in range(3):
    mesh.append(get_winglets(lambda ksi : lin_tap(ksi, c0, L), m, n, b, r, betas[i]))

alphas = np.linspace(0, 10*np.pi/180, 50)
C_L = np.empty((3, len(alphas)))
C_D = np.empty((3, len(alphas)))

for i, alpha in enumerate(alphas):
    for j in range(3):
        gamma = compute_circulation(mesh[j], m, n, U, alpha)
        C_L[j, i], C_D[j, i] = get_coefs(get_F(mesh[j], gamma, m, n, rho, U, alpha), alpha, rho, S, U)

    print(i)

#%%

A0 = (C_L[:,-1] - C_L[:,0]) / alphas[-1]
k = np.mean(C_D[:,1:]/C_L[:,1:]**2, axis=1)

for i in range(3):
    print("{}. angle = {:.0f}°, A0 = {:.2f}, k = {:.4f}".format(i, betas[i]*180/np.pi, A0[i], k[i]))

plt.figure(figsize=(10, 5))

plt.subplot(121)
for i in range(3):
    plt.plot(alphas*180/np.pi, C_L[i,:])
plt.grid()
plt.xlabel("angle of attack [°]")
plt.ylabel("$C_L$")
plt.legend(["$\\beta = \\pi/3$", "$\\beta = \\pi/6$", "$\\beta = 0$"])

plt.subplot(122)
for i in range(3):
    plt.plot(alphas*180/np.pi, C_D[i,:])
plt.grid()
plt.xlabel("angle of attack [°]")
plt.ylabel("$C_D$")
plt.legend(["$\\beta = \\pi/3$", "$\\beta = \\pi/6$", "$\\beta = 0$"])

plt.tight_layout()
plt.savefig("figures/part3_comparaison_CL_CD.png", dpi=300)

#%%

n = 50
m = 4

betas = np.linspace(0, np.pi/2, 20)
alphas = np.linspace(0, 10*np.pi/180, 6)[1:]
k = np.empty(len(betas))

for i in range(len(betas)):
    mesh = get_winglets(lambda ksi : lin_tap(ksi, c0, L), m, n, b, r, betas[i])
    C_L = np.empty(len(alphas))
    C_D = np.empty(len(alphas))
    
    for j, alpha in enumerate(alphas):
        gamma = compute_circulation(mesh, m, n, U, alpha)
        C_L[j], C_D[j] = get_coefs(get_F(mesh, gamma, m, n, rho, U, alpha), alpha, rho, S, U)
        
        print(i, j)
    
    k[i] = np.mean(C_D[1:]/C_L[1:]**2)

#%%

plt.figure(figsize=(7, 5))
plt.plot(betas*180/np.pi, k)
plt.grid()
plt.xlabel("$\\beta$ [°]")
plt.ylabel("k")
plt.show()