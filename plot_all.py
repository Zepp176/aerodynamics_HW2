from vortxl import vortxl
from functions import *
import numpy as np
import matplotlib.pyplot as plt

n = 128
m = 8

U = 10
alpha = np.pi/36
b = 1

ARs = [8, 10, 12, 15]
Ls = [0.2, 0.3, 0.4, 0.5]

for AR in ARs:
    for L in Ls:
        print(AR, L)
        
        c0 = b/AR * 2/(1+L)
                
        chord = lambda ksi : lin_tap(ksi, c0, L)
        x = get_flat_wing(chord, m, n, b)

        gamma = compute_circulation(x, m, n, U, alpha)
        circ = get_gamma_distribution(gamma, m, n)
        ksi = np.linspace(-1 + 1/n, 1 - 1/n, len(circ))
        
        f = open("results/{}_{:.0f}_Gamma.txt".format(AR, L*100), "w")
        for i in range(len(circ)//2, len(circ)):
            f.write("{} {}\n".format(ksi[i], circ[i]/(U*alpha*b)))
        f.close()