import numpy as np
import matplotlib.pyplot as plt

ARs = [8, 10, 12, 15]
Ls  = [0.2, 0.3, 0.4, 0.5]

plt.figure(figsize=(15, 15))

for i, AR in enumerate(ARs):
    for j, L in enumerate(Ls):
        
        plt.subplot(4, 4, i+4*j+1)
        
        filename = "{}_{:.0f}_Gamma.txt".format(AR, L*100)
        
        data_v = np.loadtxt("verification/" + filename)
        data_r = np.loadtxt("results/" + filename)
        
        plt.grid()
        plt.plot(data_v[:,0], data_v[:,1])
        plt.plot(data_r[:,0], data_r[:,1])
        plt.legend(["Lifting line solution", "Our lifting surface solution"])
        plt.xlabel('$\\xi$')
        plt.ylabel('$\\Gamma/(U_{\\infty} b \\alpha)$')
        plt.title("AR = {}, $\\Lambda$ = {}".format(AR, L))
        plt.ylim(bottom=0.0, top=0.48)
        plt.xlim(left=0.0, right=1.0)

plt.tight_layout()
plt.savefig("figures/verification.png",dpi=300)