from vortxl import vortxl
import numpy as np

x = np.array([1, 0, 0])
x1 = np.array([0, -1, 0])
x2 = np.array([0, 1, 0])
print(vortxl(x, x1, x2))