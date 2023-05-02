import numpy as np

a = np.array([0, 1, 2, 3, 4])
x, y, z, w, v = np.meshgrid(a, a, a, a, a)

combinations = np.vstack((x.flatten(), y.flatten(), z.flatten(), w.flatten(), v.flatten())).T
print(combinations)

a = 3
b = 6
print(str(ab))