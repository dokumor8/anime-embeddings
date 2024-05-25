import numpy as np

a = np.array([[1, 2, 3]])
b = np.array([[1, 0, 3]])
c = np.array([[1, 2, 0]])

d = [a, b, c]
print(d)
d_m = np.mean(d, axis=0)
print(d_m)
