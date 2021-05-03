import numpy as np

eps32 = np.float32(1)
eps64 = np.float64(1)
print(np.float32(0.5))

i = 0

while True:
    i = i + 1
    if np.float32(1) + (0.5 * eps32) != np.float32(1):
        print(eps32)
        break
    else:
        eps32 = np.float32(0.5) * eps32