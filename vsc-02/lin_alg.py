import numpy as np
import math


# (a) Berechnen Sie den Winkel $\alpha$ in Grad zwischen den folgenden beiden Vektoren $a=[1.,1.77]$ und $b=[1.5,1.5]$
a = np.array([-1.,1.77])
b = np.array([1.5,1.5])
# YOUR CODE HERE
dot = a.dot(b)
mag_a = math.sqrt(a[0]**2 + a[1]**2)
mag_b = math.sqrt(b[0]**2 + b[1]**2)
alpha = math.degrees(np.arccos(dot/(mag_a*mag_b)))
print(alpha)


# (b) Gegeben ist die quadratische regulaere Matrix A und ein Ergbnisvektor b. Rechnen Sie unter Nutzung der Inversen die Loesung x des Gleichungssystems Ax = b aus.
A = np.array([[2,3,4],
              [3,-2,-1],
              [5,4,3]])
b = np.array([1.4,1.2,1.4])
x = np.linalg.inv(A).dot(b)
print(x)
# (c) Schreiben Sie eine Funktion die das Matrixprodukt berechnet. Nutzen Sie dafür nicht die Numpy Implementierung.
# Hinweis: Fangen Sie bitte mögliche falsche Eingabegroessen in der Funktion ab und werfen einen AssertionError
# assert Expression[, Arguments]

def matmult(M1, M2):
    assert M1.shape[1] == M2.shape[0]
    r_mat= np.empty((M1.shape[0],M2.shape[1]))
    for idx1, row in enumerate(M1):
        for idx2, col in enumerate(M2.T):
            r_mat[idx1, idx2] = np.sum(row*col)

    return r_mat

M1 = np.array([[1,2],[3,4],[5,6]])
M2 = np.array([[2,1],[0,2]])

M3 = np.arange(8).reshape(2,4)
M4 = np.arange(16).reshape(4,4)
M_res = matmult(M3, M4)
print(M_res)