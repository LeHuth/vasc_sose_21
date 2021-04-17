import numpy as np
from math import sqrt, fabs, pi,pow
import matplotlib.pyplot as plt

# Aufgabe 1 (a)
# i ist hier die Anzahl der Iterationen
# In jeder Iteration soll ein epsilon auf 1.0 addiert werden und mit der
# Floating-Point Darstellung von np.float64(1) bzw. np.float(32) verglichen werden.
# Starten Sie dabei mit Epsilon=1.0 und halbieren Sie den Wert in jeder Iteration (wie an der Ausgabe 2^(-i) zu sehen)
# Stoppen Sie die Iterationen, wenn np.float32(1) + epsi != np.float32(1) ist.
# Hinweis: Ja - in diesem Fall dürfen Sie Floating-Point Werte vergleichen ;)

eps64 = np.float64(1)
j = np.float64(0)

# Print Anweisung vor dem Loop
print("64bit floating point")
print('j | 2^(-j) | 1 + 2^(-j) ')
print('----------------------------------------')

while np.float64(1) + np.float64(2) ** (-j) != np.float64(1):
    eps64 = np.float64(2) ** (-j)
    j += np.float64(1)

# Print Anweisung in / nach dem Loop
print('{0:4.0f} | {1:16.8e} | ungleich 1'.format(j, eps64))

print("32bit floating point")
print('i | 2^(-i) | 1 + 2^(-i) ')
print('----------------------------------------')

eps32 = np.float32(1)
i = np.float32(0)

while np.float32(1) + np.float32(2) ** (-i) != np.float32(1):
    eps32 = np.float32(2) ** (-i)
    i += np.float32(1)


# Print Anweisung in / nach dem Loop
print('{0:4.0f} | {1:16.8e} | ungleich 1'.format(i, eps32))


# Aufgabe 1 (b)
# Werten Sie 30 Iterationen aus und speichern Sie den Fehler in einem
# Fehlerarray err
N = 30
err = []
# sqrt(2) kann vorberechnet werden
sn = np.sqrt(2)
for n in range(2, N):
    # 1. Umfang u berechnen
    # 2. Fehler en berechnen und in err speichern
    # Fehler ausgeben print('{0:2d}\t{1:1.20f}\t{2:1.20e}'.format(n, u, en))
    # YOUR CODE HERE

    s2n = sqrt(2-sqrt(4-(sn)**2))
    sn = s2n
    u = sn* 2*(2**n)
    diff = (pi * 2) - u
    err.append(abs(diff))
    print('{0:2d}\t{1:1.20f}\t{2:1.20e}'.format(n, u, diff))
# Plotten Sie den Fehler
plt.figure(figsize=(6.0, 4.0))
plt.semilogy(range(2, N), err, 'rx')
plt.xlim(2, N - 1)
plt.ylim(1e-16, 10)
print("--------------------------------------------")
# Aufgabe 1 (c)
# Löschen des Arrays und wir fangen mit der Berechnung von vorn an.
# Nur diesmal mit der leicht veranderten Variante
err = []
sn1 = np.sqrt(2)
for n in range(2, N):
    s2n_alt = sn1 / sqrt(2+sqrt(4-sn1**2))
    sn1 = s2n_alt
    u = sn1 * 2 * (2 ** n)
    diff1 = (pi * 2) - u
    err.append(abs(diff1))
    print('{0:2d}\t{1:1.20f}\t{2:1.20e}'.format(n, u, diff1))

plt.figure(figsize=(6.0, 4.0))
plt.semilogy(range(2, N), err, 'rx')
plt.xlim(2, N - 1)
plt.ylim(1e-16, 10)
plt.show()



