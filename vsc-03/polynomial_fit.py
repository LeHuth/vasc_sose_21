import numpy as np
import matplotlib.pyplot as plt

# Laden der gegebenen Daten d0 - d4
y = np.load('./data/d4.npy')
x = np.linspace(-2,2,200)
#print(y)
# Implementieren Sie ein Funktion, die gegeben den x-Werten und dem Funktiongrad
# die Matrix A aufstellt.


def f(x, the_other_x, degree):
    return the_other_x[0]*(x**degree) + the_other_x[1]


def MAE(m,the_other_x, degree):
    E = 0
    for i in range(m):
        E = E + np.abs((f(x[i], the_other_x, degree) - y[i]))
    return 1/m * E



def create_matrix(x, degree):
    ...
    A = np.zeros((len(x), degree+1))
    for index in range(len(x)):
        temp = np.array([])
        for j in range(degree, -1, -1):
            temp = np.append(temp, x[index] ** j)
        #print(temp.shape, A[index].shape)
        A[index] = temp#np.array([x[index], 1])


    #print(A)

    Atb = A.T @ y
    AtA = A.T @ A
    the_other_x =  np.linalg.solve(AtA,Atb)
    #print(the_other_x)
    #plt.plot(x,f(x,the_other_x, degree))
    #print(MAE(len(x), the_other_x, degree))
    return MAE(len(x), the_other_x, degree), the_other_x
# Lösen Sie das lineare Ausgleichsproblem
# Hinweis: Nutzen Sie bitte hier nicht np.linalg.lstsq!, sondern implementieren sie A^T A x = A^T b selbst

# Stellen Sie die Funktion mit Hilfe der ermittelten Koeffizienten mit matplotlib
# np.poly1d

plt.plot(x,y,'ro')
least_error_mae = 0
best_coefficient_mae = np.array([])
polynomial_mae = 0
for d in range(1,20):
    mae, coefficient = create_matrix(x, d)
    if d == 1:
        least_error_mae = mae
        best_coefficient_mae = coefficient
        polynomial_mae = d
    if mae < least_error_mae:
        best_coefficient_mae = coefficient
        polynomial_mae = d
        least_error_mae = mae

print(least_error_mae, polynomial_mae)
print(best_coefficient_mae, polynomial_mae)
plt.plot(x,f(x, best_coefficient_mae, polynomial_mae),label=f"coefficients: {best_coefficient_mae},\n polynomial: {polynomial_mae}, least MAE: {least_error_mae}")
plt.legend(loc='upper center', prop={'size': 8})
plt.show()