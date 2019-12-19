import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.optimize import curve_fit
from scipy.optimize import fmin
#from scipy import integrate
#import pandas as pd
import numpy.random as rand
#from scipy.stats import stats
import numpy.linalg as alg
#from sklearn import datasets
#from sklearn.datasets import make_blobs as m_b
#from sklearn.decomposition import PCA

#Aufgabe 35
print("Aufgabe 35:")
print()
def f(g,epsilon):
    return np.array([[epsilon-1,epsilon],[epsilon,epsilon-1]]) @ g/(2*epsilon-1)

def V(sigma_g,epsilon):
    cov=epsilon*(epsilon-1)*np.sum(sigma_g**2)
    sigma_1 = (epsilon-1)**2 * sigma_g[0]**2 + epsilon**2 * sigma_g[1]**2
    sigma_2 = (epsilon-1)**2 * sigma_g[1]**2 + epsilon**2 * sigma_g[0]**2
    V_ = np.array([[sigma_1,cov],[cov,sigma_2]])
    return V_/(2*epsilon-1)**2

def sigma(V):
    return np.sqrt(np.diag(V))

def rho(V):
    sigma_ = sigma(V)
    return V[0,1]/np.prod(sigma_)

#d)

def analysis(g1, g2, epsilon):
    g = np.array([g1,g2])
    print("f(",g,",",epsilon,"):")
    print(f(g, epsilon))
    V_ = V(np.sqrt(g), epsilon)
    print("V:")
    print(V_)
    print("\sigma_f:")
    print(sigma(V_))
    print("\\rho:")
    print(rho(V_))

analysis(200.0,169.0,0.1)
#e)
analysis(200.0,169.0,0.4)
#f)
analysis(200.0,169.0,0.5)

#Aufgabe 36
print()
print()
print("Aufgabe 36:")
print()
#a)
def create_A(epsilon, n=3):
    A_ = np.zeros((n,n))
    np.fill_diagonal(A_, 1 - 2 * epsilon)
    A_[0,0] = A_[n-1,n-1] = 1 - epsilon
    for i in range(1,n-1):
        A_[i,i-1] = A_[i,i+1] = A_[i-1,i] = A_[i+1,i] = epsilon
    return A_

#b)
epsilon = 0.23
A = create_A(epsilon,20)

f_wahr = [193,485,664,763,804,805,779,736,684,626,566,508,452,400,351,308,268,233,202,173]
g = A @ f_wahr
print("g = ")
print(g)
print()


#c)
lambda_A, U = alg.eig(A)
D = np.zeros(np.shape(A))
np.fill_diagonal(D, lambda_A)
U_inv = U.T
#print(A)
#A_ = U @ D @ U_inv
#A_[A_<0.01]=0
#print(A_)

def sort_values(lambdas):
    lambda_ = lambdas.copy()
    swap = np.arange(np.size(lambda_))
    for k in range(0,np.size(lambda_)):
        i = np.argmax(lambda_[k:])
        lambda_k = lambda_[k]
        lambda_[k] = lambda_[i + k]
        lambda_[i + k] = lambda_k
        swap_k = swap[k]
        swap[k] = swap[i + k]
        swap[i + k] = swap_k
    return swap, lambda_

swaps, sorted_lambda_A = sort_values(lambda_A)
#print("lambda_A")
#print(lambda_A)
#print(sorted_lambda_A)
#print(swaps)

def swap(A, swaps):
    A_ = A.copy()
    for i in range(0,np.size(swaps)):
        A_[:,i] = A[:,swaps[i]]
    return A_

D_sorted = np.zeros(np.shape(A))
np.fill_diagonal(D_sorted, sorted_lambda_A)
D_sorted_inv = 1/D_sorted
D_sorted_inv[np.isinf(D_sorted_inv)] = 0
U_sorted = swap(U, swaps)
U_sorted_inv = U_sorted.T

#A_ = U_sorted @ D_sorted @ U_sorted_inv
#A_[A_<0.01]=0
#print(A_-A<0.01)

#d)
rand.seed(230)
g_mess = rand.poisson(g,size=(1,20))
print("g_{mess} = ")
print(np.mean(g_mess, axis = 0))

b_wahr = U_sorted_inv @ f_wahr
b_mess = D_sorted_inv @ U_sorted_inv @ g_mess.T

c = U_sorted_inv @ g
c_mess = U_sorted_inv @ g_mess.T

V_g_mess = np.zeros((20,20))
np.fill_diagonal(V_g_mess, np.mean(g_mess, axis = 0))

V_c_mess = U_sorted_inv @ V_g_mess @ U_sorted

V_b_mess = D_sorted_inv @ V_c_mess @ D_sorted_inv

b_mess_normed =  b_mess.copy()
sigma_b_mess = np.sqrt(np.diag(V_b_mess))

for i in range(0,np.shape(b_mess_normed)[0]):
    b_mess_normed[i,:]/=sigma_b_mess[i]
#print(np.mean(b_mess,axis = 1))
#print(np.mean(b_mess_normed, axis = 1))
#print(b_wahr)

plt.plot(np.arange(20), np.mean(b_mess_normed, axis = 1), 'kx', label = r'$b_{mess}$')
plt.hlines([1,-1],colors='blue', xmin=0, xmax=20, linestyles='dashed')
plt.xlabel(r'Index $j$')
plt.ylabel(r'Koeffizient $b_j$')
plt.yscale('symlog')
plt.legend(loc='lower right')
plt.savefig('build/A36d.pdf')
plt.clf()

#e)
f_mess = U_sorted @ b_mess
b_mess_reg = b_mess.copy()
b_mess_reg[10:] = 0
f_mess_reg = U_sorted @ b_mess_reg
x_plot = np.linspace(0,2,20)
plt.plot(x_plot, np.mean(f_mess, axis = 1), 'kx', label = r'$f_{unregularisiert}$')
plt.plot(x_plot, np.mean(f_mess_reg, axis = 1), 'bx', label = r'$f_{regularisiert}$')
plt.plot(x_plot, f_wahr, 'g-', label = r'$f_{wahr}$')
plt.xlabel('x')
plt.ylabel(r'$f$')
plt.legend(loc='best')
plt.savefig('build/A36e.pdf')
plt.clf()