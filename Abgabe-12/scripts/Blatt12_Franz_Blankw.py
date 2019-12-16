import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.optimize import curve_fit
from scipy.optimize import fmin
#from scipy import integrate
#import pandas as pd
import numpy.random as rand
#from scipy.stats import stats
#import numpy.linalg as alg
#from sklearn import datasets
#from sklearn.datasets import make_blobs as m_b
#from sklearn.decomposition import PCA

#Aufgabe 35

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
    V_ = V(g, epsilon)
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