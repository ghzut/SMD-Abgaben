import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.optimize import curve_fit
#from scipy import integrate
#import pandas as pd
import numpy.random as rand
#from scipy.stats import stats
#import numpy.linalg as alg
#from sklearn import datasets
#from sklearn.datasets import make_blobs as m_b
#from sklearn.decomposition import PCA

#Aufgabe 21
rand.seed(7212) #Anmerkung: Interessanter Seed: 721
def a(dim=1):
    return rand.multivariate_normal([1.0,1.0], [[0.04,-0.8*0.04],[-0.8*0.04,0.04]], dim).T


def sigma_y(x,a_):
    stdy = [np.std(a_[:, 0] + a_[:, 1] * x_i) for x_i in x]  
    return stdy
dimension=1000
a_=a(dimension)
plt.scatter(a_[0,:] , a_[1,:])
plt.xlabel(r'$a_0$')
plt.ylabel(r'$a_1$')
plt.savefig("build/Aufgabe_21_b_1")
plt.clf()

def exakt(x):
    return 0.2*(1-1.6*x+x**2)**(1/2)
x_plot=np.linspace(-20,20,dimension)
s_y=sigma_y(x_plot,a_)
plt.plot(x_plot,s_y,'k-', label="Numerisch")
plt.plot(x_plot,exakt(x_plot),'b-', label="Exakt")
plt.xlabel('x')
plt.ylabel(r'$\sigma_y$')
plt.legend(loc="best")


plt.savefig("build/Aufgabe_21_b_2.pdf")
plt.clf()

#Aufgabe 22

#Aufgabe 23

#Aufgabe 24