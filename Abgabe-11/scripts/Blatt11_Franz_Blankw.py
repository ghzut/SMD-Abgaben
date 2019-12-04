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


#Aufgabe 29
L_1 = np.array([0.13,1.5,0.5])
L_2 = np.array([2.0,0.5,0.05])
L_3 = np.array([0.07,0.5,1.3])
prior = np.array([0.8,0.1,0.1])

def posterior(L):
    return L*prior/np.sum(L*prior)

print("Aufgabe 29:")
print("Pion, Kaon, Proton")
print(posterior(L_1))
print(posterior(L_2))
print(posterior(L_3))

#Aufgabe 30

#Aufgabe 31
print("Aufgabe 31:")
Counts = np.array([4135,4202,4203,4218,4227,4231,4310])
Day = np.array([1,2,3,4,5,6,7])

#a)

Mean = np.mean(Counts)
print("Mittelwert konstantes Modell")
print(Mean)
def L(Mean, Counts):
    return_ = np.array(Mean)*0
    for count in Counts:
        fact = np.math.factorial(count)
        return_= np.append(return_, Mean**count/fact*np.exp(-Mean))
    return np.prod(return_)

#print(L(Mean, Counts))  #Zu großes N Fakultät

#b)
def logL(params):
    N_0 = params[0]
    A   = params[1]         #log(Counts!) weggelassen wegen zu hoher Fakultät und eh nur eine konstante, also unwichtig für maximumsbestimmung
    return -np.sum(Counts*np.log(N_0 + A*Day)-(N_0 + A*Day))

linModel = fmin(logL, [0,4000])
print("Parameter lineares Modell")
print(linModel)

plt.plot(Day, Counts, 'k.', label = 'Daten')
plt.axhline(y = Mean, label = 'Konstantes Modell', color = 'blue')
plt.plot(Day, linModel[0]+linModel[1]*Day, color = 'green', label = 'Lineares Modell')
plt.xlabel('Tag')
plt.ylabel('Counts')
plt.legend(loc='best')
plt.savefig('build/Ballon.pdf')
plt.clf()

#c)

Lambda=np.prod((Mean/(linModel[0]+linModel[1]*Day))**Counts*np.exp(-Mean+linModel[0]+linModel[1]*Day))
print("Xi^2 verteiltes -2ln(Lambda)")
print(-2*np.log(Lambda))

#d)
print("d)")
Counts = np.append(Counts, 4402)
Day = np.append(Day, 14)
Mean = np.mean(Counts)
print("Mittelwert konstantes Modell")
print(Mean)

linModel = fmin(logL, [0,4000])
print("Parameter lineares Modell")
print(linModel)

plt.plot(Day, Counts, 'k.', label = 'Daten')
plt.axhline(y = Mean, label = 'Konstantes Modell', color = 'blue')
plt.plot(Day, linModel[0]+linModel[1]*Day, color = 'green', label = 'Lineares Modell')
plt.xlabel('Tag')
plt.ylabel('Counts')
plt.legend(loc='best')
plt.savefig('build/Ballon_2.pdf')
plt.clf()

Lambda=np.prod((Mean/(linModel[0]+linModel[1]*Day))**Counts*np.exp(-Mean+linModel[0]+linModel[1]*Day))
print("Xi^2 verteiltes -2ln(Lambda)")
print(-2*np.log(Lambda))

#Aufgabe 32