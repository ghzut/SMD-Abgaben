import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
from uncertainties import ufloat
from scipy.stats import stats
np.random.seed(42)
#b)+c)
x_0 = 30
s = 2
n = 10**5
random_numbers = np.array([x_0])
N = 15/(np.pi**4)
x_i = x_0
def Planck(x):
    return N*x**3/(np.exp(x)-1)

for i in range (n - 1):
    if x_i < s: #Da Planck nur für x > 0 definiert
        x_j = np.random.uniform(0,x_i + s)
    else:
        x_j = np.random.uniform(x_i - s, x_i + s)
    p = np.min([1, Planck(x_j)/Planck(x_i)])
    rnd = np.random.uniform()
    if rnd <= p:
        x_i = x_j
    random_numbers = np.append(random_numbers, x_i)
x_plot = np.linspace(np.min(random_numbers),np.max(random_numbers),4242)
plt.hist(random_numbers,bins=100,density=True, label = 'Histogramm Planck-Verteilung')
plt.plot(x_plot,Planck(x_plot), label = 'Planck-Verteilung')#, rasterized = True)
plt.legend(loc='best')
plt.xlabel('Zufallszahl')
plt.ylabel('Wahrscheinlichkeit')
plt.tight_layout()
plt.savefig('build/Planck_Metropolis.pdf')
#d)
plt.clf()
it=np.arange(1, n + 1)
plt.plot(it,random_numbers,'b.', markersize = 0.1)#, alpha = 0.7, rasterized = True)
plt.axhline(y = x_plot[np.argmax(Planck(x_plot))], color = 'y', label = 'Maximum der Planck-Verteilung')
plt.xlabel('Iteration')
plt.ylabel('Zufallszahl')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('build/Iteration.pdf')
plt.plot(it[0:250],random_numbers[0:250],'b-', markersize = 0.1)#, alpha = 0.7, rasterized = True)
plt.axhline(y = x_plot[np.argmax(Planck(x_plot))], color = 'y', label = 'Maximum der Planck-Verteilung')
plt.xlabel('Iteration')
plt.ylabel('Zufallszahl')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('build/Iteration_Burn_In.pdf')
