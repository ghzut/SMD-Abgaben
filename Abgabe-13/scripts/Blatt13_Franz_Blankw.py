import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
#from scipy.optimize import curve_fit
#from scipy.optimize import fmin
#from scipy import integrate
import pandas as pd
import numpy.random as rand
#from scipy.stats import stats
import numpy.linalg as alg
#from sklearn import datasets
#from sklearn.datasets import make_blobs as m_b
#from sklearn.decomposition import PCA

#Aufgabe 37
print("Aufgabe 37:")
print()

#a)
data = pd.read_hdf('unfolding_data.hdf5', key='train')

observable = data.drop('energy_true', axis = 1)
energy = data['energy_true']
n = observable.count(axis='columns')[0]
for i in range(0,n):
    plt.subplot(n,2,(i+1))
    plt.hist2d(energy, observable[observable.columns[i]],bins=20)
    plt.title(observable.columns[i])
plt.tight_layout(rect=(0,-1,1,1))

plt.savefig('build/Attribute.pdf')
plt.clf()

#b)
Attribute = data.drop(['cog_y', 'cog_x', 'cov_x', 'cov_y', 'leakage'], axis = 1)

size_bins = np.linspace(120,500,25)
energy_bins = np.linspace(15,200,17)

binned_attribute = Attribute.copy()
binned_attribute['size'] = pd.cut(binned_attribute['size'], size_bins, include_lowest = True)
binned_attribute['energy_true'] = pd.cut(binned_attribute.energy_true, energy_bins, include_lowest = True)

#print(Attribute.head())
#print(binned_attribute.head())

binned_attribute.dropna(inplace=True)
binned_attribute_np = (binned_attribute.to_numpy())
size = binned_attribute_np[:,1]
energy_true = binned_attribute_np[:,0]

A_pd = pd.crosstab(size, energy_true, normalize = True)
A = A_pd.to_numpy(dtype=float)

plt.imshow(A)
plt.xlabel('energy_true')
plt.ylabel('size')

plt.tight_layout()
plt.savefig('build/A.pdf')
plt.clf()

#c)

#d)
def Hesse(f, g, tau):
    n = np.size(f)
    m = np.size(g)
    ret = np.zeros([n,n])
    for k in range(0,n):
        for l in range(0,n):
            ret[l,l] += tau
            for i in range(0,m):
                ret[k,l] += A[i,k]*g[i]*A[i,l]/(np.sum(A[i,:]*f))**2
    return ret

def grad(f, g, tau):
    n = np.size(f)
    m = np.size(g)
    ret = np.zeros(n)
    ret+=tau*f
    for k in range(0,n):
        for i in range(0,m):
            ret[k] += A[i,k]*(1-g[i]/np.sum(A[i,:]*f))
    return ret

#e)

def newton(f_k, g, tau, it=0):
    f_k1 = f_k - alg.inv(Hesse(f_k, g, tau)+10**(-6)*np.identity(np.size(f_k))) @ (grad(f_k, g, tau))
    #print(f_k)
    it+=1
    if alg.norm(f_k1-f_k)<10**(-10) or it>500:
        print(it)
        return f_k1
    else :
        return newton(f_k1, g, tau, it)

#f)
data = pd.read_hdf('unfolding_data.hdf5', key='test')

Attribute = data.drop(['cog_y', 'cog_x', 'cov_x', 'cov_y', 'leakage'], axis = 1)

binned_attribute = Attribute.copy()
binned_attribute['size'] = pd.cut(binned_attribute['size'], size_bins, include_lowest = True)
binned_attribute['energy_true'] = pd.cut(binned_attribute.energy_true, energy_bins, include_lowest = True)

binned_attribute.dropna(inplace=True)

A_pd = pd.crosstab(size, energy_true)
Crosstable = A_pd.to_numpy(dtype=float)

f_true = np.sum(Crosstable, axis = 0)
g  = np.sum(Crosstable, axis = 1)

print('f)')
print('g')
print(g)
print('f_true')
print(f_true)

taus = [10**(-6), 10**(-3), 0.1]

print('f_newton')
plt.hlines(1,linestyle='-', color='black',xmin = np.min(energy_bins)+(200-15)/16, xmax = np.max(energy_bins[:-1])+(200-15)/16)
for tau in taus:
    f_newton = newton(f_true+100, g, tau)
    print('tau = ', tau)
    print('f = ', np.around(f_newton))
    plt.plot(energy_bins[:-1]+(200-15)/16, f_newton/f_true, label = tau)
#plt.plot(energy_bins[:-1]+(200-15)/16, f_true, label='f_true')
plt.yscale('log')
plt.legend(loc='best')
plt.xlabel('Energy')
plt.ylabel(r'$\frac{\text{Count}}{\text{True Count}}$')
plt.tight_layout()
plt.savefig('build/entfaltet.pdf')