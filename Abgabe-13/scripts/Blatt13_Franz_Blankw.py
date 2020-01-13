import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
#from scipy.optimize import curve_fit
#from scipy.optimize import fmin
#from scipy import integrate
import pandas as pd
import numpy.random as rand
#from scipy.stats import stats
#import numpy.linalg as alg
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

A_pd = pd.crosstab(size, energy_true)
A = A_pd.to_numpy()
A=A/np.sum(A)

plt.imshow(A)
plt.xlabel('energy_true')
plt.ylabel('size')

plt.tight_layout()
plt.savefig('build/A.pdf')
plt.clf()

#c)
