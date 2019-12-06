import numpy as np
import math
import matplotlib.pyplot as plt
import csv
from scipy.sparse import diags
from scipy.stats import sem

#27a+d)
def loglike(x):
    return 3*x-30*np.log(x)+np.log(float(math.factorial(8)*math.factorial(9)*math.factorial(13)))

def tayloglike(x):
    return 30*(1-np.log(10))+np.log(float(math.factorial(8)*math.factorial(9)*math.factorial(13)))+0.3*(x-10)**2

x=np.linspace(0.001,20,1000)
plt.plot(x, loglike(x), 'r-', label='negative Log-Likelihood')
plt.plot(x, tayloglike(x), 'b-', label='Taylor 2.Ordnung')
plt.plot(x, 30*(1-np.log(10))+np.log(float(math.factorial(8)*math.factorial(9)*math.factorial(13)))+0*x,'g-',label='-ln(L_max)')
plt.legend(loc='best')
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$-\log(L)$')
plt.tight_layout()
plt.savefig('build/Likelihood.pdf')

#28)
def poly(x,a,b,c,d,e,f,g):
    return a*x**6+b*x**5+c*x**4+d*x**3+e*x**2+f*x+g
def MatA(x):
    f1 = np.ones(len(x))
    f2 = f1*x
    f3 = f2*x
    f4 = f3*x
    f5 = f4*x
    f6 = f5*x
    f7 = f6*x
    A = np.vstack((f1,f2,f3,f4,f5,f6,f7)).T
    return A
def readexcel(filename,part):
    x = []
    y = []
    with open(filename, 'rt') as f:
        reader = csv.reader(f)
        for row in reader:
            x.append(row[0])
            if part=='a':
                y.append(row[1])
            else:
                y.append(row[1:51])
    x = x[1:]
    x = np.asarray(x)
    x = x.astype(np.float)
    if part=='a':
        y = y[1:]
        y = np.asarray(y)
        y = y.astype(np.float)
        return x, y
    else:
        y = np.delete(y,(0), axis=0)
        y = np.asarray(y)
        y = y.astype(np.float)
        sigma = np.zeros(len(x))
        newy = np.zeros(len(x))
        for i in range(0,y.shape[0]):
            newy[i] = np.mean(y[i])
            sigma[i] = sem(y[i])
        return x, newy, sigma

#a)
x,y = readexcel('aufg_a.csv','a')
s = len(x)
A = MatA(x)
koeff = np.dot(np.linalg.inv(A.T@A)@A.T, y)

print('Die bestimmten Koeffizienten a-g sind:')
print(koeff[::-1])
xplot = np.linspace(0,8)
plt.clf()
plt.plot(x,y,'rx', label='Messwerte')
plt.plot(xplot,poly(xplot,*koeff[::-1]),'b-',label='Gefittetes Polynom')
plt.legend(loc='best')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.tight_layout()
plt.savefig('build/polyfit.pdf')

#b)
lambdaa = np.array([0.1,0.3,0.7,3,10])
C = -2*np.ones(s)
C[0] = C[s-1] = -1
dia = [C, np.ones(s-1), np.ones(s-1)]
C = diags(dia, [0,1,-1]).toarray()
T = C@A
koeff_new = []
for i in lambdaa:
    koeff_new.append(np.dot(np.linalg.inv(A.T@A+i*T.T@T)@A.T, y))
koeff_new = np.asarray(koeff_new)

print('Die mit verschiedenen Regularisierungen bestimmten Koeffizienten a-g sind:')
print('lambda=0.1:')
print(koeff_new[0][::-1])
print('lambda=0.3:')
print(koeff_new[1][::-1])
print('lambda=0.7:')
print(koeff_new[2][::-1])
print('lambda=3:')
print(koeff_new[3][::-1])
print('lambda=10:')
print(koeff_new[4][::-1])
plt.clf()
plt.plot(x, y, 'rx', label='Messwerte')
plt.plot(xplot, poly(xplot, *koeff_new[0][::-1]), label=r'$\lambda=0{,}1$')
plt.plot(xplot, poly(xplot, *koeff_new[1][::-1]), label=r'$\lambda=0{,}3$')
plt.plot(xplot, poly(xplot, *koeff_new[2][::-1]), label=r'$\lambda=0{,}7$')
plt.plot(xplot, poly(xplot, *koeff_new[3][::-1]), label=r'$\lambda=3$')
plt.plot(xplot, poly(xplot, *koeff_new[4][::-1]), label=r'$\lambda=10$')
plt.legend(loc='best')
plt.ylim(0.025,0.2)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.tight_layout()
plt.savefig('build/lambdas.pdf')
#c)
x,y,sigma = readexcel('aufg_c.csv','c')
W = np.diag(1/sigma**2*np.ones(s))
a = np.dot(np.linalg.inv(A.T@W@A)@A.T@W, y)
print('Die bestimmten Koeffizienten a-g sind:')
print(a[::-1])

plt.clf()
plt.plot(x, y, 'rx', label='Messwerte')
plt.plot(xplot, poly(xplot, *a[::-1]), label='Weighted-Squares-Fit')
plt.legend(loc='best')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.tight_layout()
plt.savefig('build/WSquared.pdf')
