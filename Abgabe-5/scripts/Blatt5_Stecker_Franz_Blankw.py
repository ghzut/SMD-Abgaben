import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy import integrate
import pandas as pd
import numpy.random as rand
from scipy.stats import stats
import numpy.linalg as alg
from sklearn import datasets
from sklearn.datasets import make_blobs as m_b
from sklearn.decomposition import PCA

#Aufgabe 12
df_P_0 = pd.read_hdf("zwei_populationen.h5", "P_0_10000")
df_P_1 = pd.read_hdf("zwei_populationen.h5", "P_1")

P_0=df_P_0.to_numpy()
P_1=df_P_1.to_numpy()
#P_01=np.append(P_1,P_0,axis=0)
#plt.scatter(P_0[:,0],P_0[:,1],label=r'$P_0$')
#plt.legend(loc='best')
#plt.savefig('build/plot1.pdf')
#plt.clf()
#plt.scatter(P_1[:,0],P_1[:,1],label=r'$P_1$')
#plt.legend(loc='best')
#plt.savefig('build/plot2.pdf')
#plt.clf()
#plt.scatter(P_0[:,0],P_0[:,1],label=r'$P_0$')
#plt.scatter(P_1[:,0],P_1[:,1],label=r'$P_1$')
#plt.legend(loc='best')
#plt.savefig('build/plot3.pdf')
#a
mu_0=P_0.mean(axis=0)
mu_1=P_1.mean(axis=0)
print("mu_0=")
print(mu_0)
print("mu_1=")
print(mu_1)
#mu_01=P_01.mean(axis=0)
#b
def Covariance_2D(x,mu):
    V=np.zeros([2,2])
    sigma=np.mean((x-mu)**2,axis=0)
    Var_xy=np.mean((x[:,0]-mu[0])*(x[:,1]-mu[1]),axis=0)
    V[0,0]=sigma[0]
    V[1,1]=sigma[1]
    V[0,1]=V[1,0]=Var_xy
    return V

V_0=Covariance_2D(P_0,mu_0)
V_1=Covariance_2D(P_1,mu_1)
#V_01=Covariance_2D(P_01,mu_01)
V_01=V_0+V_1

print("V_0")
print(V_0)
print("V_1")
print(V_1)
print("V_01")
print(V_01)
#c
lambda_=-alg.inv(V_01) @ (mu_0 - mu_1)
e_lambda=lambda_/alg.norm(lambda_)  #wird zur x-Achse mittels Drehung
print("e_lambda")
print(e_lambda)
#d
#z_proj=np.array([1,-lambda_[0]/lambda_[1]])
#e_z=z_proj/alg.norm(z_proj)         #wird zur y-Achse mittels Drehung

#M=np.append([e_lambda],[e_z],axis=0) #Drehmatrix
#P_0_proj=M@(P_0.T)
#P_1_proj=M@(P_1.T)
P_0_proj=e_lambda@(P_0.T)
P_1_proj=e_lambda@(P_1.T)
plt.hist(P_0_proj,bins=20,alpha=0.5,label=r'$P_0$')
plt.hist(P_1_proj,bins=20,alpha=0.5,label=r'$P_1$')
plt.legend(loc='best')
plt.xlabel(r'$\lambda$')
plt.savefig('build/Aufgabe_2_d.pdf')
plt.clf()

#e
def recall(l,Signal,left=True):             #Effizienz
    if left==False:l=-l
    Rec=np.array(np.size(Signal[Signal<l[0]]),dtype=float)
    for i in range(1,np.size(l)):
        Rec=np.append(Rec,np.size(Signal[Signal<l[i]]))
    return Rec/np.size(Signal)

def precision(l,Signal,Underground,left=True):          #Reinheit return NaN where no Signal or Underground
    if left==False:l=-l
    x=(np.size(Signal[Signal<l[0]])+np.size(Underground[Underground<l[0]]))
    Prec=np.zeros(1,dtype=float)
    if x==0:Prec[0]='NaN'
    else:Prec[0]=np.size(Signal[Signal<l[0]])/x
    for i in range(1,np.size(l)):
        x=(np.size(Signal[Signal<l[i]])+np.size(Underground[Underground<l[i]]))
        if x==0:Prec=np.append(Prec,'NaN')
        else:Prec=np.append(Prec,np.size(Signal[Signal<l[i]])/x)
    return Prec

lambda_cut=np.linspace(np.min(P_0_proj),np.max(P_1_proj),1000)
plt.plot(lambda_cut,recall(lambda_cut,P_0_proj),label='Effizienz')
plt.plot(lambda_cut,precision(lambda_cut,P_0_proj,P_1_proj),label='Reinheit')
plt.xlabel(r'$\lambda$')
plt.legend(loc='best')
plt.savefig('build/Aufgabe_2_e.pdf')
plt.clf()

#f             durch 0 teilen und ähnliches macht diesen Aufgabenteil nicht ohne modifikation möglich meiner meinung nach...
#def SB(l,Signal,Underground,left=True):          #size of Signal/Underground at specific l  Original (Sieht bescheuert aus)
#    if left==False:l=-l
#    x=np.size(Underground[Underground<l[0]])
#    S_B=np.zeros(1,dtype=float)
#    if x==0:S_B[0]='inf'
#    else:S_B[0]=np.size(Signal[Signal<l[0]])/x
#    for i in range(1,np.size(l)):
#        x=np.size(Underground[Underground<l[i]])
#        if x==0:S_B=np.append(S_B,'inf')
#        else:S_B=np.append(S_B,np.size(Signal[Signal<l[i]])/x)
#    return S_B
def SB(l,Signal,Underground,left=True):          #size of Signal/Underground at specific l   Abgewandelte Formel
    if left==False:l=-l
    S_B=np.zeros(1,dtype=float)
    S_B[0]=(np.size(Signal[Signal<l[0]])+1)/(np.size(Underground[Underground<l[0]])+1)
    for i in range(1,np.size(l)):
        S_B=np.append(S_B,(1+np.size(Signal[Signal<l[i]]))/(1+np.size(Underground[Underground<l[i]])))
    return S_B

y=SB(lambda_cut,P_0_proj,P_1_proj)
plt.plot(lambda_cut,y)
lambda_SB_max=lambda_cut[np.argmax(y)]
print("\lambda_{S/B_{max}}")
print(lambda_SB_max)
#plt.vlines(lambda_SB_max,0,np.max(y))
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$\frac{Signal}{Untergrund}$')
#plt.legend(loc='best')
plt.savefig('build/Aufgabe_2_f.pdf')
plt.clf()

#g
def significance(l,Signal,Underground,left=True):           #abgewandelte Formel wegen durch null teilen
    if left==False:l=-l
    S_B=np.zeros(1,dtype=float)
    S_B[0]=(np.size(Signal[Signal<l[0]])+1)/(np.size(Signal[Signal<l[0]])+np.size(Underground[Underground<l[0]])+1)**(1/2)
    for i in range(1,np.size(l)):
        S_B=np.append(S_B,(1+np.size(Signal[Signal<l[i]]))/(np.size(Signal[Signal<l[i]])+np.size(Underground[Underground<l[i]])+1)**(1/2))
    return S_B

y=significance(lambda_cut,P_0_proj,P_1_proj)
plt.plot(lambda_cut,y)
lambda_S_max=lambda_cut[np.argmax(y)]
print("\lambda_{Signifikanz_{max}}")
print(lambda_S_max)
#plt.vlines(lambda_SB_max,0,np.max(y))
plt.xlabel(r'$\lambda$')
plt.ylabel(r'Signifikanz')
#plt.legend(loc='best')
plt.savefig('build/Aufgabe_2_g.pdf')
plt.clf()

#h
#Strg+c;Strg+v und löschen redundanten Codes
df_P_0 = pd.read_hdf("zwei_populationen.h5", "P_0_1000")

P_0=df_P_0.to_numpy()
mu_0=P_0.mean(axis=0)
print("mu_0=")
print(mu_0)
#b

V_0=Covariance_2D(P_0,mu_0)
V_01=V_0+V_1
print("V_0")
print(V_0)
print("V_01")
print(V_01)

#c
lambda_=-alg.inv(V_01) @ (mu_0 - mu_1)
e_lambda=lambda_/alg.norm(lambda_)  #wird zur x-Achse mittels Drehung
print("e_lambda")
print(e_lambda)
#d
P_0_proj=e_lambda@(P_0.T)
P_1_proj=e_lambda@(P_1.T)
plt.hist(P_0_proj,bins=20,alpha=0.5,label=r'$P_0$')
plt.hist(P_1_proj,bins=20,alpha=0.5,label=r'$P_1$')
plt.legend(loc='best')
plt.xlabel(r'$\lambda$')
plt.savefig('build/Aufgabe_2_hd.pdf')
plt.clf()

#e
lambda_cut=np.linspace(np.min(P_0_proj),np.max(P_1_proj),1000)
plt.plot(lambda_cut,recall(lambda_cut,P_0_proj),label='Effizienz')
plt.plot(lambda_cut,precision(lambda_cut,P_0_proj,P_1_proj),label='Reinheit')
plt.xlabel(r'$\lambda$')
plt.legend(loc='best')
plt.savefig('build/Aufgabe_2_he.pdf')
plt.clf()

#f             durch 0 teilen und ähnliches macht diesen Aufgabenteil nicht ohne modifikation möglich meiner meinung nach...
y=SB(lambda_cut,P_0_proj,P_1_proj)
plt.plot(lambda_cut,y)
lambda_SB_max=lambda_cut[np.argmax(y)]
print("\lambda_{S/B_{max}}")
print(lambda_SB_max)
#plt.vlines(lambda_SB_max,0,np.max(y))
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$\frac{Signal}{Untergrund}$')
#plt.legend(loc='best')
plt.savefig('build/Aufgabe_2_hf.pdf')
plt.clf()

#g
y=significance(lambda_cut,P_0_proj,P_1_proj)
plt.plot(lambda_cut,y)
lambda_S_max=lambda_cut[np.argmax(y)]
print("\lambda_{Signifikanz_{max}}")
print(lambda_S_max)
#plt.vlines(lambda_SB_max,0,np.max(y))
plt.xlabel(r'$\lambda$')
plt.ylabel(r'Signifikanz')
#plt.legend(loc='best')
plt.savefig('build/Aufgabe_2_hg.pdf')
plt.clf()

#Nr.13
X, y = m_b(n_samples=1000, centers=2, n_features=4,
random_state=0)
X = X - X.mean(axis=0)
plt.scatter(X[:,1], X[:,2], c=y, s=40, edgecolor="gr")
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.savefig('build/scatter.pdf')
plt.clf()
pca = PCA(n_components = 4)
pca.fit(X)
X_1 = pca.transform(X)
c = pca.get_covariance()
EW = alg.eigvals(c)
print(EW)

plt.clf()
#meins
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.hist(X[:,i], color = "k", density = True, rasterized = True, bins = 40, histtype ='step')
    plt.hist(X_1[:,i], color = "r", density = True, rasterized = True, bins = 40, histtype ='step')
    plt.ylabel(r'$Wahrscheinlichkeit$')
    if(i==0):
        plt.xlabel(r'$1. Dimension$')
    if(i==1):
        plt.xlabel(r'$2. Dimension$')
    if(i==2):
        plt.xlabel(r'$3. Dimension$')
    if(i==3):
        plt.xlabel(r'$4. Dimension$')
    plt.tight_layout()
plt.savefig('build/hists_meine.pdf')
#Lars
plt.clf()
plt.subplot(2, 2, 1)
plt.hist(X[:,0],color='r', density=True, histtype ='step', bins=40, label='1. Dimension', alpha=0.7)
plt.ylabel('Wahrscheinlichkeit')
plt.xlabel('1. Dimension')
plt.tight_layout()
plt.subplot(2, 2, 2)
plt.hist(X[:,1],color='y', density=True, histtype ='step', bins=40, label='2. Dimension', alpha=0.7)
plt.ylabel('Wahrscheinlichkeit')
plt.xlabel('2. Dimension')
plt.tight_layout()
plt.subplot(2, 2, 3)
plt.hist(X[:,2],color='b', density=True, histtype ='step', bins=40, label='3. Dimension', alpha=0.7)
plt.ylabel('Wahrscheinlichkeit')
plt.xlabel('3. Dimension')
plt.tight_layout()
plt.subplot(2, 2, 4)
plt.hist(X[:,3],color='g', density=True, histtype ='step', bins=40, label='4. Dimension', alpha=0.7)
plt.ylabel('Wahrscheinlichkeit')
plt.xlabel('4. Dimension')
plt.tight_layout()
plt.subplot(2, 2, 1)
plt.hist(X_1[:,0],color='k', density=True, histtype ='step', bins=40, label='1. Dimension', alpha=0.7)
plt.ylabel('Wahrscheinlichkeit')
plt.xlabel('1. Dimension')
plt.tight_layout()
plt.subplot(2, 2, 2)
plt.hist(X_1[:,1],color='k', density=True, histtype ='step', bins=40, label='2. Dimension', alpha=0.7)
plt.ylabel('Wahrscheinlichkeit')
plt.xlabel('2. Dimension')
plt.tight_layout()
plt.subplot(2, 2, 3)
plt.hist(X_1[:,2],color='k', density=True, histtype ='step', bins=40, label='3. Dimension', alpha=0.7)
plt.ylabel('Wahrscheinlichkeit')
plt.xlabel('3. Dimension')
plt.tight_layout()
plt.subplot(2, 2, 4)
plt.hist(X_1[:,3],color='k', density=True, histtype ='step', bins=40, label='4. Dimension', alpha=0.7)
plt.ylabel('Wahrscheinlichkeit')
plt.xlabel('4. Dimension')
plt.tight_layout()
plt.savefig('build/hists_Lars.pdf')


#Jan Lukas
x1, x2, x3, x4 = zip(*X_1)
x_array = [x1, x2, x3, x4]
i=0
for count in x_array:
    i += 1
    xfirst = count * y
    xfirst = xfirst[xfirst != 0]
    xsec = count * (1-y)
    xsec = xsec[xsec != 0]

    plt.subplot(2,2,i)
    plt.hist(xfirst, density = True, bins = 40, histtype ='step', color = 'k')
    plt.hist(xsec, density = True, bins = 40, histtype ='step', color = 'r')
plt.savefig('build/hists_Jan_Lukas.pdf')
