from table2 import makeTable
from table2 import makeNewTable
from linregress import linregress
from customFormatting import *
from bereich import bereich
from weightedavgandsem import weighted_avg_and_sem
from weightedavgandsem import avg_and_sem
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp
import scipy.constants as const
from scipy import integrate
import pandas as pd
# BackwardsVNominal = []
# BackwardsVStd = []
# for value in BackwardsV:
#     BackwardsVNominal.append(unp.nominal_values(value))
#     BackwardsVStd.append(unp.std_devs(value))
# BackwardsVNominal = np.array(BackwardsVNominal)
# BackwardsVStd = np.array(BackwardsVStd)

# einfacher:
# BackwardsVNominal = unp.nominal_values(BackwardsV)
# BackwardsVStd = unp.std_devs(BackwardsV)

# makeTable([Gaenge, ForwardsVNominal, ForwardsVStd, ], r'{Gang} & \multicolumn{2}{c}{$v_\text{v}/\si[per-mode=reciprocal]{\centi\meter\per\second}$} & ', 'name', ['S[table-format=2.0]', 'S[table-format=2.3]', ' @{${}\pm{}$} S[table-format=1.3]', ], ["%2.0f", "%2.3f", "%2.3f",])

#[per-mode=reciprocal],[table-format=2.3,table-figures-uncertainty=1]

# unp.uarray(np.mean(), stats.sem())
# unp.uarray(*avg_and_sem(values)))
# unp.uarray(*weighted_avg_and_sem(unp.nominal_values(bneuDiff), 1/unp.std_devs(bneuDiff)))

# plt.cla()
# plt.clf()
# plt.plot(ForwardsVNominal*100, DeltaVForwardsNominal, 'gx', label='Daten mit Bewegungsrichtung aufs Mikrofon zu')
# plt.plot(BackwardsVNominal*100, DeltaVBackwardsNominal, 'rx', label='Daten mit Bewegungsrichtung vom Mikrofon weg')
# plt.ylim(0, line(t[-1], *params)+0.1)
# plt.xlim(0, t[-1]*100)
# plt.xlabel(r'$v/\si{\centi\meter\per\second}$')
# plt.ylabel(r'$\Delta f / \si{\hertz}$')
# plt.legend(loc='best')
# plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
# plt.savefig('build/'+'VgegenDeltaV')

# a = unp.uarray(params[0], np.sqrt(covar[0][0]))
# params = unp.uarray(params, np.sqrt(np.diag(covar)))
# makeNewTable([convert((r'$c_\text{1}$',r'$c_\text{2}$',r'$T_{\text{A}1}$',r'$T_{\text{A}2}$',r'$\alpha$',r'$D_1$',r'$D_2$',r'$A_1$',r'$A_2$',r'$A_3$',r'$A_4$'),strFormat),convert(np.array([paramsGes2[0],paramsGes1[0],deltat2*10**6,deltat1*10**6,-paramsDaempfung[0]*2,4.48*10**-6 *paramsGes1[0]/2*10**3, 7.26*10**-6 *paramsGes1[0]/2*10**3, (VierteMessung-2*deltat2*10**6)[0]*10**-6 *1410 /2*10**3, unp.uarray((VierteMessung[1]-VierteMessung[0])*10**-6 *1410 /2*10**3, 0), unp.uarray((VierteMessung[2]-VierteMessung[1])*10**-6 *2500 /2*10**3, 0),unp.uarray((VierteMessung[3]-VierteMessung[2])*10**-6 *1410 /2*10**3, 0)]),unpFormat,[[r'\meter\per\second',"",True],[r'\meter\per\second',"",True],[r'\micro\second',"",True],[r'\micro\second',"",True],[r'\per\meter',"",True],[r'\milli\meter',"",True],[r'\milli\meter',"",True],[r'\milli\meter',"",True],[r'\milli\meter',r'1.3f',True],[r'\milli\meter',r'1.3f',True],[r'\milli\meter',r'2.2f',True]]),convert(np.array([2730,2730]),floatFormat,[r'\meter\per\second','1.0f',True])+convert((r'-',r'-'),strFormat)+convert(unp.uarray([57,6.05,9.9],[2.5,0,0]),unpFormat,[[r'\per\meter',"",True],[r'\milli\meter',r'1.2f',True],[r'\milli\meter',r'1.2f',True]])+convert((r'-',r'-',r'-',r'-'),strFormat),convert(np.array([(2730-paramsGes2[0])/2730*100,(2730-paramsGes1[0])/2730*100]),unpFormat,[r'\percent','',True])+convert((r'-',r'-'),strFormat)+convert(np.array([(-paramsDaempfung[0]*2-unp.uarray(57,2.5))/unp.uarray(57,2.5)*100,(4.48*10**-6 *paramsGes1[0]/2*10**3-6.05)/6.05*100, (-7.26*10**-6 *paramsGes1[0]/2*10**3+9.90)/9.90*100]),unpFormat,[r'\percent','',True])+convert((r'-',r'-',r'-',r'-'),strFormat)],r'{Wert}&{gemessen}&{Literaturwert\cite{cAcryl},\cite{alphaAcryl}}&{Abweichung}','Ergebnisse', ['c ','c',r'c','c'])

#Aufgabe 1
import numpy.random as random  #gegebene zufällige gleichverteilung von 0 bis 1 ist rand.random_sample()
random.seed(431)
#a)
def Von_x_bis_y(x_min,x_max,dim=1):
    return (x_max-x_min) * random.random_sample(dim) + x_min
#random=Von_x_bis_y(0,3,10)
#print(random)

#b)
def Exponential(tau,dim=1):
    return -tau*np.log(1-random.random_sample(dim))

#random_exp=Exponential(10,100)
#print(random_exp)

#c)
def Potenz(n,x_min,x_max,dim=1):
    assert n>=2
    return x_min / (1-random.random_sample(dim)* (1-(x_min/x_max)**(n+1)))**(1/(n+1))

#random_potenz=Potenz(2,1,10,100)
#print(random_potenz)

#d)

def Cauchy(dim=1):
    return np.tan((random.random_sample(dim)-1/2)*const.pi)

#random_cauchy=Cauchy(100)
#print(random_cauchy)

#e)
x,y=np.genfromtxt('empirisches_histogramm.csv', delimiter=',', unpack=True,skip_header=1)

def Intg(xx):
    bereich=(x<=xx)
    return integrate.trapz(y[bereich],x=x[bereich])
y=y/Intg(1)         #Normierung
Y=y*0
for i in range(0,y.size):
    Y[i]=Intg(x[i])        #Werte-Array des Integrals (kriegen wir aufgrund von Dimensionsproblemen nicht anders hin)
#print(Y)

def Empirisch():                #Ergibt die Zufalls-Zahlen
    Ran=random.random_sample(1)
    return (x[Y<=Ran])[-1]

dim_emp=100

random_emp=np.arange(dim_emp)*0.0
for i in range(0,dim_emp):
    random_emp[i]=Empirisch()        # Zufalls-Zahlen (kriegen wir aufgrund von Dimensionsproblemen nicht anders hin)

print(random_emp)


#Aufgabe 2 mit viel for oder mit weniger
def LKG(a,b,m,x,anzahl):
    random=[None]
    for i in range(anzahl*m):
        n_1=((a*x+b)%m)
        random[i]=n_1/m
        random+=[None]
        x=n_1
    random=random[:-1]
    return random
'''
def period(rand):
    for i in range(len(rand)):
        for j in range(len(rand)-i):
            if j!=0 and rand[i+j]==rand[i]:
                for k in range(len(rand)-i-j):
                    if rand[i+k]!=rand[i+j+k]:
                        break
                    elif k==len(rand)-i-j-1:
                        return j
'''
def period(rand):
    rand_ar=np.array(rand)
    index=np.where(rand_ar==rand_ar[0])
    if len(index[0])<2:
        return 0
    else:
        ix=index[0][1]
        for i in range(ix):
            if rand_ar[i+ix]!=rand_ar[i]:
                break
            elif i==ix-1:
                return ix
#a)
b=3
m=1024
anzahl=2
x_0=0

for a in range(50):
    plt.plot(a,period(LKG(a,b,m,x_0,anzahl)),'b.',rasterized=True)
    plt.xlabel("a")
    plt.ylabel("Periodendauer")
plt.savefig("build/Periodendauer.pdf")

a=1601
b=3456
m=10000

#b)
plt.clf()
plt.cla()
for x_val in range(4):
    if x_val < 2:
        plt.subplot(2,2,x_val+1)
        plt.hist(LKG(a,b,m,x_val,1),density=True,bins=25)
        plt.xlabel("Zufallszahl")
        plt.ylabel("Wahrscheinlichkeit")
    else:
        plt.subplot(2,2,x_val+1)
        plt.hist(LKG(a,b,m,x_val,1),density=True,bins=25)
        plt.xlabel("Zufallszahl")
        plt.ylabel("Wahrscheinlichkeit")
plt.savefig("build/Wahrschkeit.pdf")

#c) Plot ist noch hässlich überlappend
def pair(x,i):
    j = i % len(x)
    y=x[j:] + x[:j]
    return y

x=LKG(a,b,m,x_0,1)
y=pair(x,1)
z=pair(x,2)
plt.clf()
plt.cla()
plt.scatter(x, y)
plt.xlabel(r'$x_{i}$')
plt.ylabel(r'$x_{i+1}$')
plt.savefig("build/2D-Scatter-Plot.pdf")

#plt.clf()
#plt.cla()

f=plt.figure()
ax=Axes3D(f)
ax.scatter(x, y, z)
ax.set_xlabel(r'$x_{i}$')
ax.set_ylabel(r'$x_{i+1}$')
ax.set_zlabel(r'$x_{i+2}$')
plt.show()

x=[None]
for i in range(10000):
    x_val=np.random.uniform()
    x[i]=x_val
    x+=[None]

#Histogramm np.random.uniform()
for i in range(4):
    if i < 2:
        plt.subplot(2,2,i+1)
        plt.hist(LKG(a,b,m,x[i],1),density=True,bins=25)
        plt.xlabel("Zufallszahl")
        plt.ylabel("Wahrscheinlichkeit")
    else:
        plt.subplot(2,2,i+1)
        plt.hist(LKG(a,b,m,x[i],1),density=True,bins=25)
        plt.xlabel("Zufallszahl")
        plt.ylabel("Wahrscheinlichkeit")
plt.savefig("build/Wahrschkeit_uniform.pdf")


#2D-Scatter np.radnom.uniform()
y=pair(x,1)
z=pair(x,2)
plt.clf()
plt.cla()
plt.scatter(x, y)
plt.xlabel(r'$x_{i}$')
plt.ylabel(r'$x_{i+1}$')
plt.savefig("build/2D-Scatter-Plot-Uniform.pdf")

plt.clf()
plt.cla()
#3D-Scatter np.random.uniform()
f2=plt.figure()
ax2=Axes3D(f2)
ax2.scatter(x, y, z)
ax2.set_xlabel(r'$x_{i}$')
ax2.set_ylabel(r'$x_{i+1}$')
ax2.set_zlabel(r'$x_{i+2}$')
plt.savefig("build/3D-Scatter-Plot-Uniform.pdf")

#Ganzzahlige Überprüfung der x-Startwerte ob der LKG 1/2 liefert
m=4
x_data=LKG(a,b,m,x_0,1)
b=3
a=5
m=1024
for j in x_data:
    x_neu = LKG(a,b,m,j,1)
    counter=0
    for i in range(len(x_neu)):
        if x_neu[i]==1/2:
            counter += 1
    print("Für einen ganzzahligen Startwert liefert der LKG ",counter,"mal den Wert 1/2") #ergibt 0 logisch da nur ganzzahlig

#Nichtganzzahlige Überprüfung
x_data_dezimal=np.random.uniform(size=4)
for j in x_data_dezimal:
    x_neu = LKG(a,b,m,j,1)
    counter=0
    for i in range(len(x_neu)):
        if x_neu[i]==1/2:
            counter += 1
    print("Für einen nicht ganzzahligen Startwert liefert der LKG ",counter,"mal den Wert 1/2")#sollte laut Lars 16 liefern tut es aber nicht ->nochmal angucken, mit Startwert x_0=0.5 geht es auch nicht 


#Aufgabe 4
sx0=3.5
ux0=0
sy0=2.6
uy0=3
rho=0.9
cov_0=[[sx0**2,rho*sx0*sy0],[rho*sx0*sy0,sy0**2]]
mean_0=(ux0,uy0)

P0x,P0y=random.multivariate_normal(mean_0,cov_0,10000).T

a=-0.5
b=0.6
ux1=6
sx1=3.5
corxy1=np.sign(b)
uy1=a*ux1+b
sy1=1
cov_1=[[sx1**2,corxy1],[corxy1,sy1**2]]
mean_1=(ux1,uy1)

P1x,P1y=random.multivariate_normal(mean_1,cov_1,10000).T

plt.clf()

plt.scatter(P0x,P0y,lw=0,s=1,label=r'$P_0$')
plt.scatter(P1x,P1y,lw=0,s=1,label=r'$P_1$',color='green')
plt.legend(loc='best')
plt.savefig('build/Aufgabe_4_a.pdf')

Mittel_Stich_0=np.array([np.mean(P0x),np.mean(P0y)])
Varianz_Stich_0=np.array([np.mean((P0x-Mittel_Stich_0[0])**2),np.mean((P0y-Mittel_Stich_0[1])**2)])**(1/2)
Vxy0=np.mean(P0x*P0y)-Mittel_Stich_0[0]*Mittel_Stich_0[1]
Cov_Stich_0=[[Varianz_Stich_0[0]**2,Vxy0],[Vxy0,Varianz_Stich_0[1]]]
Korrelationskoeffizient_0=Vxy0/(Varianz_Stich_0[0]*Varianz_Stich_0[1])
print("P0")
print(Mittel_Stich_0)
print(Varianz_Stich_0)
print(Vxy0)
print(Cov_Stich_0)
print(Korrelationskoeffizient_0)

Mittel_Stich_1=np.array([np.mean(P1x),np.mean(P1y)])
Varianz_Stich_1=np.array([np.mean((P1x-Mittel_Stich_1[0])**2),np.mean((P1y-Mittel_Stich_1[1])**2)])**(1/2)
Vxy1=np.mean(P1x*P1y)-Mittel_Stich_1[0]*Mittel_Stich_1[1]
Cov_Stich_1=[[Varianz_Stich_1[0]**2,Vxy1],[Vxy1,Varianz_Stich_1[1]]]
Korrelationskoeffizient_1=Vxy1/(Varianz_Stich_1[0]*Varianz_Stich_1[1])
print("P1")
print(Mittel_Stich_1)
print(Varianz_Stich_1)
print(Vxy1)
print(Cov_Stich_1)
print(Korrelationskoeffizient_1)

P_Ges_x=np.append(P0x,P1x)
P_Ges_y=np.append(P0y,P1y)

Mittel_Stich_Ges=np.array([np.mean(P_Ges_x),np.mean(P_Ges_y)])
Varianz_Stich_Ges=np.array([np.mean((P_Ges_x-Mittel_Stich_Ges[0])**2),np.mean((P_Ges_y-Mittel_Stich_Ges[1])**2)])**(1/2)
VxyGes=np.mean(P_Ges_x*P_Ges_y)-Mittel_Stich_Ges[0]*Mittel_Stich_Ges[1]
Cov_Stich_Ges=[[Varianz_Stich_Ges[0]**2,VxyGes],[VxyGes,Varianz_Stich_Ges[1]]]
Korrelationskoeffizient_Ges=VxyGes/(Varianz_Stich_Ges[0]*Varianz_Stich_Ges[1])
print("P_Ges")
print(Mittel_Stich_Ges)
print(Varianz_Stich_Ges)
print(VxyGes)
print(Cov_Stich_Ges)
print(Korrelationskoeffizient_Ges)

#c)
df0= pd.DataFrame({'P_0_x':P0x,'P_0_y':P0y})
df0.to_hdf('build/Populationen.h5',key='P0',mode='w')
df1= pd.DataFrame({'P_1_x':P1x,'P_1_y':P1y})
df1.to_hdf('build/Populationen.h5',key='P1')
dfGes= pd.DataFrame({'P_Ges_x':P_Ges_x,'P_Ges_y':P_Ges_y})
dfGes.to_hdf('build/Populationen.h5',key='PGes')
