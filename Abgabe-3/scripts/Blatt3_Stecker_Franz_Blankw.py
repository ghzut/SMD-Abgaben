﻿from table2 import makeTable
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
    return np.tan(random.random_sample(dim)*const.pi)

#random_cauchy=Cauchy(100)
#print(random_cauchy)

#e)
x,y=np.genfromtxt('empirisches_histogramm.csv', delimiter=',', unpack=True)

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
for i in range(1,dim_emp):
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
        print('Für a=',a,'existiert keine Periode')
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
