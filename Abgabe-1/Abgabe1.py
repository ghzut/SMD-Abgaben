import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import matplotlib
matplotlib.use('TKAgg')

#Aufgabe 1
def f(x):
    return (x**3+1/3)-(x**3-1/3)
def g(x):
    return ((3+x**3/3)-(3-x**3/3))/x**3 #Hier sieht man schon das x=0 ausgeschlossen ist.

#Überprüfung des ganzzahligen Bereichs zur Bestimmung der Grenze
x_value = np.linspace(-175000,175000,350001)
for i in x_value:
    if np.abs(f(i))==0:
        ns = i
    if i<0:
        if np.abs(f(i)*3/2-1)>0.01:
            x_min=i+1
    if i>0:
        if np.abs(f(i)*3/2-1)>0.01:
            x_max=i-1
            print("Für ganzzahlige x-Werte im Bereich von ", x_min," bis ", x_max, "ist der Fehler der Formel f(x) kleiner als 1 Prozent")
            print("Für Zahlen x < ", ns, "und x > ", -ns, " ist f(x)=0")
            break
        elif i==175000:
            print("175000 ist keine Grenze")
xplot=np.linspace(int(x_min),int(x_max),int(x_max)-int(x_min)+1)
plt.plot(xplot, np.abs(f(xplot)*3/2-1)*100,'r.')
plt.xlabel('x')
plt.ylabel(r'Fehler in $\%$')

# Logarithmische Überprüfung des (positiven) Bereichs
xplotlog=np.logspace(-8,4.615793,1000000)
#print(np.abs(f(xplotlog)*3/2-1)*100)
plt.plot(xplotlog, np.abs(f(xplotlog)*3/2-1)*100,'b-')
plt.xlabel('x')
plt.ylabel(r'Fehler in $\%$')
plt.xscale('log')

#Überprüfung von g(x)
x_value_g=np.logspace(-6,1,100000)
for i in x_value_g:
    if np.abs(g(i))==0:
        ns_g = i
    if np.abs(g(i)*3/2-1)>0.01:
            x_min_g=i
print("Für x-Werte", x_min_g, "< x ist der Fehler der Formel g(x) kleiner als 1 Prozent")
print("Für Zahlen x <", ns_g, "ist g(x)=0")

#Aufgabe 2
alpha=const.alpha
m_e=const.physical_constants["electron mass energy equivalent in MeV"][0]
E_e=50000 #Alle Energien in MeV
gamma=E_e/m_e
beta=np.sqrt(1-1/gamma**2)
s=(2*E_e)**2
def dsdO(theta):
    return alpha**2/s*(2+np.sin(theta)**2)/(1-beta**2*np.cos(theta)**2)
def dsdO_num(theta):
    return alpha**2/s*(2+np.sin(theta)**2)/(1/gamma**2+beta**2*np.sin(theta)**2)

pl_0=np.linspace(-np.pi/1800000,np.pi/1800000,1000) #Im Bereich um theta=0°
pl_pi_2=np.linspace(np.pi/2-np.pi/1800000,np.pi/2+np.pi/1800000,1000) #Im Bereich um theta=90°
pl_pi=np.linspace(np.pi-np.pi/1800000,np.pi+np.pi/1800000,1000) #Im Bereich um theta=180°

#plt.plot(pl/2/np.pi*360,-dsdO(pl)*10**12+dsdO_num(pl)*10**12,'b-')
plt.subplot(321)
plt.plot(pl_0/2/np.pi*360000,dsdO(pl_0)*10**6,'b-')
#plt.xlabel(r'$\theta/\si{\milli\degree}$')
#plt.ylabel(r'$\frac{\mathrm{d}\sigma}{\mathrm{d}\Omega}/\si{\milli\meter\squared}$')
plt.subplot(322)
plt.plot(pl_0/2/np.pi*360000,dsdO_num(pl_0)*10**6,'r-')
#plt.xlabel(r'$\theta/\si{\milli\degree}$')
#plt.ylabel(r'$\frac{\mathrm{d}\sigma}{\mathrm{d}\Omega}/\si{\milli\meter\squared}$')
plt.subplot(323)
plt.plot(pl_pi_2/2/np.pi*360000,dsdO(pl_pi_2)*10**6,'b-')
#plt.xlabel(r'$\theta/\si{\milli\degree}$')
#plt.ylabel(r'$\frac{\mathrm{d}\sigma}{\mathrm{d}\Omega}/\si{\milli\meter\squared}$')
plt.subplot(324)
plt.plot(pl_pi_2/2/np.pi*360000,dsdO_num(pl_pi_2)*10**6,'r-')
#plt.xlabel(r'$\theta/\si{\milli\degree}$')
#plt.ylabel(r'$\frac{\mathrm{d}\sigma}{\mathrm{d}\Omega}/\si{\milli\meter\squared}$')
plt.subplot(325)
plt.plot(pl_pi/2/np.pi*360000,dsdO(pl_pi)*10**6,'b-')
#plt.xlabel(r'$\theta/\si{\milli\degree}$')
#plt.ylabel(r'$\frac{\mathrm{d}\sigma}{\mathrm{d}\Omega}/\si{\milli\meter\squared}$')
plt.subplot(326)
plt.plot(pl_pi/2/np.pi*360000,dsdO_num(pl_pi)*10**6,'r-')
plt.xlabel(r'$\theta/\si{\milli\degree}$')
plt.ylabel(r'$\frac{\mathrm{d}\sigma}{\mathrm{d}\Omega}/\si{\milli\meter\squared}$')
plt.savefig('plots.pdf')