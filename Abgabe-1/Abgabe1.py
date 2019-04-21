import numpy as np
import scipy.constants as const
import matplotlib
matplotlib.use('pgf')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'pgf.texsystem': 'lualatex',
    'pgf.preamble': r'\input{header-matplotlib.tex}',})

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
plt.tight_layout()
plt.savefig('build/A1_1.pdf')
plt.clf()

# Logarithmische Überprüfung des (positiven) Bereichs; ist das gefragt?
xplotlog=np.logspace(-8,4.615793,1000000)
plt.plot(xplotlog, np.abs(f(xplotlog)*3/2-1)*100,'b-')
plt.xlabel('x')
plt.ylabel(r'Fehler in $\%$')
plt.xscale('log')
plt.savefig('build/A1_12.pdf')
plt.clf()

#Überprüfung von g(x)
x_value_g=np.logspace(-6,1,100000)
for i in x_value_g:
    if np.abs(g(i))==0:
        ns_g = i
    if np.abs(g(i)*3/2-1)>0.01:
            x_min_g=i
print("Für x-Werte", x_min_g, "< x ist der Fehler der Formel g(x) kleiner als 1 Prozent")
print("Für Zahlen x <", ns_g, "ist g(x)=0")

xplotlog=np.logspace(np.log(x_min_g),np.log(x_max),1000000)
plt.plot(xplotlog, np.abs(g(xplotlog)*3/2-1)*100,'b-')
plt.xlabel('x')
plt.ylabel(r'Fehler in $\%$')
plt.xscale('log')
plt.savefig('build/A1_2.pdf')
plt.clf()

#Aufgabe 2


alpha=const.alpha
m_e=const.physical_constants["electron mass energy equivalent in MeV"][0]/1000
E_e=50 #Alle Energien in GeV
gamma=E_e/m_e
beta=np.sqrt(1-1/gamma**2)
s=(2*E_e)**2
def dsdO(theta):
    return alpha**2/s*(2+np.sin(theta)**2)/(1-beta**2*np.cos(theta)**2)
def dsdO_num(theta):
    return alpha**2/s*(2+np.sin(theta)**2)/(1/gamma**2*np.cos(theta)**2+np.sin(theta)**2)

pl_0=np.linspace(-np.pi/1800000,np.pi/1800000,1000) #Im Bereich um θ=0°
pl_pi_2=np.linspace(np.pi/2-np.pi/1800000,np.pi/2+np.pi/1800000,1000) #Im Bereich um θ=90°
pl_pi=np.linspace(np.pi-np.pi/1800000,np.pi+np.pi/1800000,1000) #Im Bereich um θ=180°

f1=plt.figure()
plt.subplot(121)
plt.plot(pl_0/2/np.pi*360000,dsdO(pl_0)*10**6,'b-')
plt.xlabel(r'$\Theta/10^{-3}\si{\degree}$')
plt.ylabel(r'$\frac{\mathrm{d}\sigma}{\mathrm{d}\Omega}/10^{-6}\si{\giga\eV^{-2}}$')
plt.subplot(122)
plt.plot(pl_0/2/np.pi*360000,dsdO_num(pl_0)*10**6,'r-')
plt.xlabel(r'$\Theta/10^{-3}\si{\degree}$')
plt.ylabel(r'$\frac{\mathrm{d}\sigma}{\mathrm{d}\Omega}/10^{-6}\si{\giga\eV^{-2}}$')
plt.tight_layout()
f1.savefig('build/plot1.pdf')

f2=plt.figure()
plt.subplot(121)
plt.plot(pl_pi_2/2/np.pi*360000,dsdO(pl_pi_2)*10**6,'b-')
plt.xlabel(r'$\Theta/10^{-3}\si{\degree}$')
plt.ylabel(r'$\frac{\mathrm{d}\sigma}{\mathrm{d}\Omega}/10^{-6}\si{\giga\eV^{-2}}$')
plt.subplot(122)
plt.plot(pl_pi_2/2/np.pi*360000,dsdO_num(pl_pi_2)*10**6,'r-')
plt.xlabel(r'$\Theta/10^{-3}\si{\degree}$')
plt.ylabel(r'$\frac{\mathrm{d}\sigma}{\mathrm{d}\Omega}/10^{-6}\si{\giga\eV^{-2}}$')
plt.tight_layout()
f2.savefig('build/plot2.pdf')

f3=plt.figure()
plt.subplot(121)
plt.plot(pl_pi/2/np.pi*360000,dsdO(pl_pi)*10**6,'b-')
plt.xlabel(r'$\Theta/10^{-3}\si{\degree}$')
plt.ylabel(r'$\frac{\mathrm{d}\sigma}{\mathrm{d}\Omega}/10^{-6}\si{\giga\eV^{-2}}$')
plt.subplot(122)
plt.plot(pl_pi/2/np.pi*360000,dsdO_num(pl_pi)*10**6,'r-')
plt.xlabel(r'$\Theta/10^{-3}\si{\degree}$')
plt.ylabel(r'$\frac{\mathrm{d}\sigma}{\mathrm{d}\Omega}/10^{-6}\si{\giga\eV^{-2}}$')
plt.tight_layout()
f3.savefig('build/plot3.pdf')

plt.clf()
#Konditionszahl K, x bezeichnet hier den Winkel theta
def K(x):
    return x * np.abs((2*np.sin(x)*np.cos(x)*(1-3*beta**2))/(2+np.sin(x)**2)/(1-beta**2*np.cos(x)**2))
theta_plot=np.linspace(0,np.pi,1000)
plt.plot(theta_plot/2/np.pi*360,K(theta_plot),'b-')
plt.xlabel(r'$\Theta/\si{\degree}$')
plt.ylabel(r'$K(\Theta)$')
plt.savefig('build/kondition.pdf')
