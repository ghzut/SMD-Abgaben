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

def error(func):
    return np.abs(func * 3.0/2.0 - 1)

def f(x):
    return (x**3+1.0/3.0)-(x**3-1.0/3.0)
def g(x):
    return ((3.0+x**3/3.0)-(3-x**3/3.0))/x**3 #Hier sieht man schon das x=0 ausgeschlossen ist -> logarithmische Darstellung sinnvoll.

#Überprüfung des ganzzahligen Bereichs zur Bestimmung der Grenze
x_value = np.logspace(3.0,9.0, 1000)
f_x=f(x_value)
error_f=error(f_x)
x_max=x_value[error_f-0.01>0][0]
print('Für |x|<',x_max,'ist der Fehler von f(x) kleiner als 1%')
ns=x_value[f_x==0][0]
print('Bei |x|>',ns,'ist f(x)=0')
'''
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
plt.plot(xplot, np.abs(f(xplot)*3/2-1)*100,rasterized=True)
plt.xlabel('x')
plt.ylabel(r'Fehler in $\%$')
plt.tight_layout()
plt.savefig('build/A1_1.pdf')
plt.clf()

# Logarithmische Überprüfung des (positiven) Bereichs;
'''
plt.plot(x_value, error_f*100,'b-',rasterized=True)
plt.xlabel('x')
plt.ylabel(r'Fehler in $\%$')
plt.xscale('log')
plt.savefig('build/A1_12.pdf')
plt.clf()

#Überprüfung von g(x)
x_value_g=np.logspace(-8,1,2000)
g_x=g(x_value_g)
error_g=error(g_x)
x_max_g=x_value_g[error_g-0.01<=0][0]
print('Für x < ',x_max_g,'ist der Fehler von g(x) kleiner als 1%')
ns_g=x_value_g[g_x!=0][0]
print('Bei x < ',ns_g,'ist g(x)=0')
'''
for i in x_value_g:
    if np.abs(g(i))==0:
        ns_g = i
    if np.abs(g(i)*3/2-1)>0.01:
            x_min_g=i


print("Für x-Werte", x_min_g, "< x ist der Fehler der Formel g(x) kleiner als 1 Prozent")
print("Für Zahlen x <", ns_g, "ist g(x)=0")

xplotlog=np.logspace(np.log(x_min_g),np.log(x_max),1000)
'''

plt.plot(x_value_g, error_g * 100, rasterized=True)
plt.xlabel('x')
plt.ylabel(r'Fehler in $\%$')
plt.xscale('log')
plt.savefig('build/A1_2.pdf')
plt.clf()

#Aufgabe 2


alpha=const.alpha
m_e=const.physical_constants["electron mass energy equivalent in MeV"][0]/1000 #Alle Energien in GeV
E_e=50 #Alle Energien in GeV
gamma=E_e/m_e
beta=np.sqrt(1-1/gamma**2)
s=(2*E_e)**2
def dsdO(theta):
    return alpha**2/s*(2+np.sin(theta)**2)/(1-beta**2*np.cos(theta)**2)
def dsdO_num(theta):
    return alpha**2/s*(2+np.sin(theta)**2)/(1/gamma**2*np.cos(theta)**2+np.sin(theta)**2)

pl=np.linspace(0,2*np.pi,1000)
pl_0=np.linspace(-np.pi/180000000,np.pi/180000000,2000) #Im Bereich um θ=0°
pl_pi_2=np.linspace(np.pi/2-np.pi/180000000,np.pi/2+np.pi/180000000,2000) #Im Bereich um θ=90°
pl_pi=np.linspace(np.pi-np.pi/180000000,np.pi+np.pi/180000000,2000) #Im Bereich um θ=180°

f1=plt.figure()
plt.plot(pl_0/2/np.pi*360000000,dsdO(pl_0),rasterized=True)
plt.plot(pl_0/2/np.pi*360000000,dsdO_num(pl_0),'r-',rasterized=True)
plt.xlabel(r'$\Theta/10^{-8}\si{\degree}$')
plt.ylabel(r'$\frac{\mathrm{d}\sigma}{\mathrm{d}\Omega}/\si{\giga\eV^{-2}}$')
plt.tight_layout()
f1.savefig('build/plot1.pdf')

f2=plt.figure()
plt.plot(pl_pi_2/2/np.pi*360000000,dsdO(pl_pi_2),'b-',rasterized=True)
plt.plot(pl_pi_2/2/np.pi*360000000,dsdO_num(pl_pi_2),'r-',rasterized=True)
plt.xlabel(r'$\Theta/10^{-8}\si{\degree}$')
plt.ylabel(r'$\frac{\mathrm{d}\sigma}{\mathrm{d}\Omega}/\si{\giga\eV^{-2}}$')
plt.tight_layout()
f2.savefig('build/plot2.pdf')

f3=plt.figure()
plt.plot(pl_pi/2/np.pi*360000000,dsdO(pl_pi),'b-',rasterized=True)
plt.plot(pl_pi/2/np.pi*360000000,dsdO_num(pl_pi),'r-',rasterized=True)
plt.xlabel(r'$\Theta/10^{-8}\si{\degree}$')
plt.ylabel(r'$\frac{\mathrm{d}\sigma}{\mathrm{d}\Omega}/\si{\giga\eV^{-2}}$')
plt.tight_layout()
f3.savefig('build/plot3.pdf')

f4=plt.figure()
plt.plot(pl/2/np.pi*360,dsdO_num(pl)-dsdO(pl),'b-',rasterized=True)
plt.xlabel(r'$\Theta/\si{\degree}$')
plt.ylabel(r'$\frac{\mathrm{d}\sigma}{\mathrm{d}\Omega}_\text{stab}-\frac{\mathrm{d}\sigma}{\mathrm{d}\Omega}/\si{\giga\eV^{-2}}$')
plt.tight_layout()
f4.savefig('build/Differenz.pdf')
plt.clf()
#Konditionszahl K, x bezeichnet hier den Winkel theta
def K(x):
    return np.abs(x * (2 * np.sin(x) * np.cos(x) * (1 - 3 * beta**2))/(2 + np.sin(x)**2)/(1 - beta**2 * np.cos(x)**2))
theta_plot=np.linspace(0,np.pi,1000)
plt.plot(theta_plot/2/np.pi*360,K(theta_plot),'b-')
plt.xlabel(r'$\Theta/\si{\degree}$')
plt.ylabel(r'$K(\Theta)$')
plt.yscale('log')
plt.savefig('build/kondition.pdf')
