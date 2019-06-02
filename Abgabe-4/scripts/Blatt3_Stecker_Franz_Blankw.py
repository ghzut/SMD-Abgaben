import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy import integrate
import pandas as pd
import numpy.random as rand
rand.seed(732)
#Aufgabe 1
#a)
gamma=2.7
Ereignisse=10**5
def Neutrinoenergie(dim=1):
    return (1-rand.random_sample(dim))**(1/(1-gamma))

df1=pd.DataFrame({'Energy':Neutrinoenergie(Ereignisse)})

#b)
def P(E):                      #Wahrscheinlichkeitsdichte ein Neutrino mit Energie E zu detektieren
    return (1-np.exp(-E/2))**3


Zufall_Akzeptanz=rand.random_sample(np.size(df1.Energy))

df1['AcceptanceMask']=np.greater(P(df1.Energy),Zufall_Akzeptanz)

plt.scatter(df1.Energy[np.invert(df1.AcceptanceMask)],Zufall_Akzeptanz[np.invert(df1.AcceptanceMask)], c='blue', label='Nicht detektiert', s=0.01)
plt.scatter(df1.Energy[df1.AcceptanceMask] ,Zufall_Akzeptanz[df1.AcceptanceMask], c='green', label='Detektiert', s=0.01)
#bins=10
#plt.hist(df1.get('Energy').get_values()[np.invert(df1.AcceptanceMask)], bins=bins, c='blue', label='Nicht detektierte Neutrinos')
#plt.hist(df1.get('Energy').get_values()[df1.AcceptanceMask].to_Matrix(), bins=bins, c='green', label='Detektierte Neutrinos')
#df1.hist('Energy')
Energie=np.linspace(df1.Energy.min(),df1.Energy.max(),1000)
plt.plot(Energie,P(Energie), c='black')
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$E/\si{\tera\electronvolt}$')
plt.ylabel('Likelyhood of Acceptance')
#plt.ylabel('Anzahl')
plt.legend(loc='best')
plt.savefig('build/Akzeptanz')
plt.clf()

#c)
def Normalverteilt_groessereins(mittel,sigma,dim=1):
    arr=np.arange(dim,dtype=float)*0                           #Startwert für alle Elemente = -1
    while(np.any(arr<1)):                          #Alles negative wird berschrieben
        size=np.size(arr[arr<1])                   #Rechnung nur noch für weitere negative Elemente
        u1=rand.random_sample(size)                #Polaralgorithmus
        u2=rand.random_sample(size)
        v1=2*u1-1
        v2=2*u2-1
        s=v1**2+v2**2
        akzeptierte_s=s<1                          #Verwerfung im Polaralgorithmus
        x1=v1[akzeptierte_s]*(-2/s[akzeptierte_s]*np.log(s[akzeptierte_s]))**(1/2)     #Normalverteilte (mittel=1, sigma=1) Zufallszahl
        x2=v2[akzeptierte_s]*(-2/s[akzeptierte_s]*np.log(s[akzeptierte_s]))**(1/2)     #Normalverteilte (mittel=1, sigma=1) Zufallszahl
        if(np.size(sigma)>1 and np.size(mittel)>1):
            x1=x1*(sigma[arr<1])[akzeptierte_s] + (mittel[arr<1])[akzeptierte_s]       #Normalverteilte mit gewünschtem mittel und sigma
            x2=x2*(sigma[arr<1])[akzeptierte_s] + (mittel[arr<1])[akzeptierte_s]       #Normalverteilte mit gewünschtem mittel und sigma
        if(np.size(sigma)==1 and np.size(mittel)==1):
            x1=x1*(sigma) + (mittel)                                                   #Normalverteilte mit gewünschtem mittel und sigma
            x2=x2*(sigma) + (mittel)                                                   #Normalverteilte mit gewünschtem mittel und sigma
        if(np.size(sigma)>1 and np.size(mittel)==1):
            x1=x1*(sigma[arr<1])[akzeptierte_s] + (mittel)                             #Normalverteilte mit gewünschtem mittel und sigma
            x2=x2*(sigma[arr<1])[akzeptierte_s] + (mittel)                             #Normalverteilte mit gewünschtem mittel und sigma
        if(np.size(sigma)==1 and np.size(mittel)>1):
            x1=x1*(sigma) + (mittel[arr<1])[akzeptierte_s]       #Normalverteilte mit gewünschtem mittel und sigma
            x2=x2*(sigma) + (mittel[arr<1])[akzeptierte_s]       #Normalverteilte mit gewünschtem mittel und sigma
        temp1=arr[arr<1]                           #Weise x1 zu
        temp1[akzeptierte_s]=x1
        arr[arr<1]=temp1
        #((arr[arr<1])[akzeptierte_s])[x1>1]=x1[x1>1]
        #((arr[arr<1])[akzeptierte_s])[x2>1]=x2[x2>1]
        #print(np.shape(x1[x1>1]))
        #print(np.shape(((arr[arr<1])[akzeptierte_s])[x1>1]))
        #print(((arr[arr<1])[akzeptierte_s])[x1>1]+x1[x1>1])
        #print(np.shape(((arr[arr<1])[akzeptierte_s])[x1>1]+x1[x1>1]))
        #print(arr)
    return arr

def Hits(Ereignis):
    return np.around(Normalverteilt_groessereins(10*Ereignis.get_values(),2*Ereignis.get_values(),np.size(Ereignis))) #Aufgerundet

df1['NumberOfHits']=Hits(df1.Energy)


#d)
def Ortsmessung(x_real,y_real,Ereignis,x_min=0,x_max=10,y_min=0,y_max=10):
    hits=Ereignis.get_values()
    sigma=1/(np.log10(hits+1))
    xy=np.arange(2*np.size(Ereignis),dtype=float).reshape(2,np.size(Ereignis))*0+x_max+y_max+1 #Rückgabewert so initialisieren, dass die Schleifen durchlaufen wird
    x_bool=np.logical_or(xy[0]>x_max,xy[0]<x_min)            #x in [x_min,x_max] als boolean-Array
    while(np.any(x_bool)):
        x=xy[0]
        x[x_bool]=rand.normal(x_real,sigma[x_bool])          #Berechne neue Zufalls-Werte
        xy[0]=x                                              #Uberschreibe Elemente
        x_bool=np.logical_or(xy[0]>x_max,xy[0]<x_min)        #update boolean-Array
    y_bool=np.logical_or(xy[1]>y_max,xy[1]<y_min)
    while(np.any(y_bool)):                                   #Analog zu oben
        y=xy[1]
        y[y_bool]=rand.normal(y_real,sigma[y_bool])
        xy[1]=y
        y_bool=np.logical_or(xy[1]>y_max,xy[1]<y_min)
    return xy
xy=Ortsmessung(7,3,df1.NumberOfHits)

df1['x']=xy[0]
df1['y']=xy[1]
plt.hist2d(xy[0],xy[1],bins=100)
plt.savefig('build/Orte_Signal.pdf')
plt.clf()


#e)
def Korrelierte_Norm(x_mittel,y_mittel,sigma_x,rho,dim=1,sigma_y=None,x_min=0,x_max=10,y_min=0,y_max=10):
    if sigma_y==None:sigma_y=sigma_x
    x=np.arange(dim,dtype=float)*0+x_max+y_max+1
    y=np.arange(dim,dtype=float)*0+x_max+y_max+1
    x_bool=np.logical_or(x>x_max,x<x_min)
    y_bool=np.logical_or(y>y_max,y<y_min)
    ges_bool=np.logical_or(x_bool,y_bool)
    while(np.any(ges_bool)):
        size=np.size(ges_bool[ges_bool])
        x1=x
        y1=y
        x1[ges_bool]=rand.normal(0,1,size)
        y1[ges_bool]=rand.normal(0,1,size)
        x[ges_bool]=(1-rho**2)**(1/2)*sigma_x*x1[ges_bool]+rho*sigma_y*y1[ges_bool]+x_mittel
        y[ges_bool]=sigma_y*y1[ges_bool]+y_mittel
        x_bool=np.logical_or(x>x_max,x<x_min)
        y_bool=np.logical_or(y>y_max,y<y_min)
        ges_bool=np.logical_or(x_bool,y_bool)
    return np.array([x,y])

Untergrund=10**7
xy_Untergrund=Korrelierte_Norm(5,5,3,0.5,dim=Untergrund)
NumberOfHits_Untergrund=np.around(10**(Normalverteilt_groessereins(3,1,dim=Untergrund)-1))
df2=pd.DataFrame({'NumberOfHits':NumberOfHits_Untergrund, 'x':xy_Untergrund[0],'y':xy_Untergrund[1]})

plt.hist2d(xy_Untergrund[0],xy_Untergrund[1],bins=100)
plt.savefig('build/Orte_Untergrund')
plt.clf()

plt.hist(np.log10(NumberOfHits_Untergrund),bins=100)
plt.savefig('build/Hits_Untergrund')
#print(df1.to_string())
#print(df2.to_string())
