import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
from uncertainties import ufloat
from scipy.stats import stats
#import time
#start_time = time.clock()

#Nr.17)
temperature, weather, humidity, wind, soccer = np.genfromtxt('data.txt', unpack=True)
makeTable([temperature, weather, humidity, wind, soccer], r'{'+r'$Temperature/\si{\degree\Celsius}$'+r'} & {'+r'$Weather$'+r'} & {'+r'$Humidity/\si{\percent}$'+r'} & {'+r'$Wind$'+r'} & {'+r'$Soccer$'+r'}','tabData',['S[table-format=2.1]', 'S[table-format=1.0]', 'S[table-format=2.0]', 'S[table-format=1.0]', 'S[table-format=1.0]'],["%2.1f", "%1.0f", "%2.0f", "%1.0f", "%1.0f"])

def entropy(p1, p2):
    if p1 == 0:
        return - p2 * np.log2(p2)
    elif p2 == 0:
        return - p1 * np.log2(p1)
    else:
        return - p1 * np.log2(p1) - p2 * np.log2(p2)

p_soc_true = len(soccer[soccer==1])/len(soccer)
p_soc_false = 1 - p_soc_true
S = entropy(p_soc_true, p_soc_false)

def information_gain(att, cut):
    IG = np.zeros(len(cut))
    for i in range(len(cut)):
        under_cut = len(att[att < cut[i]])
        over_cut = len(att) - under_cut
        p_over = over_cut/len(att)
        p_under = under_cut/len(att)

        if over_cut == 0 or under_cut == 0:
            IG[i] = 0
        else:
            soc_false_under_cut = 0
            ind = np.where([att < cut[i]])[1] #Frage an Gruppenleiter: Wieso ist np.where() soviel (Faktor 3) langsamer als selbst eine nornale for-loop?
            add = len(soccer[ind][soccer[ind] == 0])
            soc_false_under_cut += add
            p_soc_false_under = soc_false_under_cut/under_cut
            p_soc_true_under = 1 - p_soc_false_under
            S_under= entropy(p_soc_true_under, p_soc_false_under)

            soc_false_over_cut = 0
            ind = np.where([att >= cut[i]])[1]
            add = len(soccer[ind][soccer[ind] == 0])
            soc_false_over_cut += add
            p_soc_false_over = soc_false_over_cut/over_cut
            p_soc_true_over = 1 - p_soc_false_over
            S_over = entropy(p_soc_true_over, p_soc_false_over)

            IG[i] = S - p_under * S_under - p_over * S_over
    return IG

weath_cut_array = np.linspace(0 , 3 , 101)
temp_cut_array = np.linspace(15, 30, 101)
hum_cut_array = np.linspace(60, 100, 101)

weath_IG=information_gain(weather,weath_cut_array)
temp_IG=information_gain(temperature,temp_cut_array)
hum_IG=information_gain(humidity,hum_cut_array)


extra_cut_weather = 1
under_c = len(weather[weather == 1])
over_c = len(weather) - under_c
p_u = under_c/len(weather)
p_o = 1 - p_u

soc_f_u = 0
ind = np.where([weather == 1])[1]
soc_f_u += len(soccer[ind][soccer[ind] == 0])
p_soc_f_u = soc_f_u/under_c
p_soc_t_u = 1 - p_soc_f_u
S_u = entropy(p_soc_t_u, p_soc_f_u)

soc_f_o = 0
ind = np.where([weather != 1])[1]
soc_f_o += len(soccer[ind][soccer[ind] == 0])
p_soc_f_o = soc_f_o/over_c
p_soc_t_o = 1 - p_soc_f_o
S_o = entropy(p_soc_t_o, p_soc_f_o)

extra_IG_weather = S - p_u * S_u - p_o * S_o



plt.plot(weath_cut_array, weath_IG, label='IG weather cuts')
plt.plot(extra_cut_weather, extra_IG_weather, 'rx', label='extra cut weather')
plt.xlabel(r'$ weather cut $')
plt.ylabel(r'$ IG$')
plt.legend(loc='best')
plt.savefig('weather.pdf')
plt.clf()

plt.plot(temp_cut_array, temp_IG, label='IG temperature cuts')
plt.xlabel(r'$ temperature cut $')
plt.ylabel(r'$ IG$')
plt.legend(loc='best')
plt.savefig('temp.pdf')
plt.clf()

plt.plot(hum_cut_array, hum_IG, label='IG humidity cuts')
plt.xlabel(r'$ humidity cut $')
plt.ylabel(r'$ IG$')
plt.legend(loc='best')
plt.savefig('humidity.pdf')
plt.clf()


print('Weather max IG: ', np.max(weath_IG), 'at ', weath_cut_array[np.where(weath_IG == np.max(weath_IG))[0][0]])
print('... or maybe: ', extra_IG_weather, 'with the extra cut')
print('Temperature max IG: ', np.max(temp_IG), 'at ',  temp_cut_array[np.where(temp_IG == np.max(temp_IG))[0][0]])
print('Humidity max IG: ', np.max(hum_IG), 'at ',  hum_cut_array[np.where(hum_IG == np.max(hum_IG))[0][0]])

#print("--- %s seconds ---" % (time.clock() - start_time))
