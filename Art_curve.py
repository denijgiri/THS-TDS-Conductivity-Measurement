# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 10:50:24 2024

@author: Denij Giri
"""

import numpy as np
import matplotlib.pyplot as plt


x_real_ma = [0.15246609053525187,0.475336243603255,0.8026902909513447,1.1255604440193478,1.4035872066574848,1.704035835573209,2.031389882921299,2.331838511837023,2.6860983197427863,2.9596408459755295]
y_real_ma = [417.18748952262195,363.2811728166485,297.65628597233115,241.40625523868897,201.56253562308527,171.09380622859538,140.6249874271463,128.90623847488408,103.12502654269097,91.40627759042876]

x_imag_ma = [0.20597008093079558,0.3895524437747787,0.6223881280244287,0.9179104477611942,1.1955225930285096,1.5223880597014927,1.938806106795126,2.2791045459348758,2.8522386693242767,2.605970354222539]	
y_imag_ma = [119.29834580508381,166.08195483756995,208.1870958876242,219.8829981457454,229.23975564530383,222.22214290430884,208.1870958876242,	191.81290411237612,177.77785709569116,184.79529137138113]


x_real = [0.5361678627733925,0.7015069737510233,0.8952637967824979,1.1639396303841743,1.4455327760508299,1.755543535221563,2.0578041436727474,2.3264799772744236,2.579655706536535,2.796663080276422]
y_real = [394.72818883166605,362.4382401600841,330.1482914885022,	291.927537970143,255.6837152110365,	234.59639512358504,224.0527476488785,226.0296784081312,237.89128810168557,241.84514962019094]

x_imag = [0.5439182105924738,0.73509145295123,0.9650161387560998,1.2078579395264142,1.4713670059817245,1.6857911932479612,1.9234660297723767,2.148223948430878,	2.411733014886188,2.6984929276992102]
y_imag = [95.5518963649171,116.63921645236857,130.47778204321412,126.5238953866703,121.25207164931702,110.04945563420578,99.50575788342239,85.66721743061531,68.53378399970754,39.538715737207056]

plt.figure()

sdv_real_ma = np.std(y_real_ma,axis=0)
sdv_imag_ma = np.std(y_imag_ma,axis =0)
sdv_real = np.std(y_real,axis=0)
sdv_imag = np.std(y_imag,axis=0)

plt.plot(x_real_ma,y_real_ma, color ='red')

#plt.errorbar(x_real_ma,y_real_ma, yerr = sdv_real_ma, errorevery = 1, elinewidth =0.3, color ='red')
#plt.errorbar(x_imag_ma,y_imag_ma, yerr = sdv_imag_ma, errorevery = 1, elinewidth =0.3, color ='blue')
#plt.errorbar(x_real,y_real, yerr = sdv_real, errorevery = 1, elinewidth =0.3, color ='orange')
#plt.errorbar(x_imag,y_imag, yerr = sdv_imag, errorevery = 1, elinewidth =0.3, color ='green')

plt.plot(x_imag_ma,y_imag_ma,color ='blue')
plt.plot(x_real,y_real,color = 'orange')
plt.plot(x_imag,y_imag,color ='green')


plt.tick_params(axis = 'x', which = 'both', top ='True',bottom = 'True',direction = 'in', labelsize = 14)
plt.tick_params(axis = 'y', which = 'both', left ='True',right = 'True',direction = 'in', labelsize = 14)
plt.xticks(fontsize = 14, fontweight = 'bold')
plt.yticks(fontsize = 14, fontweight = 'bold')

plt.annotate('Real part', xy = (1,320), xytext = (1,400),color='orange', arrowprops = dict(facecolor='orange',color='orange', arrowstyle='->'), fontsize =14, fontweight = 'bold')
plt.annotate('Real part(Marl.)', xy =(1,270), xytext = (1.5,350), color ='red', arrowprops = dict(facecolor='red',color='red', arrowstyle='->'), fontsize =14, fontweight = 'bold')

plt.annotate('Imag. part(Marl.)', xy = (1,220), xytext = (0.7,50),color='blue', arrowprops = dict(facecolor='blue',color='blue', arrowstyle='->'), fontsize =14, fontweight = 'bold')
plt.annotate('Imag. part', xy = (1.8,100), xytext = (1.8,40),color='green', arrowprops = dict(facecolor='green',color='green', arrowstyle='->'), fontsize =14, fontweight = 'bold')

plt.xlabel('Frequency [THz]', weight = 'bold', fontsize = 14)
plt.ylabel('Conductivity [S/cm]',weight = 'bold', fontsize = 14)
plt.title('Conducitivity Vs Frquency(Now and Marielena)',weight = 'bold', fontsize = 14)
plt.show()


 

 	
	
	
	
	
	
	
	
	
	


