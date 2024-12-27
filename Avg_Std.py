# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 14:07:33 2024

@author: Denij Giri
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path

def phase_correction(data_td):
    freq_range = np.array([0.25,1.5]) * 10 ** 12
    freqs = data_td[:,0].real
    phase = np.unwrap(np.angle(data_td[:,1]))
    freq_slice_idx = (freqs >= freq_range[0]) * (freqs <= freq_range[1])
    z = np.polyfit(freqs[freq_slice_idx], phase[freq_slice_idx],1)
    phase_corrected =  phase#- z[1]
    return phase_corrected

def fourier_transform(data_td):
    n = data_td[:,1].size
    timestep = 0.05 / 10 ** 12
    freq = np.fft.rfftfreq(n, d = timestep)
    fourier = np.fft.rfft(data_td[:,1])
    pos_slice = freq > 0
    data_fd = np.array([freq[pos_slice], fourier[pos_slice]]).T
    return data_fd
    

def time_window(data_td, win_width = 15, win_start = None, type = ""):
    t = data_td[:,0] - data_td[0,0]
    dt = np.mean(np.diff(t))
    c = data_td[0,1]
    data_td[:,1] = data_td[:,1] - c
    win_width = int(win_width / dt)

    
    if win_start is not None:
        win_start = int(win_start/dt)
    else:
        win_start = np.argmax(np.abs(data_td[:,1])) - win_width //2
        if win_start < 0:
         win_start = 0
    
    window_axis = signal.windows.tukey(win_width, alpha = 0.70)
    zero_pad0 = np.zeros(win_start)
    window_axis = np.concatenate((zero_pad0, window_axis))
    zero_pad1 = np.zeros(len(t) - win_width - win_start)
    window_axis = np.concatenate((window_axis,zero_pad1))
    
    data_td[:,1] = window_axis * data_td[:,1]
    return data_td

base_path_sub = Path(r"C:\Users\Denij Giri\Desktop\Conductivity\Semiconductors\GaAs_wafer25_sub_remeasure")
sub_ref = list((base_path_sub/"Reference").glob("*.txt"))
sub_sam = list((base_path_sub/"Substrate").glob("*.txt"))
 
base_path_sample = Path(r"C:\Users\Denij Giri\Desktop\Conductivity\Semiconductors\GaAsTe_wafer19073_remeasure")
film_ref = list((base_path_sample/"Reference").glob("*.txt"))
film_sam = list((base_path_sample/"Sample").glob("*.txt"))


sub_refs, sub_sams, film_refs, film_sams = [],[],[],[]

for i, (file1 , file2, file3, file4) in enumerate(zip(sub_ref, sub_sam, film_ref, film_sam)):
    sub_refs.append(file1),sub_sams.append(file2), film_refs.append(file3),film_sams.append(file4)

result = []
phi_diff = []
N = []
t = (sub_refs, sub_sams, film_refs, film_sams)
for i , (sub_ref, sub_sam, film_ref, film_sam) in enumerate(zip(*t)):
    sub_sam_td = np.loadtxt(sub_sam)
    sub_ref_td = np.loadtxt(sub_ref)
    film_sam_td = np.loadtxt(film_sam)
    film_ref_td = np.loadtxt(film_ref)
   # print(sub_sam_td.shape)
    
    sub_sam_ad =  time_window(sub_sam_td, type ="sub_sam_td")
    sub_ref_ad = time_window(sub_ref_td, type = "sub_ref_td")
    film_sam_ad =  time_window(film_sam_td, type ="film_sam_td")
    film_ref_ad = time_window(film_ref_td, type = "film_ref_td")
    
    sub_sam_bd = fourier_transform(sub_sam_ad)
    sub_ref_bd = fourier_transform(sub_ref_ad)
    film_sam_bd = fourier_transform(film_sam_ad)
    film_ref_bd = fourier_transform(film_ref_ad)
    #print(sub_sam_bd.shape)
    #print(sub_sam_bd[:,0])
   
    phi_sub_sam_cd = phase_correction(sub_sam_bd)
    phi_sub_ref_cd = phase_correction(sub_ref_bd)
    phi_film_sam_cd = phase_correction(film_sam_bd)
    phi_film_ref_cd = phase_correction(film_ref_bd)
    #print(phi_sub_sam_cd.shape)
    
    phi_sub =   phi_sub_sam_cd - phi_sub_ref_cd 
    phi_sam =  phi_film_sam_cd -  phi_film_ref_cd 
    phi_dif =  phi_sam - phi_sub

    

    r_sub_sam_bd = np.abs(sub_sam_bd[:,1])
    r_sub_ref_bd = np.abs(sub_ref_bd[:,1])
    r_film_sam_bd = np.abs(film_sam_bd[:,1])
    r_film_ref_bd = np.abs(film_ref_bd[:,1])
    
    delta_sub = phi_sub_ref_cd - phi_sub_sam_cd
    
    freqs = sub_sam_bd[:,0].real
  
    sub_sam_ed = np.array([freqs,r_sub_sam_bd * np.exp(-1j * phi_sub_sam_cd)]).T
    sub_ref_ed = np.array([freqs,r_sub_ref_bd * np.exp(-1j * phi_sub_ref_cd)]).T
    film_sam_ed = np.array([freqs,r_film_sam_bd * np.exp(-1j * phi_film_sam_cd)]).T
    film_ref_ed = np.array([freqs,r_film_ref_bd * np.exp(-1j * phi_film_ref_cd)]).T
    #print(sub_sam_ed.shape)
    #print(sub_sam_ed.real[1])
    
    T_sub = np.array([freqs, sub_sam_ed[:,1] / sub_ref_ed[:,1]]).T
    T_film = np.array([freqs, film_sam_ed[:,1] / film_ref_ed[:,1]]).T
    #print(T_sub.shape)
    c = 3 * 10 ** 8 * 100
    d_sub = 500 * 10 ** -6 * 100
    omega = 2*np.pi*freqs
    epsilon_0 = 8.854127 * 10 ** -12 / 100
    d_film = 700 * 10 **-9 * 100
    n = 1 + (c * delta_sub) / (omega*d_sub)

    N.append(n)
  
    #n = 3.8 * np.ones_like(n)

    freqs_range = (freqs>=0.4*10**12)*(freqs< 3*10**12)
    freqs = freqs[freqs_range]
    sigma = (T_sub[:,1] - T_film[:,1]) * epsilon_0 * c * (1+n)/((T_film[:,1])*d_film)
   # print(sigma.shape)
    
    phi_diff.append(phi_dif)
    result.append(sigma)
    
Ids = np.array(N)
avg_idx = np.mean(Ids, axis = 0)

avg_idx = avg_idx[freqs_range]

sdv_idx = np.std(Ids,axis = 0)
sdv_idx = sdv_idx[freqs_range] 
print(sdv_idx)

plt.figure("RI")

plt.tick_params(axis = 'x', which = 'both', top ='True',bottom = 'True',direction = 'in', labelsize = 14)
plt.tick_params(axis = 'y', which = 'both', left ='True',right = 'True',direction = 'in', labelsize = 14)
plt.xticks(fontsize = 14, fontweight = 'bold')
plt.yticks(fontsize = 14, fontweight = 'bold')


plt.errorbar(freqs/10**12,avg_idx, yerr = sdv_idx, errorevery = 1, elinewidth =0.5,color ='blue')
plt.annotate('Refractive index', xy = (1.8,3.635), xytext = (1.8,3.66), color='blue', arrowprops = dict(facecolor='blue', color ='blue', arrowstyle='->'), fontsize =14, fontweight = 'bold')

plt.ylim(2,5)
plt.xlabel('Frequency[THz]', weight = 'bold', fontsize = 14)
plt.ylabel('Refractive index', weight = 'bold', fontsize = 14)
plt.title('Refractive index Vs Frequency',fontsize = 14,weight = 'bold')

    

    
results = np.array(result)
avg_sigma = np.mean(results,axis=0) 
#print(freqs.shape)
avg_sigma = avg_sigma[freqs_range]
#print(avg_sigma)
sdv_sigma = np.std(results, axis = 0)
sdv_sigma = sdv_sigma[freqs_range]
#print(sdv_sigma.shape)
plt.figure(figsize=(10,10))
plt.subplot(2,1,1)

plt.tick_params(axis = 'x', which = 'both', top ='True',bottom = 'True',direction = 'in', labelsize = 14)
plt.tick_params(axis = 'y', which = 'both', left ='True',right = 'True',direction = 'in', labelsize = 14)
plt.xticks(fontsize = 14, fontweight = 'bold')
plt.yticks(fontsize = 14, fontweight = 'bold')


plt.errorbar(freqs/10**12,avg_sigma.real, yerr = sdv_sigma, errorevery = 1, elinewidth =0.5, color ='red')
plt.errorbar(freqs/10**12,avg_sigma.imag, yerr = sdv_sigma, errorevery = 1, elinewidth =0.5, color ='blue')

plt.annotate('Real Part', xy = (1.5,300), xytext = (1.5,450), color ='red', arrowprops = dict(facecolor='red',color ='red', arrowstyle='->'), fontsize =14, fontweight = 'bold')
plt.annotate('Imaginary part', xy =(1.5,0), xytext = (1,-150),color ='blue', arrowprops = dict(facecolor='blue',color='blue', arrowstyle='->'), fontsize =14, fontweight = 'bold')

plt.xlabel('Frequency[THz]',  weight = 'bold', fontsize = 14)
plt.ylabel('Conductivity[S/cm]', weight = 'bold', fontsize = 14)

plt.title('Top:Conductivity  Bottom:Phase difference (Vs Frequency)', weight = 'bold', fontsize = 16)
#plt.legend(prop = {'size' : 14, 'weight':'bold'})
plt.show()

phi_diff = np.array(phi_diff)
avg_phi_diff = np.mean(phi_diff,axis=0) 
sdv_phi = np.std(phi_diff, axis = 0)
sdv_phi = sdv_phi[freqs_range]
avg_phi_diff = avg_phi_diff[freqs_range]
plt.subplot(2,1,2)
plt.tick_params(axis = 'x', which = 'both', top ='True',bottom = 'True',direction = 'in', labelsize = 14)
plt.tick_params(axis = 'y', which = 'both', left ='True',right = 'True',direction = 'in', labelsize = 14)
plt.xticks(fontsize = 14, fontweight = 'bold')
plt.yticks(fontsize = 14, fontweight = 'bold')
plt.errorbar(freqs/10**12,avg_phi_diff, yerr = sdv_phi, errorevery = 1, elinewidth =0.5,color ='blue')
#plt.errorbar(freqs/10**12,avg_phi_diff, yerr = sdv_phi, errorevery = 1, elinewidth =0.5)

plt.annotate('Phase difference', xy = (1.5,0.0), xytext = (1,-0.2),color='blue', arrowprops = dict(facecolor='blue',color ='blue', arrowstyle='->'), fontsize =14, fontweight = 'bold')

plt.xlabel('Frequency[THz]', weight = 'bold', fontsize = 14)
plt.ylabel('Phase difference(radians)', weight = 'bold', fontsize = 14)
#plt.title('Frequency Vs Phase_Difference', weight = 'bold', fontsize = 8)
#plt.legend()






