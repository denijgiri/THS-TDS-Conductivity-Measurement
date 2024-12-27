# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 10:48:44 2024

@author: Denij Giri
"""


import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import re
from scipy import signal
from pathlib import Path


def phase_correction(data_td):
    freq_range = np.array([0.25,1.5]) * 10 ** 12
    freqs = data_td[:,0].real
    phase = np.unwrap(np.angle(data_td[:,1]))
    freq_slice_idx = (freqs >= freq_range[0]) * (freqs <= freq_range[1])
    #z = np.polyfit(freqs[freq_slice_idx], phase[freq_slice_idx],1)
    phase_corrected =  phase#- z[1]
    plt.plot(freqs,phase)
    return phase_corrected

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
        
def fourier_transform(data_td):
    n = data_td[:,1].size
    timestep = 0.05 / 10 ** 12
    freq = np.fft.fftfreq(n, d = timestep)
    fourier = np.fft.fft(data_td[:,1])
    pos_slice = freq > 0
    data_fd = np.array([freq[pos_slice], fourier[pos_slice]]).T
    return data_fd


base_path  = Path(r"C:\Users\Denij Giri\Desktop\Conductivity\GaAaTe_wafer_sam Reameasure")
all_files = list(base_path.glob("*.txt"))

sub_refs = []
sub_sams = []
film_refs = []
film_sams = []


for file in all_files:
        if 'reference' in str(file):
            sub_refs.append(file)
        elif 'substrate' in str(file):
            sub_sams.append(file)
        elif 'hole' in str(file):
            film_refs.append(file)
        else:
            film_sams.append(file)
 



result = []
phase_err = []
t = (sub_refs, sub_sams, film_refs, film_sams)
for i , (sub_ref, sub_sam, film_ref, film_sam) in enumerate(zip(*t)):
    sub_ref_td = np.loadtxt(sub_ref)
    sub_sam_td = np.loadtxt(sub_sam)
    film_ref_td = np.loadtxt(film_ref)
    film_sam_td = np.loadtxt(film_sam)
    
    
    sub_sam_ad =  time_window(sub_sam_td, type ="sub_sam_td")
    sub_ref_ad = time_window(sub_ref_td, type = "sub_ref_td")
    film_sam_ad =  time_window(film_sam_td, type ="film_sam_td")
    film_ref_ad = time_window(film_ref_td, type = "film_ref_td")
    
    sub_sam_bd = fourier_transform(sub_sam_ad)
    sub_ref_bd = fourier_transform(sub_ref_ad)
    film_sam_bd = fourier_transform(film_sam_ad)
    film_ref_bd = fourier_transform(film_ref_ad)
    
   
    phi_sub_sam_cd = phase_correction(sub_sam_bd)
    phi_sub_ref_cd = phase_correction(sub_ref_bd)
    phi_film_sam_cd = phase_correction(film_sam_bd)
    phi_film_ref_cd = phase_correction(film_ref_bd)
    
   
    sub_sam_amp = np.abs(sub_sam_bd[:,1])
    sub_ref_amp = np.abs(sub_ref_bd[:,1])
    film_sam_amp = np.abs(film_sam_bd[:,1])
    film_ref_amp = np.abs(film_ref_bd[:,1])
     
    R_film_sam_bd =  film_sam_amp / film_ref_amp        
    R_sub_sam_bd =  sub_sam_amp /  sub_ref_amp  
     

    delta_sub =   phi_sub_sam_cd - phi_sub_ref_cd 
    delta_sam =  phi_film_sam_cd - phi_film_ref_cd 
    
    
    
    r_sub_sam_bd = np.abs(sub_sam_bd[:,1])
    r_sub_ref_bd = np.abs(sub_ref_bd[:,1])
    r_film_sam_bd = np.abs(film_sam_bd[:,1])
    r_film_ref_bd = np.abs(film_ref_bd[:,1])
    
    delta_sub1 = phi_sub_sam_cd -  phi_sub_ref_cd 
    
 
    
    
    freqs = sub_sam_bd[:,0].real
    
    freqs_range = (freqs>=0.999*10**12)*(freqs< 1.02*10**12)
    
   
  
    sub_sam_ed = np.array([freqs,r_sub_sam_bd * np.exp(-1j * phi_sub_sam_cd)]).T
    sub_ref_ed = np.array([freqs,r_sub_ref_bd * np.exp(-1j * phi_sub_ref_cd)]).T
    film_sam_ed = np.array([freqs,r_film_sam_bd * np.exp(-1j * phi_film_sam_cd)]).T
    film_ref_ed = np.array([freqs,r_film_ref_bd * np.exp(-1j * phi_film_ref_cd)]).T
    
  
    
    T_sub = np.array([freqs, sub_sam_ed[:,1] / sub_ref_ed[:,1]]).T
    T_film = np.array([freqs, film_sam_ed[:,1] / film_ref_ed[:,1]]).T

    c = 3 * 10 ** 8 * 100
    d_sub = 500 * 10 ** -6 * 100
    omega = 2*np.pi*freqs
    epsilon_0 = 8.854127 * 10 ** -12 / 100
    d_film = 700 * 10 **-9 * 100
    n = 3.8 # 1 + (c * delta_sub1) / (omega*d_sub)
  
    #n = 3.8 * np.ones_like(n)
    
    sigma = (T_sub[:,1] - T_film[:,1]) * epsilon_0 * c * (1+n)/((T_film[:,1])*d_film)

    
    #phi_2 = (-c *epsilon_0 *c) / (omega * d_sub * d_film)
    
     
    w = (2*R_sub_sam_bd*1j*np.exp(1j*delta_sub))/(R_film_sam_bd*d_film*np.exp(1j*delta_sam))
    x = (R_sub_sam_bd*c)*((delta_sub*1j*np.exp(1j*delta_sub))+(np.exp(1j*delta_sub))) 
    y = omega*d_sub*R_film_sam_bd*d_film*np.exp(1j*delta_sam)
    z = c /(d_film * omega *d_sub) 
    phi2 = (w + (x/y) - z) * (epsilon_0*c)

    
    e = (2*R_sub_sam_bd*(-1j)*np.exp(1j*delta_sub)) /  (R_film_sam_bd * d_film * np.exp(1j*delta_sam))
    f =( R_sub_sam_bd * c * (-1j) * np.exp(1j*delta_sub) * delta_sub ) / (omega*d_sub*R_film_sam_bd*d_film*np.exp(1j*delta_sam))
    phi1 = (e + f) * (epsilon_0 *c)
    
    result.append(sigma)
    phase_error = np.sqrt(((phi2*(0.29))**2) + ((phi1*(0.22))**2))
   
    
    phase_err.append(phase_error)
    
    
phase_err = np.array(phase_err) 
phase_err = np.mean(phase_err, axis = 0)
phase_err = phase_err[freqs_range]
print(phase_err)

    

sigma_1 = np.mean(result,axis=0) 
sigma_1 = sigma_1[freqs_range]
print(sigma_1)

#phase_err = np.array(phase_err)

#sdv_phase = np.mean(phase_err,axis =0)

#sdv_phase = sdv_phase[freqs_range]
#plt.figure()
#plt.plot(freqs[freqs_range],sdv_phase, color = 'Orange')
#plt.scatter(freqs[freqs_range],sdv_phase, color = 'Orange')
#plt.figure()
#plt.errorbar(freqs[freqs_range],phase_error[freqs_range], yerr = sdv_phase, errorevery = 1, elinewidth =0.5, label ='Real Part')
#plt.figure()
#plt.errorbar(freqs[freqs_range],avg_sigma.imag, yerr = sdv_phase, errorevery = 1, elinewidth =0.5, label = 'Imaginary Part')
#plt.xlabel('Frequency[THz]', weight = 'bold')
#plt.ylabel('Phase Error[Radians]', weight = 'bold')
#plt.legend()

