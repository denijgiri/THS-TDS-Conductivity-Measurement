# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 16:29:31 2024

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
    freq = np.fft.fftfreq(n, d = timestep)
    fourier = np.fft.fft(data_td[:,1])
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
phi_diff = []
amp_error = []
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
    R_sub_sam_bd =  sub_sam_amp / sub_ref_amp 
     
    
    
    delta_sub =   phi_sub_ref_cd - phi_sub_sam_cd 
    delta_sam =   phi_film_ref_cd  - phi_film_sam_cd 
    
    freqs = sub_sam_bd[:,0].real
    
    
    r_sub_sam_bd = np.abs(sub_sam_bd[:,1])
    r_sub_ref_bd = np.abs(sub_ref_bd[:,1])
    r_film_sam_bd = np.abs(film_sam_bd[:,1])
    r_film_ref_bd = np.abs(film_ref_bd[:,1])
    

  
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
    n = 1 + (c * delta_sub) / (omega*d_sub)
    #n = 3.8 * np.ones_like(n)

    freqs_range = (freqs>=0.99*10**12)*(freqs< 1.01*10**12)
    
    
    
    sigma = (T_sub[:,1] - T_film[:,1]) * epsilon_0 * c * (1+n)/((T_film[:,1])*d_film)
    result.append(sigma)
      
    a = (2 * np.exp(1j*delta_sub)) / (np.exp(1j*delta_sam)*d_film*R_film_sam_bd)
    b = (c * np.exp(1j*delta_sub)*delta_sub) / (np.exp(1j*delta_sam)*d_film*omega *d_sub *R_film_sam_bd)
    e =  (a + b )* (epsilon_0 * c)
    
    x = (2 * np.exp(1j *delta_sub) * R_sub_sam_bd * (-1)) / (np.exp(1j*delta_sam) * d_film * R_film_sam_bd**2) 
    y = (c * np.exp(1j * delta_sub)*delta_sub * R_sub_sam_bd * (-1)) / (np.exp(1j*delta_sam) * d_film *omega *d_sub *R_film_sam_bd**2)
    z = (x + y) * (epsilon_0 * c)
    
    am_error = (e*1.84)**2 + (z*1.64)**2
    am_error = np.sqrt(am_error)
    amp_error.append(am_error)
    
   
amp_error = np.array(amp_error)
amp_error = np.mean(amp_error,axis=0)
amp_error = amp_error[freqs_range]
print(amp_error)

results = np.array(result)
avg_sigma = np.mean(results,axis=0) 
avg_sigma = avg_sigma[freqs_range]
print(avg_sigma)

#sdv_amp = sdv_amp[freqs_range]
#plt.figure()
#plt.plot(freqs[freqs_range],sdv_amp)
#plt.figure()
#plt.errorbar(freqs[freqs_range],avg_sigma.real, yerr = sdv_amp, errorevery = 1, elinewidth =0.5, label ='Real Part')
#plt.errorbar(freqs[freqs_range],avg_sigma.imag, yerr = sdv_amp, errorevery = 1, elinewidth =0.5, label = 'Imaginary Part')
#plt.xlabel('Frequency[THz]', weight = 'bold')
#plt.ylabel('Conductivity[Ω−1cm-1]', weight = 'bold')
#plt.legend()


#phi_diff = np.array(phi_diff)
#avg_phi_diff = np.mean(phi_diff,axis=0) 
#sdv_phi = np.std(phi_diff, axis = 0)
#sdv_phi = sdv_phi[freqs_range]
#avg_phi_diff = avg_phi_diff[freqs_range]
#plt.figure()
#plt.errorbar(freqs[freqs_range],avg_phi_diff, yerr = sdv_phi, errorevery = 1, elinewidth =0.5)
#plt.xlabel('Frequency[THz]', weight = 'bold')
#plt.ylabel('Conductivity[Ω−1cm-1]', weight = 'bold')
#plt.legend()






