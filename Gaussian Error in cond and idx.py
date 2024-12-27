# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:45:09 2024

@author: Denij Giri
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def phase_correction(data_td):
    freq_range = np.array([0.25,0.5]) * 10 ** 12
    freqs = data_td[:,0].real
    phase = np.unwrap(np.angle(data_td[:,1]))
    freq_slice_idx = (freqs >= freq_range[0]) * (freqs <= freq_range[1])
    z = np.polyfit(freqs[freq_slice_idx], phase[freq_slice_idx],1)
    phase_corrected = freqs * z[0] 
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


def analysis(film_sam, film_ref, sub_sam, sub_ref):
    sub_sam_td = np.loadtxt(sub_sam)
    sub_ref_td = np.loadtxt(sub_ref)
    film_sam_td = np.loadtxt(film_sam)
    film_ref_td = np.loadtxt(film_ref)

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
   
    r_sub_sam_bd = np.abs(sub_sam_bd[:,1])
    r_sub_ref_bd = np.abs(sub_ref_bd[:,1])
    r_film_sam_bd = np.abs(film_sam_bd[:,1])
    r_film_ref_bd = np.abs(film_ref_bd[:,1])
    
    
    delta_sub = phi_sub_ref_cd - phi_sub_sam_cd
    delta_sam =  phi_film_ref_cd  -   phi_film_sam_cd  
    
    
    
    
    freqs = sub_sam_bd[:,0].real
    
    
    
    sub_sam_ed = np.array([freqs,r_sub_sam_bd * np.exp(-1j * phi_sub_sam_cd)]).T
    sub_ref_ed = np.array([freqs,r_sub_ref_bd * np.exp(-1j * phi_sub_ref_cd)]).T
    film_sam_ed = np.array([freqs,r_film_sam_bd * np.exp(-1j * phi_film_sam_cd)]).T
    film_ref_ed = np.array([freqs,r_film_ref_bd * np.exp(-1j * phi_film_ref_cd)]).T
    
    T_sub = np.array([freqs, sub_sam_ed[:,1] / sub_ref_ed[:,1]]).T
    T_film = np.array([freqs, film_sam_ed[:,1] / film_ref_ed[:,1]]).T
    
    
    
    c = 3 * 10 ** 8 * 100
    d_sub = 1 * 10 ** -3 * 100
    omega = 2*np.pi*freqs
    epsilon_0 = 8.854127 * 10 ** -12 / 100
    d_film = 350 * 10 **-9 * 100
    n = 1 + c * delta_sub / (omega*d_sub)
    n_dev = c / (omega * d_sub)
    
    n_error = np.sqrt((n_dev * 0.778)**2)
    
    sigma = (T_sub[:,1] - T_film[:,1]) * epsilon_0 * c * (1+n)/((T_film[:,1])*d_film)
 
    
    w = (2*r_sub_sam_bd*1j*np.exp(1j*delta_sub))/(r_film_sam_bd*d_film*np.exp(1j*delta_sam))
    x = (r_sub_sam_bd*c)*((delta_sub*1j*np.exp(1j*delta_sub))+(np.exp(1j*delta_sub))) 
    y = omega*d_sub*r_film_sam_bd*d_film*np.exp(1j*delta_sam)
    z = c /(d_film * omega *d_sub) 
    phi2 = (w + (x/y) - z) * (epsilon_0*c)

    
    e = (2*r_sub_sam_bd*(-1j)*np.exp(1j*delta_sub)) /  (r_film_sam_bd * d_film * np.exp(1j*delta_sam))
    f =( r_sub_sam_bd * c * (-1j) * np.exp(1j*delta_sub) * delta_sub ) / (omega*d_sub*r_film_sam_bd*d_film*np.exp(1j*delta_sam))
    phi1 = (e + f) * (epsilon_0 *c)
    

    
     
    sigma_error = np.sqrt(((phi2*0.778)**2) + ((phi1*0.778)**2))
    #print(sigma_error)
    

    
   #print(sigma_error.real)
    #print(np.mean(sigma.real))
    #print(np.mean(sigma_error.real))
    
    
    #standard_sigma = (T_sub[:,1] - T_film[:,1]) * epsilon_0 * c**2 / T_film[:,1]*(d_film*omega*d_sub)

    #sigma_error = np.sqrt((standard_sigma * phase_difference) **2)
    
    #derivative_idx = c / (omega * d_sub)
    #idx_error = np.sqrt((derivative_idx * phi_std)**2)
    
    return  freqs, sigma, sigma_error , n , n_error

sub_sam = r"C:\Users\Denij Giri\Desktop\Conductivity\Silver Sample\sample3\img0\2023-12-11T17-02-14.186639-sam3_20avg_0rh-sam-X_45.000 mm-Y_5.000 mm.txt"
sub_ref =r"C:\Users\Denij Giri\Desktop\Conductivity\Silver Sample\sample3\img0\2023-12-11T17-02-00.790640-sam3_20avg_0rh-ref-X_-10.000 mm-Y_5.000 mm.txt"
film_sam = r"C:\Users\Denij Giri\Desktop\Conductivity\Silver Sample\sample3\img0\2023-12-11T16-59-33.214643-sam3_20avg_0rh-sam-X_25.000 mm-Y_5.000 mm.txt"
film_ref = r"C:\Users\Denij Giri\Desktop\Conductivity\Silver Sample\sample3\img0\2023-12-11T16-58-48.481640-sam3_20avg_0rh-ref-X_-10.000 mm-Y_5.000 mm.txt"

freqs, sigma , sigma_error, n , n_error = analysis(film_sam, film_ref, sub_sam, sub_ref)



#print(sigma)
#print(np.mean(sigma))
#print(np.mean(idx_error.real))
#print(sigma_error.real)
#print(np.mean(sigma_error.real))
#print(n)
#print(idx_error)

#plt.figure()
#plt.plot(freqs/10**12, sigma , color = 'red', label = 'Real part')
#plt.plot(freqs/10**12, sigma.imag , color = 'Orange', label = 'Imaginary part')
#plt.plot(freqs/10**12, sigma.imag, color = 'blue',label = 'Imag part')
#plt.plot(freqs/10**12,n)
#plt.fill_between(freqs/10**12, n + n_error.real, n - n_error.real, alpha=0.5, color='gray',interpolate = False)
#plt.title('Refractive error of the sample(Real Part)')
#plt.xlabel('Frequency(THz)')
#plt.ylabel('Refractive Index(n)')
#plt.legend()


#plt.plot(freqs/10**12, sigma.real)
#plt.xlabel('Frequency(THz)')
#plt.ylabel('Conductivity of the sample(Seimens per meter')
#plt.title('Conductivity Vs Frequency of the sample')

plt.figure()
plt.plot(freqs/10**12, sigma.real , color = 'Blue')
plt.fill_between(freqs/10**12, sigma.real + sigma_error.real, sigma.real - sigma_error.real, alpha=0.5, color='grey')
plt.title('Conductivity error of the sample(Real Part)')
plt.xlabel('Frequency(THz)')
plt.ylabel('Conductivity(Ω−1cm-1)')

plt.figure()
plt.plot(freqs/10**12, sigma.imag, color = 'Red')
plt.fill_between(freqs/10**12, sigma.imag + sigma_error.imag, sigma.imag - sigma_error.imag, alpha=0.5, color='grey')
plt.title('Conductivity error of the sample(Imaginary Part)')
plt.xlabel('Frequency(THz)')
plt.ylabel('Conductivity(Ω−1cm-1)')
plt.legend()


#plt.figure()
#plt.plot(freqs/10**12, sigma)
#plt.fill_between(freqs/10**12, sigma.imag + sigma_error.imag, sigma.imag - sigma_error.imag, alpha=0.5, color='gray',label='Error band (±1σ)')
#plt.title('Conductivity error of the sample(Imaginary Part)')
#plt.xlabel('Frequency(THz)')
#plt.ylabel('Conductivity(Siemens per meter)')
#print(refractive_index)
#print(np.mean(standard_refractive_idx))
#print(np.mean(refractive_index))
#print(np.mean(sigma))
#print(np.mean(standard_sigma) )

#plt.figure()
#plt.plot(freqs/10**12, n, color='Blue')
#plt.xlabel('Frequency (THz)')
#plt.ylabel('Refractive index')
#plt.title('Refractive index Vs Frequency(THz)')



