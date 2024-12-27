# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 09:26:35 2024

@author: Denij Giri
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def phase_correction(data_td):
    freq_range = np.array([0.25,0.5]) * 10 ** 12
    freqs = data_td[:,0].real
    angle1 = np.angle(data_td[:,1])
    phase = np.unwrap(angle1)
    #freq_slice_idx = (freqs >= freq_range[0]) * (freqs <= freq_range[1])
    #z = np.polyfit(freqs[freq_slice_idx], phase[freq_slice_idx],1)
    #phase_corrected =  phase - z[1]
    return phase

def fourier_transform(data_td):
    n = data_td[:,1].size
    timestep = 0.05 / 10 **12
    freq = np.fft.fftfreq(n, d = timestep)
    
    fourier = np.fft.fft(data_td[:,1])
    #pos_slice = freq > 0
    pos_slice = (freq > 1.98 * 10 **12) * (freq < 2.01*10**12)
    #f = freq[pos_slice]
    #pos_slice = (freq >=  0.99/10**12) * (freq <= 8/10**12)
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
    
    #plt.figure()
    #plt.plot(sub_sam_td[:,0],sub_sam_td[:,1],color = 'blue', label = 'substrate')
    #plt.plot(sub_ref_td[:,0],sub_ref_td[:,1],color = 'green',label ='Reference')
    #plt.plot(film_sam_td[:,0],film_sam_td[:,1],color = 'red',label = 'sample')
    #plt.legend()
    
    sub_sam_ad =  time_window(sub_sam_td, type ="sub_sam_td")
    sub_ref_ad = time_window(sub_ref_td, type = "sub_ref_td")
    film_sam_ad =  time_window(film_sam_td, type ="film_sam_td")
    film_ref_ad = time_window(film_ref_td, type = "film_ref_td")
    
    sub_sam_bd = fourier_transform(sub_sam_ad)
    sub_ref_bd = fourier_transform(sub_ref_ad)
    film_sam_bd = fourier_transform(film_sam_ad)
    film_ref_bd = fourier_transform(film_ref_ad)
    
    #plt.figure()
    #plt.plot(sub_sam_bd[:,0]/10**12,20*np.log10(sub_sam_bd[:,1]), color = 'blue', label = 'substrate')
   # plt.plot(sub_ref_bd[:,0]/10**12,20*np.log10(sub_ref_bd[:,1]), color = 'green', label = 'Reference')
   # plt.plot(film_sam_bd[:,0]/10**12,20*np.log10(film_sam_bd[:,1]), color = 'red', label = 'sample')
    
   
    phi_sub_sam_cd = phase_correction(sub_sam_bd)
    phi_sub_ref_cd = phase_correction(sub_ref_bd)
    phi_film_sam_cd = phase_correction(film_sam_bd)
    phi_film_ref_cd = phase_correction(film_ref_bd)
    
    
    
    
   
    r_sub_sam_bd = np.abs(sub_sam_bd[:,1])
    r_sub_ref_bd = np.abs(sub_ref_bd[:,1])
    r_film_sam_bd = np.abs(film_sam_bd[:,1])
    r_film_ref_bd = np.abs(film_ref_bd[:,1])
    
    
    delta_sub = phi_sub_ref_cd - phi_sub_sam_cd
    #print(len(delta_sub))
    

    
    
    freqs = sub_sam_bd[:,0].real
    #plt.figure()
    #plt.plot(freqs/10**12,delta_sub,label ='Phase_differece', color ='Red')
    #plt.legend()
    #print(len(freqs))

    

    
    
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

     
    R = ( r_sub_sam_bd / r_sub_ref_bd) / ( r_film_sam_bd / r_film_ref_bd)
    K = (epsilon_0 * c * (n + 1)) / d_film
    phi_sub =  phi_sub_ref_cd -  phi_sub_sam_cd
    phi_film = phi_film_ref_cd - phi_film_sam_cd
    phi_diff = phi_sub - phi_film
    angle_sin = np.sin(phi_sub - phi_film)
    
    imag_part = R * K *angle_sin
    Real_part = K * (R * np.cos(phi_film) - 1)
    
  
    
    sigma = (T_sub[:,1] - T_film[:,1]) * epsilon_0 * c * (1+n)/((T_film[:,1])*d_film)
 
    
    return freqs, sigma ,R , K , angle_sin , Real_part, phi_diff, angle_sin, imag_part, phi_sub, phi_film
    

sub_sam = [r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\01 GaAs Wafer 25\2022-02-14T10-50-43.433088-data-sample-X_35.000 mm-Y_12.000 mm.txt",r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\01 GaAs Wafer 25\2022-02-14T10-51-45.973960-data-sample-X_35.000 mm-Y_12.000 mm.txt",r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\01 GaAs Wafer 25\2022-02-14T10-52-48.495953-data-sample-X_35.000 mm-Y_12.000 mm.txt",r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\01 GaAs Wafer 25\2022-02-14T10-53-51.198952-data-sample-X_35.000 mm-Y_12.000 mm.txt",r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\01 GaAs Wafer 25\2022-02-14T10-54-54.117951-data-sample-X_35.000 mm-Y_12.000 mm.txt",r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\01 GaAs Wafer 25\2022-02-14T10-55-57.088959-data-sample-X_35.000 mm-Y_12.000 mm.txt",r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\01 GaAs Wafer 25\2022-02-14T10-56-59.954960-data-sample-X_35.000 mm-Y_12.000 mm.txt",r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\01 GaAs Wafer 25\2022-02-14T10-58-02.818952-data-sample-X_35.000 mm-Y_12.000 mm.txt",r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\01 GaAs Wafer 25\2022-02-14T10-59-05.642951-data-sample-X_35.000 mm-Y_12.000 mm.txt",r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\01 GaAs Wafer 25\2022-02-14T11-00-08.219953-data-sample-X_35.000 mm-Y_12.000 mm.txt"]
sub_ref = [r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\01 GaAs Wafer 25\2022-02-14T10-50-11.991054-data-reference-X_-12.000 mm-Y_12.000 mm.txt",r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\01 GaAs Wafer 25\2022-02-14T10-51-14.871952-data-reference-X_-12.000 mm-Y_12.000 mm.txt",r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\01 GaAs Wafer 25\2022-02-14T10-52-17.408953-data-reference-X_-12.000 mm-Y_12.000 mm.txt",r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\01 GaAs Wafer 25\2022-02-14T10-53-19.773960-data-reference-X_-12.000 mm-Y_12.000 mm.txt",r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\01 GaAs Wafer 25\2022-02-14T10-54-22.476953-data-reference-X_-12.000 mm-Y_12.000 mm.txt",r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\01 GaAs Wafer 25\2022-02-14T10-55-25.709961-data-reference-X_-12.000 mm-Y_12.000 mm.txt",r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\01 GaAs Wafer 25\2022-02-14T10-56-28.356959-data-reference-X_-12.000 mm-Y_12.000 mm.txt",r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\01 GaAs Wafer 25\2022-02-14T10-57-31.403953-data-reference-X_-12.000 mm-Y_12.000 mm.txt",r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\01 GaAs Wafer 25\2022-02-14T10-58-34.295960-data-reference-X_-12.000 mm-Y_12.000 mm.txt",r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\01 GaAs Wafer 25\2022-02-14T10-59-36.791953-data-reference-X_-12.000 mm-Y_12.000 mm.txt"]
film_ref = [r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\GaAs_Te 19073\2022-02-14T11-27-50.530909-100avg-reference-X_0.000 mm-Y_5.500 mm.txt",r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\GaAs_Te 19073\2022-02-14T11-31-58.990391-100avg-reference-X_0.000 mm-Y_6.000 mm.txt",r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\GaAs_Te 19073\2022-02-14T11-37-21.800393-100avg-reference-X_0.000 mm-Y_6.500 mm.txt",r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\GaAs_Te 19073\2022-02-14T11-59-17.415981-100avg-reference-X_0.000 mm-Y_8.000 mm.txt",r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\GaAs_Te 19073\2022-02-14T12-07-52.595976-100avg-reference-X_0.000 mm-Y_8.500 mm.txt",r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\GaAs_Te 19073\2022-02-14T12-17-07.067968-100avg-reference-X_0.000 mm-Y_9.000 mm.txt",r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\GaAs_Te 19073\2022-02-14T12-26-20.233786-100avg-reference-X_0.000 mm-Y_9.500 mm.txt",r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\GaAs_Te 19073\2022-02-14T12-36-12.274052-100avg-reference-X_0.000 mm-Y_10.000 mm.txt",r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\GaAs_Te 19073\2022-02-14T12-46-05.339292-100avg-reference-X_0.000 mm-Y_10.500 mm.txt",r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\GaAs_Te 19073\2022-02-14T12-56-35.543309-100avg-reference-X_0.000 mm-Y_11.000 mm.txt"]
film_sam = [r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\GaAs_Te 19073\2022-02-14T15-16-52.261847-100avg-sample-X_33.000 mm-Y_17.500 mm.txt",r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\GaAs_Te 19073\2022-02-14T15-16-33.073854-100avg-sample-X_32.500 mm-Y_17.500 mm.txt",r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\GaAs_Te 19073\2022-02-14T15-16-13.827848-100avg-sample-X_32.000 mm-Y_17.500 mm.txt",r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\GaAs_Te 19073\2022-02-14T15-15-54.988855-100avg-sample-X_31.500 mm-Y_17.500 mm.txt",r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\GaAs_Te 19073\2022-02-14T15-15-35.983856-100avg-sample-X_31.000 mm-Y_17.500 mm.txt",r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\GaAs_Te 19073\2022-02-14T15-15-17.033856-100avg-sample-X_30.500 mm-Y_17.500 mm.txt",r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\GaAs_Te 19073\2022-02-14T15-15-17.033856-100avg-sample-X_30.500 mm-Y_17.500 mm.txt",r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\GaAs_Te 19073\2022-02-14T15-15-17.033856-100avg-sample-X_30.500 mm-Y_17.500 mm.txt",r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\GaAs_Te 19073\2022-02-14T15-15-17.033856-100avg-sample-X_30.500 mm-Y_17.500 mm.txt",r"C:\Users\Denij Giri\Desktop\Conductivity\Mariel\2022_02_14\GaAs_Te 19073\2022-02-14T15-15-17.033856-100avg-sample-X_30.500 mm-Y_17.500 mm.txt"]
  



#sigma1 = []

for i in range(5):
    sub_sams = sub_sam[i]
    sub_refs = sub_ref[i]
    film_sams = film_sam[i]
    film_refs = film_ref[i]
    
    freqs,sigma, R, K , angle_sin, Real_part, phi_diff,angle_sin, imag_part,phi_sub, phi_film = analysis(film_sams, film_refs, sub_sams, sub_refs)
    #plt.plot(freqs/10**12,phi_film)
    #plt.figure()
    #plt.plot(freqs/10**12,sigma.real, color = 'blue', label = 'Real part')
    #plt.plot(freqs/10**12,sigma.imag, color = 'Red', label = 'Imaginary part')
    #plt.figure()
    #plt.plot(freqs/10**12,R,label = 'frequency versus R')
    #plt.legend()
    #plt.figure()
    #plt.plot(freqs/10**12,K,label ='frequency versus K')
    #plt.legend()
    plt.figure()
    plt.scatter(freqs/10**12,np.unwrap(phi_sub),label ='frequency versus angle')
    plt.scatter(freqs/10**12,np.unwrap(phi_film),label ='frequency versus angle')
    #plt.figure()
    #plt.plot(freqs/10**12,phi_diff)
    #plt.figure()
    #plt.plot(freqs/10**12,np.unwrap(phi_diff))
    
    
    #print(phi_sub)
    #print(np.mean(phi_sub))
    
   
    
    #print(np.std(phi_sub))

    
    
    #plt.figure()
    #plt.plot(freqs/10**12, phi_sub)
    #plt.fill_between(freqs/10**12, phi_sub + np.std(phi_sub), phi_sub - np.std(phi_sub), color = 'Grey', alpha = 0.5, label = 'phi_sub')
    #plt.legend()
    
    time_shift = phi_film / (2 * np.pi * freqs)
    print(time_shift/10**-12)
    
    time_shift2 = phi_sub / (2 * np.pi * freqs)
    print(time_shift2/10**-12)
    #print(phi_film)
    #print(np.mean(phi_film))
    #print(np.std(phi_film))
    
    #plt.figure()
    #plt.plot(freqs/10**12, phi_film)
    #plt.fill_between(freqs/10**12, phi_film + np.std(phi_film), phi_film - np.std(phi_film),color = 'grey' , alpha = 0.5, label = 'phi_film')
    #plt.legend()
    
    
    
   #print(np.std(phi_film))    
    #plt.xlabel('Frequency[THz]', weight = 'bold')
    #plt.ylabel('Conductivity[Ω−1cm-1]', weight = 'bold')
    #plt.legend()
    #plt.figure()
    #plt.plot(freqs/10**12, Real_part, label = 'Frequency versus real part')
    #plt.legend()
    
#average = np.mean(sigma1, axis = 0)
    





#plt.figure()
#plt.plot(freqs/10**12, average.real , color = 'Blue')
#plt.title('Conductivity(Real Part)', weight = 'bold')
#plt.xlabel('Frequency[THz]', weight = 'bold')
#plt.ylabel('Conductivity[Ω−1cm-1]', weight = 'bold')
 
#plt.figure()
#plt.plot(freqs/10**12, average.imag, color = 'Red')
#plt.title('Conductivity(Imaginary Part)', weight = 'bold')
#plt.xlabel('Frequency[THz]', weight = 'bold')
#plt.ylabel('Conductivity[Ω−1cm-1]', weight = 'bold')

