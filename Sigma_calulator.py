# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 17:45:21 2024

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



base_path_sub = Path(r"C:\Users\Denij Giri\Desktop\Conductivity\Semiconductors\GaAs_wafer25_sub_remeasure")
sub_ref = list((base_path_sub/"Reference").glob("*.txt"))
sub_sam = list((base_path_sub/"Substrate").glob("*.txt"))
 
base_path_sample = Path(r"C:\Users\Denij Giri\Desktop\Conductivity\Semiconductors\GaAsTe_wafer19073_remeasure")
film_ref = list((base_path_sample/"Reference").glob("*.txt"))
film_sam = list((base_path_sample/"Sample").glob("*.txt"))


for i, (file1 , file2, file3, file4) in enumerate(zip(sub_ref, sub_sam, film_ref, film_sam)):
    sub_refs.append(file1),sub_sams.append(file2), film_refs.append(file3),film_sams.append(file4)


result = []
phi_diff = []
amp_error = []

a = []
b = []
x = []
d = []
e = []
f = []
g = []
h = []
y = []


t = (sub_refs, sub_sams, film_refs, film_sams)
for i , (sub_ref, sub_sam, film_ref, film_sam) in enumerate(zip(*t)):
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
    
    phi_sub =  phi_sub_ref_cd - phi_sub_sam_cd
    phi_sam =  phi_film_ref_cd - phi_film_sam_cd
    phi_dif =  phi_sub - phi_sam
    

    r_sub_sam_bd = np.abs(sub_sam_bd[:,1])
    r_sub_ref_bd = np.abs(sub_ref_bd[:,1])
    r_film_sam_bd = np.abs(film_sam_bd[:,1])
    r_film_ref_bd = np.abs(film_ref_bd[:,1])
    
    delta_sub = phi_sub_ref_cd - phi_sub_sam_cd
    delta_sam =  phi_film_ref_cd - phi_film_sam_cd  
    
    freqs = sub_sam_bd[:,0].real

  
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
    freqs_range = (freqs>=1.9999*10**12)*(freqs< 2.01*10**12)
    
    
    
    sigma = (T_sub[:,1] - T_film[:,1]) * epsilon_0 * c * (1+n)/((T_film[:,1])*d_film)
    
    r_sub_sam_bd = np.array(r_sub_sam_bd)
    r_sub_ref_bd = np.array(r_sub_ref_bd)
    r_film_sam_bd = np.array(r_film_sam_bd)
    r_film_ref_bd = np.array(r_film_ref_bd)
    n = np.array(n)
    phi_sub_sam_cd  = np.array(phi_sub_sam_cd)
    phi_sub_ref_cd  = np.array(phi_sub_ref_cd)
    phi_film_sam_cd = np.array(phi_film_sam_cd)
    phi_film_ref_cd = np.array(phi_film_ref_cd)
    

    
    a.append(r_sub_sam_bd)
    b.append(r_sub_ref_bd)
    x.append(r_film_sam_bd)
    d.append(r_film_ref_bd)
    e.append(n)
    f.append(phi_sub_sam_cd)
    g.append(phi_sub_ref_cd)
    h.append(phi_film_sam_cd)
    y.append(phi_film_ref_cd)
  

     
a = np.array(a)
a = np.mean(a,axis=0)
a = a[freqs_range]
print(a)
b = np.array(b)
b = np.mean(b,axis=0)
b = b[freqs_range]
print(b)
x = np.array(x)
x = np.mean(x,axis=0)
x = x[freqs_range]
print(x)
d = np.array(d)
d = np.mean(d,axis=0)
d = d[freqs_range]
print(d)
e = np.array(e)
e = np.mean(e,axis=0)
e = e[freqs_range]
print(e)
f = np.array(f)
f = np.mean(f,axis=0)
f = f[freqs_range]
print(f)
g = np.array(g)
g = np.mean(g,axis=0)
g = g[freqs_range]
print(g)
h = np.array(h)
h = np.mean(h,axis=0)
h = h[freqs_range]
print(h)
y = np.array(y)
y = np.mean(y,axis=0)
y = y[freqs_range]
print(y)

   
    
