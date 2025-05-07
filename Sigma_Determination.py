# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:03:29 2024

@author: Denij Giri
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path
import os 
from scipy.optimize import curve_fit
from scipy.constants import e

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
 

phi_difff = []

sigma_all = []
    
t = (sub_refs, sub_sams, film_sams, film_refs)
for i , (sub_ref, sub_sam, film_sam, film_ref) in enumerate(zip(*t)):
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
    phi_diff = phi_sub - phi_sam
    
   
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
    
    T_sub = np.array([freqs, sub_sam_ed[:,1] / sub_ref_ed[:,1]]).T
    T_film = np.array([freqs, film_sam_ed[:,1] / film_ref_ed[:,1]]).T
   
    c = 3 * 10 ** 8 * 100
    d_sub = 500 * 10 ** -6 * 100
    omega = 2*np.pi*freqs
    epsilon_0 = 8.854127 * 10 ** -12 / 100
    d_film = 700 * 10 **-9 * 100
    n = 1 + (c * delta_sub) / (omega*d_sub)
    #n = 3.8 * np.ones_like(n)

    sigma = (T_sub[:,1] - T_film[:,1]) * epsilon_0 * c * (1+n)/((T_film[:,1])*d_film)
    freqs = freqs/ 10**12
    
    freqs_range = (freqs >= 0.5) * (freqs< 2.5)
    phi_difff.append(phi_diff)
    sigma_all.append(sigma)
    
    
    #if i in [300,400,500]:
     #   plt.figure("Conductivity")
      #  plt.plot(freqs[freqs_range],sigma.real[freqs_range])
       # plt.plot(freqs[freqs_range],sigma.imag[freqs_range])
        
        #plt.figure("Phi_Difference")
        #plt.plot(freqs[freqs_range],phi_diff[freqs_range])
    
    
    
sigma_all = np.array(sigma_all)
sigma_avg = np.mean(sigma_all, axis = 0)


#die_func = 1 + (1j *sigma_all/(omega*epsilon_0))
#refrac = np.sqrt(die_func)
#refrac = np.mean(refrac,axis = 0)

#plt.figure("Refractive_idx")
#plt.plot(freqs[freqs_range],refrac.real[freqs_range])
#plt.plot(freqs[freqs_range],refrac.imag[freqs_range])


sdv_sigma = np.std(sigma_all, axis = 0)
sdv_sigma = sdv_sigma[freqs_range]

#plt.figure("Conductivity")
#plt.tick_params(axis = 'x', which = 'both', top ='True',bottom = 'True',direction = 'in', labelsize = 14)
#plt.tick_params(axis = 'y', which = 'both', left ='True',right = 'True',direction = 'in', labelsize = 14)
#plt.xticks(fontsize = 14, fontweight = 'bold')
#plt.yticks(fontsize = 14, fontweight = 'bold')

#plt.figure()
#plt.plot(freqs[freqs_range],sigma_avg.real[freqs_range])
#plt.errorbar(freqs[freqs_range],sigma_avg.real[freqs_range], yerr = sdv_sigma,color='red', errorevery = 1, elinewidth =0.5)
#plt.fill_between(freqs[freqs_range], (sigma_avg.real + sdv)[freqs_range], (sigma_avg.real - sdv)[freqs_range], color = 'grey')
#plt.figure()
#plt.plot(freqs[freqs_range],sigma_avg.imag[freqs_range])
#plt.errorbar(freqs[freqs_range],sigma_avg.imag[freqs_range], yerr = sdv_sigma,color='blue', errorevery = 1, elinewidth =0.5)
#plt.fill_between(freqs[freqs_range], (sigma_avg.imag + sdv)[freqs_range], (sigma_avg.imag - sdv)[freqs_range], color = 'orange')


#plt.annotate('Real part', xy = (1.5,280), xytext = (1.5,360),color ='red', arrowprops = dict(facecolor='red',color ='red', arrowstyle='->'), fontsize =14, fontweight = 'bold')
#plt.annotate('Imaginary part', xy = (1.5,90), xytext = (1.5,10),color ='blue', arrowprops = dict(facecolor='blue',color = 'blue',arrowstyle='->'), fontsize =14, fontweight = 'bold')

#plt.xlabel("Frequency[THz]",weight = 'bold',fontsize = 14 )
#plt.ylabel("Conductivity[S/cm]", weight = 'bold',fontsize = 14 )
#plt.title("Conductivity Vs Frequency", weight = 'bold',fontsize = 14)
#plt.legend(prop = {'size' : 14, 'weight':'bold'})

    
#phi_difff = np.array(phi_difff)
#phi_diff_avg = np.mean(phi_difff, axis = 0)
#y = phi_diff_avg
#sdv_phi = np.std(phi_difff, axis = 0)
#sdv_phi = sdv_phi[freqs_range]
#plt.figure("Phi_Difference")
#plt.plot(freqs[freqs_range], y[freqs_range], label = 'Phi_Difference')
#plt.errorbar(freqs[freqs_range], y[freqs_range], yerr = sdv_phi, errorevery = 1, elinewidth =0.5)
#plt.fill_between(freqs[freqs_range], (y + sdv)[freqs_range], (y-sdv)[freqs_range], color = 'grey')
#plt.xlabel("Frequency[THz]")
#plt.ylabel("Phi_Difference(Radians)")
#plt.title("Phi_Difference")
#plt.legend()



def drude_model(x,mobility,N,mass_eff):
    sigma_dc = N * mobility *e
    tau = mobility * mass_eff /(e)
    x = x * 1e8
    omega = 2 * np.pi * x
    sigma_cc = (sigma_dc /(1 - (1j*omega*tau)))
    return sigma_cc.real


#freq = np.linspace(0.5*10**12,2.5*10**12,1000)
freqs = freqs[freqs_range]
sigma_avg = sigma_avg[freqs_range]


initial_guess = [2500,1.6*10**18, 0.067*9.31*10**-31]
bounds = (0, [3000,1.9e18, 0.080*9.31e-31])

mob = []
con = []
mass = []
fitted_model = []
drude = []
Sdv_fitted = []

for i, freq in enumerate(freqs):
    freq_point = np.array([freq])
    sigma_point = np.array([sigma_avg.real[i]])

    popt, _ = curve_fit(drude_model, freq_point, sigma_point, p0=initial_guess, bounds=bounds)
    
    mob.append(popt[0])
    con.append(popt[1])
    mass.append(popt[2])
    
    
    fit_value = drude_model(freq_point,*popt)
    fitted_model.append(fit_value)
    #sdv_fitted = np.std(fit_value)
    #Sdv_fitted.append(sdv_fitted)
    
    drud_value = drude_model(freq_point,*initial_guess)
    drude.append(drud_value)

print(np.mean(mob))
print(np.mean(con))
print(np.mean(mass))
fitted_model = np.array(fitted_model)
fitted_model = np.mean(fitted_model,axis=1)
print(fitted_model.shape)
Sdv_fitted = np.std(fitted_model)
print(Sdv_fitted.shape)


plt.figure()
plt.plot(freqs,fitted_model, label = 'fitted_value', color = 'blue')
plt.plot(freqs, fitted_model + Sdv_fitted, linestyle = '--')
plt.plot(freqs,fitted_model - Sdv_fitted, linestyle = '--')

#plt.errorbar(freqs, fitted_model, yerr = Sdv_fitted, color = 'blue',errorevery = 1, elinewidth =0.5 )
#plt.errorbar(freqs,sigma_avg.real, yerr = sdv_sigma,color='red', errorevery = 1, elinewidth =0.5)
plt.plot(freqs,sigma_avg.real, label = 'Original condu', color = 'red')
#plt.plot(freqs,drude, label = 'Drude Model', color = 'Green')
plt.xlabel("Frequency[THz]")
plt.ylabel("Conductivity[S/cm]")
plt.legend()

    





#drude_cond = drude_model(freqs,*intial_guess)
#plt.figure("Drude Fit")
#plt.plot(freqs,drude_cond, color = 'red', label = 'Drude model')
#plt.legend()

#popt,pcov = curve_fit(drude_model, freqs, sigma_avg.real, p0 = intial_guess, bounds = bounds)
#print(popt)
#plt.plot(freqs,sigma_avg.real,label = 'Experimental data', color = 'blue')
#plt.legend()

#drude_cond_fitted = drude_model(freqs,*popt)
#drude_cond_fitted_sdv = np.std(drude_cond_fitted,axis = 0)
#plt.plot(freqs,drude_cond_fitted, color = 'green', label = 'fitted curve' )
#plt.errorbar(freqs[freqs_range],drude_cond_fitted[freqs_range],yerr = drude_cond_fitted_sdv, errorevery = 1, elinewidth =0.5)
#plt.xlabel('Frequency [THz]')
#plt.ylabel('Conductivity [S/cm]')
#plt.legend()



sigma_avg = sigma_avg.real[freqs_range]
freqs = freqs[freqs_range]
sigma_avg = sigma_avg.reshape(1,-1)
sigma_avg = sigma_avg.reshape(250,1)
print(sigma_avg.shape)

plt.figure("Conductivity")
#plt.tick_params(axis = 'x', which = 'both', top ='True',bottom = 'True',direction = 'in', labelsize = 14)
#plt.tick_params(axis = 'y', which = 'both', left ='True',right = 'True',direction = 'in', labelsize = 14)
#plt.xticks(fontsize = 14, fontweight = 'bold')
#plt.yticks(fontsize = 14, fontweight = 'bold')
plt.imshow(sigma_avg, aspect='auto',cmap = 'viridis')
plt.axis('off')
plt.colorbar(label = 's_value')
plt.show()

    
    
    
    
    
    
    
 
    
    



