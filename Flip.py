# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:21:00 2024

@author: Denij Giri
"""

import numpy as np
import matplotlib.pyplot as plt

data_txt1 = np.loadtxt(r"C:\Users\Denij Giri\Desktop\Conductivity\Remeasure GaAs_Te\Wafer_25_and_wafer_19073\2024-09-26T18-10-32.216690-data-Empty-164.000 mm.txt")
data_txt2 = np.loadtxt(r"C:\Users\Denij Giri\Desktop\Conductivity\Remeasure GaAs_Te\Wafer_25_and_wafer_19073\2024-09-26T17-25-59.171806-data-Reference-100.000 mm.txt")
data_sub1 = np.loadtxt(r"C:/Users/Denij Giri/Desktop/Conductivity/Remeasure GaAs_Te/Wafer_25_and_wafer_19073/2024-09-26T17-30-43.398553-data-SamSub-141.000 mm.txt")
data_sam1 = np.loadtxt(r"C:/Users/Denij Giri/Desktop/Conductivity/Remeasure GaAs_Te/Wafer_25_and_wafer_19073/2024-09-26T17-31-40.937663-data-SamFilm-186.000 mm.txt")

data_sam_diff = np.loadtxt(r"C:\Users\Denij Giri\Desktop\Conductivity\Semiconductors\GaAsTe_wafer19073\2023-07-25T15-37-19.935720-100avg-sample-X_25.000 mm-Y_5.000 mm.txt")


plt.figure()
plt.plot(data_sub1[:,0],np.flip(data_sub1[:,1],axis =0), label = 'Substrate')
plt.plot(data_sam1[:,0],np.flip(data_sam1[:,1],axis=0), label = 'Sample')
plt.plot(data_txt1[:,0],np.flip(data_txt1[:,1],axis =0), label = 'reference')
plt.legend()

plt.figure()
plt.plot(data_sam_diff[:,0],data_sam_diff[:,1], label = 'Smaple_Previous')
plt.legend()



n = data_txt1[:,0].size
d = 0.05
freqs = np.fft.rfftfreq(n,d)
fourier1 = np.fft.rfft(data_txt1[:,1])
fourier2 = np.fft.rfft(data_txt2[:,1])
fourier3 = np.fft.rfft(data_sub1[:,1])
fourier4 = np.fft.rfft(data_sam1[:,1])

plt.figure()
plt.plot(freqs, 20*np.log10(fourier1), label = 'Refernece')
plt.plot(freqs, 20*np.log10(fourier3), label = 'Substrate')
plt.plot(freqs, 20*np.log10(fourier4), label = 'Sample')
plt.legend()


