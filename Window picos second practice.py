# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 07:12:34 2024

@author: Denij Giri
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def window_correction(data_ad):
    data_td = np.loadtxt(data_ad)
    plt.plot(data_td[:,0],data_td[:,1] - data_td[0,1])
    win_width = 15
    t = data_td[:,0]
    #print(len(t))
    dt = np.mean(np.diff(data_td[:,0]))
    #print(np.diff(data_td[:,0]))
    #print(dt)
    win_width = int(win_width/dt)
    print(win_width)
    win_start = np.argmax(np.abs(data_td[:,1])) - win_width // 2
    print(np.argmax(data_td[:,1]))
    print(win_start)
    window_axis = signal.windows.tukey(win_width, alpha = 0.5)
    zero1 = np.zeros(win_start)
    print(len(zero1))
    window_axis = np.concatenate((zero1,window_axis))
    zero2 = np.zeros(len(t) - win_start - win_width)
    print(len(zero2))
    window_axis = np.concatenate((window_axis,zero2))
    plt.plot(data_td[:,0], window_axis * np.max(data_td[:,1]))
    
    
film_ref = r"C:\Users\Denij Giri\Desktop\Conductivity\Semiconductors\GaAsTe_wafer19073\2023-07-25T15-35-37.391340-100avg-reference-X_60.000 mm-Y_-10.000 mm.txt"
window_correction(film_ref)