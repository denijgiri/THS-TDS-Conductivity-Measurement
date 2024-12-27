# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 16:17:06 2024

@author: Denij Giri
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import e

class Dude():
    def __init__(self, freq_range, mobility, N, mass_eff):
        self.freq_range  = freq_range
        self.mobility = mobility
        self.N = N
        self.mass_eff = mass_eff

  

    def plot_real(self, freq_range, mobility, N , mass_eff):
        self.sigma_DC =  self.N * self.mobility * e
        self.tau = self.mobility * self.mass_eff / (e * 10**4)
        self.omega = 2 * np.pi * self.freq_range
        self.sigma_cc = (self.sigma_DC / (1 -(1j *self.omega *self.tau)))

        
        plt.figure(figsize=(8, 6))
        plt.plot(self.freq_range / 10 **12, self.sigma_cc.real)
        plt.xlabel('Frequency (omega)')
        plt.ylabel('COnductivity')
        plt.title('Real Part of the conductivity vs Frequency')
        plt.grid(True)
        plt.show()
        return self.sigma_cc.real
        
    def plot_imag(self):
       self.sigma_DC =  self.N * self.mobility * e
       self.tau = self.mobility * self.mass_eff / (e * 10**4)
       self.omega = 2 * np.pi * self.freq_range
       self.sigma_cc = (self.sigma_DC / (1 -(1j *self.omega *self.tau)))

            
       plt.figure(figsize=(8, 6))
       plt.plot(self.freq_range, self.sigma_cc.imag)
       plt.xlabel('Frequency (omega)')
       plt.ylabel('COnductivity')
       plt.title('Imaginary Part of the conductivity vs Frequency')
       plt.grid(True)
       plt.show()
       return self.sigma_cc.imag
            
   
    
   