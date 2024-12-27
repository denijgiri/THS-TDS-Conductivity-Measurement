# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 21:57:37 2024

@author: Denij Giri
"""


import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import re
import pathlib

def extract_date_time_from_file(file_list):
    pattern = r'((\d{4})-(\d{2})-(\d{2})T(\d{2})-(\d{2})-(\d{2}\.\d+))'
    file_list = str(file_list)
    match = re.search(pattern, file_list)

    if match:
        date_time_str = match.group(1)
        time_part_dt = date_time_str.split('T')[0]
        years = int(time_part_dt.split('-')[0])
        month = int(time_part_dt.split('-')[1])
        day = int(time_part_dt.split('-')[2])
        time_part_dt_t = date_time_str.split('T')[1]
        hours = int(time_part_dt_t.split('-')[0])
        minutes = int(time_part_dt_t.split('-')[1])
        seconds = float(time_part_dt_t.split('-')[2])
        seconds = int(seconds)
        
        dt = datetime(years,month,day, hours, minutes, seconds)
        
        return dt
    
    
def phase_calculation(data_ad):
    data_td = np.loadtxt(data_ad)
    fourier = np.fft.fft(data_td[:,1])
    phase_angle = np.angle(fourier)
    n = data_td[:,1].size
    timestep = 5 * 10 ** -14
    freqs = np.fft.fftfreq(n, d = timestep)
    freqs = freqs / 10 **12
    freqs_range = (freqs >= 0) * (freqs <= 3)
    freqs = freqs[freqs_range]
    phase_angle = phase_angle[freqs_range]
    
    return phase_angle,freqs
        
    

data_dir = r'C:\Users\Denij Giri\Desktop\Conductivity\Silver Sample\sample3\img0'
data_path = pathlib.Path(data_dir)
data_list = list(data_path.glob("*"))

Phase_angle_sam = []
Date_time_sam = []

Phase_angle_ref = []
Date_time_ref = []
freqs_ref = []


for data_files in data_list:
    if 'ref' in str(data_files):
     date_time = extract_date_time_from_file(data_files)
     phase_angle,freqs = phase_calculation(data_files)
     Date_time_ref.append(date_time)
     Phase_angle_ref.append(phase_angle)
     freqs_ref.append(freqs)
    else:
     date_time = extract_date_time_from_file(data_files)
     phase_angle = phase_calculation(data_files)
     Date_time_sam.append(date_time)
     Phase_angle_sam.append(phase_angle) 
     
     
#print(len(Phase_angle_sam))
#print(len(Phase_angle_ref))
#print(len(freqs_ref))

     
#Phase_angle_sam_arr = np.array([Phase_angle_sam])
#Phase_angle_ref_arr = np.array([Phase_angle_ref])




  

#print(difference_phase)
phase_std = np.std(Phase_angle_ref)
print(np.mean(phase_std))
delf_f = np.sqrt((phase_std/Phase_angle_ref)**2)
print(len(delf_f))
print(np.mean(delf_f))



plt.figure()
plt.plot(freqs_ref,delf_f)
plt.xlabel('Frequency')
plt.ylabel('Error in refractive index')


#print(phase_difference_reference)     
#print(np.mean(np.abs(phase_difference_reference)))

#print(phase_difference_sample)
#print(np.mean(phase_difference_sample))
    

#plt.figure()
#plt.plot(Date_time_sam,Phase_angle_sam, color = 'orange',label = 'Sample Measurement')
#plt.plot(Date_time_ref,Phase_angle_ref, color = 'Red', label = 'Reference Measurement')
#plt.xlabel('Real Clock Time')
#plt.ylabel('Phase angle in radians')
#plt.legend()

