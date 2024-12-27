# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:11:18 2024

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
    amplitude = data_td[:,1]
    
    fourier = np.fft.fft(data_td[:,1])
    phase_angle = np.angle(fourier)
    n = data_td[:,1].size
    timestep = 5 * 10 ** -14
    freqs = np.fft.fftfreq(n, d = timestep)
    freqs = freqs / 10 **12
    freqs_range = (freqs >= 0.99) * (freqs <= 1)
    freqs = freqs[freqs_range]
    phase_angle = phase_angle[freqs_range]
    
    return amplitude 
        
    

data_dir = r'C:\Users\Denij Giri\Desktop\Conductivity\Silver Sample\sample3\img0'
data_path = pathlib.Path(data_dir)
data_list = list(data_path.glob("*"))

Phase_angle_sam = []
Date_time_sam = []

Phase_angle_ref = []
Date_time_ref = []


for data_files in data_list:
    if 'ref' in str(data_files):
     date_time = extract_date_time_from_file(data_files)
     phase_angle = phase_calculation(data_files)
     Date_time_ref.append(date_time)
     Phase_angle_ref.append(phase_angle)
    else:
     date_time = extract_date_time_from_file(data_files)
     phase_angle = phase_calculation(data_files)
     Date_time_sam.append(date_time)
     Phase_angle_sam.append(phase_angle)   
     
phase_difference_reference = []
phase_difference_sample = []

for i in range(1,len(Phase_angle_ref)):
  phase_difference_ref = np.abs(Phase_angle_ref[i]) - np.abs(Phase_angle_ref[i-1])
  phase_difference_reference.append(phase_difference_ref)

for i in range(1,len(Phase_angle_sam)):
    phase_difference_sam = np.abs(Phase_angle_sam[i]) - np.abs(Phase_angle_sam[i-1])
    phase_difference_sample.append(phase_difference_sam)

  

  



#print(phase_difference_sample)     
print(np.max(np.abs(phase_difference_sample)))
print(np.max(np.abs(phase_difference_reference)))


#print(phase_difference_sample)
#print(np.mean(phase_difference_sample))
    

plt.figure()
plt.plot(Date_time_sam,Phase_angle_sam, color = 'orange',label = 'Sample Measurement')
plt.plot(Date_time_ref,Phase_angle_ref, color = 'Red', label = 'Reference Measurement')
plt.xlabel('Real Clock Time')
plt.ylabel('Phase angle in radians')
plt.legend()

