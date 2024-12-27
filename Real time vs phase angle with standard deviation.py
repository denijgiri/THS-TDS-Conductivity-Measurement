# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:16:09 2024

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
    freqs_range = (freqs >= 0.99) * (freqs <= 1)
    freqs = freqs[freqs_range]
    phase_angle = phase_angle[freqs_range]
    
    return phase_angle 
        
    

data_dir = r'C:\Users\Denij Giri\Desktop\Conductivity\Silver Sample\sample3\img0'
data_path = pathlib.Path(data_dir)
data_list = list(data_path.glob("*"))

Phase_angle_sam = []
Date_time_sam = []

Phase_angle_ref = []
Date_time_ref = []


for file in data_list:
    if '-sam-' in str(file):
     date_time = extract_date_time_from_file(file)
     phase_angle = phase_calculation(file)
     Date_time_sam.append(date_time)
     Phase_angle_sam.append(phase_angle)
    else:
     date_time = extract_date_time_from_file(file)
     phase_angle = phase_calculation(file)
     Date_time_ref.append(date_time)
     Phase_angle_ref.append(phase_angle)
        
    
    

standard_dev = np.std(Phase_angle_ref) 
print(standard_dev)

print(np.mean(Phase_angle_ref))
print(np.min(Phase_angle_ref))
print(np.max(Phase_angle_ref))
average_ref = np.mean(Phase_angle_ref)
average_sam = np.mean(Phase_angle_sam)


plt.figure()
plt.plot(Date_time_sam,Phase_angle_sam, color = 'orange',label = 'Sample Measurement')
plt.fill_between(Date_time_ref, average_ref + standard_dev, average_ref - standard_dev, alpha = 0.5, edgecolor='#CC4F1B', facecolor='#FF9848' )
plt.plot(Date_time_ref,Phase_angle_ref, color = 'Red', label = 'Reference Measurement')
plt.fill_between(Date_time_sam, average_sam + standard_dev, average_sam - standard_dev, alpha = 0.5, edgecolor='#CC4F1B', facecolor='#FF9848' )
plt.xlabel('Real Clock Time')
plt.ylabel('Phase angle in radians')
plt.legend()





    

