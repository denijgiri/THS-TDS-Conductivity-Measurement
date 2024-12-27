# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 09:36:28 2024

@author: Denij Giri
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import re
from pathlib import Path

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
    
def frequency_domain(data_ad):
    data_td = np.loadtxt(data_ad)
    fourier = np.fft.fft(data_td[:,1])
    n = data_td[:,1].size
    timestep = 0.05
    freqs = np.fft.fftfreq(n, d = timestep)
    pos_freq = (freqs > 1.999999999) * (freqs < 2.01)
    freqs = freqs[pos_freq]
    fourier = fourier[pos_freq]
    fourier_dp = np.abs(20*np.log(fourier))
    
    return fourier_dp
    

base_path_sub = Path(r"C:\Users\Denij Giri\Desktop\Conductivity\Semiconductors\GaAs_wafer25_sub_remeasure")
sub_ref = list((base_path_sub/"Reference").glob("*.txt"))
sub_sam = list((base_path_sub/"Substrate").glob("*.txt"))
 
base_path_sample = Path(r"C:\Users\Denij Giri\Desktop\Conductivity\Semiconductors\GaAsTe_wafer19073_remeasure")
film_ref = list((base_path_sample/"Reference").glob("*.txt"))
film_sam = list((base_path_sample/"Sample").glob("*.txt"))


sub_refs, sub_sams, film_refs, film_sams = [],[],[],[]

for i, (file1 , file2, file3, file4) in enumerate(zip(sub_ref, sub_sam, film_ref, film_sam)):
    sub_refs.append(file1),sub_sams.append(file2), film_refs.append(file3),film_sams.append(file4)

    
 


sub_amp = []
sam_amp = []
date1 = []



for i, (data_list1,data_list2) in enumerate(zip(sub_sams,film_sams)):
      a_sub = frequency_domain(data_list1)
      b_sub = frequency_domain(data_list2)
      date = extract_date_time_from_file(data_list1)
      date1.append(date)
      sub_amp.append(a_sub)
      sam_amp.append(b_sub)
      

plt.scatter(date1,sam_amp)

sub_amp_diff = []
sam_amp_diff = []
print(np.mean(sub_amp))
for i in range(1,len(sub_amp)):
    sub_diff = sub_amp[i] - sub_amp[i-1]
    sam_diff =  sam_amp[i] - sam_amp[i-1]
    sub_amp_diff.append(sub_diff)
    sam_amp_diff.append(sam_diff)
     
print(np.max(sub_amp_diff))
print(np.max(sam_amp_diff))
    


    
    
    






