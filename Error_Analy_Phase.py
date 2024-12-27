# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:08:24 2024

@author: Denij Giri
"""


import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import re
from pathlib import Path
import matplotlib.dates as mdates

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
    fourier_dp = np.abs(fourier)
    
    return np.angle(fourier)
    

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
Date = []
Sub_ref = []

for i, (data_list1,data_list2, data_list3) in enumerate(zip(sub_sams,film_sams, sub_refs)):
      a_sub = frequency_domain(data_list1)
      b_sub = frequency_domain(data_list2)
      date = extract_date_time_from_file(data_list3)
      sub_ref = frequency_domain(data_list3)
      Date.append(date)
      Sub_ref.append(sub_ref)
      sub_amp.append(a_sub)
      sam_amp.append(b_sub)
      
      
      
time_only = [dt.time() for dt in Date]  
date_for_plot = [datetime.combine(datetime(2024, 10, 18), t) for t in time_only]
times_mpl = mdates.date2num(date_for_plot)


plt.plot(times_mpl,Sub_ref, color ='blue')
plt.scatter(times_mpl,Sub_ref, color = 'blue')
plt.tick_params(axis = 'x', which = 'both', top ='True',bottom = 'True',direction = 'in', labelsize = 14)
plt.tick_params(axis = 'y', which = 'both', left ='True',right = 'True',direction = 'in', labelsize = 14)
plt.xticks(fontsize = 10, fontweight = 'bold')
plt.yticks(fontsize = 14, fontweight = 'bold')

plt.annotate('Phase difference', xy = (mdates.date2num(datetime(2024,10,18,14,38,15)),0.86), xytext = (mdates.date2num(datetime(2024,10,18,14,38,15 )),0.89), color = 'blue', arrowprops = dict(facecolor='blue', color ='blue', arrowstyle='->'), fontsize =14, fontweight = 'bold')

plt.xlabel('Measurement time', weight = 'bold', fontsize = 14)
plt.ylabel('Phase difference[radians]', weight = 'bold', fontsize = 14)
plt.title('Phase difference Vs Measurement time[2THz]', weight = 'bold', fontsize = 14)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=1))  
#plt.legend(prop = {'size' : 14, 'weight':'bold'})
plt.show()


     
sub_amp_diff = []
sam_amp_diff = []

for i in range(1,len(sub_amp)):
    sub_diff = sub_amp[i] - sub_amp[i-1]
    sam_diff =  sam_amp[i] - sam_amp[i-1]
    sub_amp_diff.append(sub_diff)
    sam_amp_diff.append(sam_diff)
    

    
    
    
    






