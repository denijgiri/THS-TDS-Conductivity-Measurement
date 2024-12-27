# -*- coding: utf-8 -*-
"""
Created on Mon May  6 13:01:56 2024

@author: Denij Giri
"""

import numpy as np
import re
import pathlib
import matplotlib.pyplot as plt 

def extract_coordinate_from_file_path(file_list):
    pattern = r'X_([-+]?\d+\.\d+) mm-Y_([-+]?\d+\.\d+) mm'
    file_list = str(file_list)
    
    match = re.search(pattern, file_list)
    if match:
        x_coordinate = float(match.group(1))
        y_coordinate = float(match.group(2))
        return x_coordinate, y_coordinate

def convert_index(x_int,y_int):
    i = int(x_int) - 20
    j = int(y_int) + 15
    return i, j

def cal_p2p(file):
    data = np.loadtxt(file)
    amp = data[:,1]
    diff_amp = np.max(amp) - np.min(amp)
    return diff_amp

data_dir = r'C:\Users\Denij Giri\Desktop\Conductivity\Silver Sample\sample3\img0'
data_path = pathlib.Path(data_dir)
data_list = list(data_path.glob("*"))
empt_array = np.zeros((31,28))

for file in data_list:
    if 'ref' in str(file) :
        continue 
    x, y = extract_coordinate_from_file_path(file)
    i,j = convert_index(x,y)
    empt_array[i,j]= cal_p2p(file)
    n = empt_array
        
#n[15,20] = 0
#n = np.transpose(n)
#n = np.rot90(n)
extent = [20,50,-15,12]

#print(n)
plt.figure("Image")
#plt.xlim(30,50)
#plt.ylim(-15,12)
plt.imshow(n,extent = extent, origin = 'upper')
#plt.imshow(n, origin = 'upper')
plt.xlabel('X_Coordinate')
plt.ylabel('Y_Coordinate')
#plt.ylim(bottom = -15, top = 12
plt.colorbar()
plt.show() 







