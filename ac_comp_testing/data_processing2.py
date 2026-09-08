# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def load_ac_data(force_path, rx_path):
    force_data = pd.read_csv(force_path)
    # force data
    force = force_data[r'Channel 1 (Force)'].tolist()
    # force time
    ftimes = force_data[r'Time (Grams)'].tolist()
    # dynamic row finder for rx data
    header_row = 0
    with open(rx_path, 'r') as f:
        for idx, line in enumerate(f):
            if r"R(OHM)" in line:
                header_row = idx
                break
    rx_data = pd.read_csv(rx_path, skiprows=header_row) # avoids preamble
    res = rx_data[r"R(OHM)"].tolist()
    react = rx_data[r"X(OHM)"].tolist()
    # generate rx time data - assume evenly spaced.
    tot_time = ftimes[-1]
    rxtimes = np.linspace(0, tot_time, len(rx_data)).tolist()
    
    return force, ftimes, res, react, rxtimes

def calculate_avg_force(all_peak_indices, filterforce):
    # find avg force
    avg_force = []
    for i in range(len(all_peak_indices)-1):
        forcesum = 0
        for j in range(all_peak_indices[i], all_peak_indices[i+1]):
            forcesum += filterforce[j]
        avg_force.append(forcesum / (all_peak_indices[i+1] - all_peak_indices[i]))
    return avg_force

def calculate_avg_resistance(all_peak_times, rtimes, relres):
    # convert to np arrays
    rtimes_array = np.array(rtimes)

    # use np sorted search
    avg_relres = []
    for i in range(len(all_peak_times)-1):
        time1 = all_peak_times[i] # start time
        time2 = all_peak_times[i+1] # end time
        
        index1 = np.searchsorted(rtimes_array, time1) # start index
        index2 = np.searchsorted(rtimes_array, time2) # end index
        
        window = relres[index1:index2]
        
        avg_relres.append(np.mean(window))
        
    avg_relres = [float(x) for x in avg_relres] # converting back to standard floats for ease of reading
    return avg_relres

def split_curve(avg_force, avg_relres):
    # convert to np arrays
    f_arr = np.array(avg_force)
    r_arr = np.array(avg_relres)
    indexmax = np.argmax(f_arr)
    fmax = f_arr[indexmax]
    
    f_up = f_arr[:indexmax+1]
    f_down = f_arr[indexmax:]
    
    r_up = r_arr[:indexmax+1]
    r_down = r_arr[indexmax:]
    
    return f_up, f_down, r_up, r_down, fmax
    

def hysteresis_value(avg_force, avg_relres, fmin=10.0):

    # convert to np arrays
    f_arr = np.array(avg_force)
    r_arr = np.array(avg_relres)
    r_arr = r_arr - np.min(r_arr)
    indexmax = np.argmax(f_arr)
    fmax = f_arr[indexmax]
    
    f_up = f_arr[:indexmax+1]
    f_down = f_arr[indexmax:]
    
    r_up = r_arr[:indexmax+1]
    r_down = r_arr[indexmax:]
        
    f_down_flip = np.flip(f_down)
    r_down_flip = np.flip(r_down)
    
    uniaxis = np.linspace(fmin, fmax, 100)
    
    r_up_i = np.interp(uniaxis, f_up, r_up)
    r_down_i = np.interp(uniaxis, f_down_flip, r_down_flip)
    
    areaup = np.trapezoid(r_up_i, uniaxis)
    areadown = np.trapezoid(r_down_i, uniaxis)
    
    ratio = areadown / areaup
    
    return ratio



