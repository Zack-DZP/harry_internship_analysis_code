# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def load_force_data(filepath, ftdelay):
    force_data = pd.read_csv(filepath)
    # force data
    force = force_data['Channel 1 (Force)'].tolist()
    # force time
    ftimes = force_data['Time (Grams)'].tolist()

    # correct force times
    cftimes = []
    cftimes = ftimes + np.full(len(ftimes), ftdelay)
    return force, cftimes

def load_resistance_data(filepath):
    header_row = 0
    with open(filepath, 'r') as f:
        for idx, line in enumerate(f):
            if 'Channel 1 (Ω)' in line:
                header_row = idx
                break
    res_data = pd.read_csv(filepath, skiprows=header_row)
    # resistance data
    res = res_data['Channel 1 (Ω)'].tolist()
    # resistance times
    time_parts = res_data["Time (s)"].str.split(":", expand=True)
    minutes = pd.to_numeric(time_parts[0])
    seconds = pd.to_numeric(time_parts[1])
    raw_seconds = 60 * minutes + seconds
    timedrops = raw_seconds.diff() < 0
    hours = timedrops.cumsum()
    res_data["total_ms"] = 1000 * (3600 * hours + raw_seconds)
    res_data["elapsed_rtime"] = res_data["total_ms"] - res_data["total_ms"].iloc[0]
    rtimes = res_data["elapsed_rtime"].tolist()
        
    return res, rtimes

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

def fft_analysis(cftimes, force):
    t_uneven = 0.001*np.array(cftimes) # converts from ms to s
    force_raw = np.array(force)
    
    # interpolate to even time intervals (needed for fft)
    t_even = np.linspace(t_uneven[0], t_uneven[-1], len(t_uneven))
    force_even = np.interp(t_even, t_uneven, force_raw)

    # spacing
    n = len(force_even)
    dt = (t_even[-1] - t_even[0]) / (n - 1)  # average uniform time interval

    fft_vals = np.fft.fft(force_even)
    fft_freqs = np.fft.fftfreq(n, d=dt)

    # mask to only pos frequencies
    pos_mask = fft_freqs >= 0
    frequencies = fft_freqs[pos_mask]
    magnitude = np.abs(fft_vals[pos_mask]) / n
    
    return frequencies, magnitude



