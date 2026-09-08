# -*- coding: utf-8 -*-

import math
from pathlib import Path
import pandas as pd
import numpy as np

from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

def load_rx_data(rx_path, header_name):
    header_row = 0
    with open(rx_path, 'r') as f:
        for idx, line in enumerate(f):
            if header_name in line:
                header_row = idx
                break
    rx_data = pd.read_csv(rx_path, skiprows=header_row) # avoids preamble
    data = rx_data[header_name].to_numpy()
    return data

def load_names(rx_path):
    filename = Path(rx_path).stem
    # split by underscore
    parts = filename.split("_")
    # extract names etc
    sample_name = parts[0]
    freq = parts[-2]
    extension = parts[-1]
    return [sample_name, freq, extension, None, None, None]
    
def find_peaks_troughs_detrended(array):
    window_size = 100
    baseline = uniform_filter1d(array, size=window_size)

    detrended = array - baseline
    
    prominence_auto = np.std(detrended)
    min_dist = 17
    
    peak_indices, _ = find_peaks(detrended, 
                              distance=min_dist,
                              prominence=prominence_auto)
    
    if len(peak_indices) == 0:
        return np.array([]), 0
    
    start_idx = peak_indices[0]
    end_idx = peak_indices[-1]
    
    invert_mid_arr = -detrended[start_idx:end_idx]
    rel_trough_indices, _ = find_peaks(invert_mid_arr,
                                distance=min_dist,
                                prominence=prominence_auto)
    trough_indices = rel_trough_indices + np.full_like(rel_trough_indices, 
                                                       start_idx)
    all_event_indices = np.sort(np.concatenate((peak_indices, trough_indices)))
    return all_event_indices

def find_ptp_res(peak_indices, res):
    amplitudes = []
    for i in range(len(peak_indices)-1):
        ptp = abs(res[peak_indices[i]]-res[peak_indices[i+1]])
        amplitudes.append(ptp)
    return np.array(amplitudes)

def find_midline(peak_indices, res):
    midline = []
    for i in range(len(peak_indices)-1):
        start_idx = peak_indices[i]
        end_idx = peak_indices[i+1]
        midline.append(0.5 * (res[start_idx] + res[end_idx]))
    return np.array(midline)

def to_4sf(num):
    if num == 0:
        return "0.000"
    
    # Calculate the scale / order of magnitude of the number
    magnitude = math.floor(math.log10(abs(num)))
    decimals = 3 - magnitude
    
    # Force fixed-point format using the calculated decimals
    if decimals < 0:
        return f"{num:.0f}"  # Large integers don't need trailing decimals
    return f"{num:.{decimals}f}"
