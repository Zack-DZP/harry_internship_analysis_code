# -*- coding: utf-8 -*-

import numpy as np

from scipy.signal import find_peaks

# low pass filter
def simple_filter(x, alpha):
   y = []
   y.append(x[0]) # edge case at start point
   for i in range(1, len(x)):
       y.append(alpha * y[i-1] + (1 - alpha) * x[i])
   return np.array(y)

# differentiate
def differentiator(x, y):
    z = [0]
    for i in range(1, (len(y)-1)):
        z.append((y[i+1]-y[i-1])/(x[i+1]-x[i-1]))
    z.append(0)
    return z

def detect_events(fil_ffdot_array, cftimes_array, threshold, filffdot_sigma, distance=20):
    # positive peak indices
    pos_indices, _ = find_peaks(
        fil_ffdot_array,  
        height=threshold * filffdot_sigma,
        distance=distance
    )
    positive_peak_times = cftimes_array[pos_indices] # positive peak times

    # negative peak indices
    neg_indices, _ = find_peaks(
        - fil_ffdot_array,
        height= threshold * filffdot_sigma,
        distance=distance
    )
    negative_peak_times = cftimes_array[neg_indices] # negative peak times

    # combine and sort times to use for resistance
    all_peak_times = np.sort(np.concatenate((positive_peak_times, negative_peak_times))).tolist()
    # combine and sort indices to use for force
    all_peak_indices = np.sort(np.concatenate((pos_indices, neg_indices))).tolist()
    
    return all_peak_indices, all_peak_times

def discretise(data, indices, lb=0.8, ub=0.99):
    disc_data = []
    for i in range(len(indices)-1):
        index1 = round((1-lb) * indices[i] + lb * indices[i+1]) # start from 80% through the stage
        index2 = round((1-ub) * indices[i] + ub * indices[i+1]) # end at 99% (avoid the step)
        
        window = data[index1:index2]
        disc_data.append(np.mean(window))
    return np.array(disc_data)
