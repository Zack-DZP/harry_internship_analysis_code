# -*- coding: utf-8 -*-

import numpy as np
from scipy.signal import find_peaks

# low pass filter
def simple_filter(x, alpha):
   y = []
   y.append(x[0]) # edge case at start point
   for i in range(1, len(x)):
       y.append(alpha * y[i-1] + (1 - alpha) * x[i])
   return y

# differentiate
def differentiator(x, y):
    z = [0]
    for i in range(1, (len(y)-1)):
        z.append((y[i+1]-y[i-1])/(x[i+1]-x[i-1]))
    z.append(0)
    return z

# peak detection for differentiated signal
def detect_events(fil_ffdot_array, cftimes_array, threshold, filffdot_sigma):
    # positive peak indices
    pos_indices, _ = find_peaks(
        fil_ffdot_array,  
        height=threshold * filffdot_sigma,
        distance=20
    )
    positive_peak_times = cftimes_array[pos_indices] # positive peak times

    # negative peak indices
    neg_indices, _ = find_peaks(
        - fil_ffdot_array,
        height= threshold * filffdot_sigma,
        distance=20
    )
    negative_peak_times = cftimes_array[neg_indices] # negative peak times

    # combine and sort times to use for resistance
    all_peak_times = np.sort(np.concatenate((positive_peak_times, negative_peak_times))).tolist()
 #  all_peak_times = np.insert(all_peak_times, 0, 0.0)
    # combine and sort indices to use for force
    all_peak_indices = np.sort(np.concatenate((pos_indices, neg_indices))).tolist()
  # all_peak_indices = np.insert(all_peak_indices, 0, 0.0)

    
    return all_peak_indices, all_peak_times
    
