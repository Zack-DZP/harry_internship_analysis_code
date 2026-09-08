# -*- coding: utf-8 -*-

import numpy as np
import math

# low pass filter
def simple_filter(x, alpha):
   y = []
   y.append(x[0]) # edge case at start point
   for i in range(1, len(x)):
       y.append(alpha * y[i-1] + (1 - alpha) * x[i])
   return np.array(y)

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
        