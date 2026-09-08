# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import ccc_data_processing
import ccc_utils

# expt information
sample = "S2"              # sample name 

# file paths
temp_path = r"C:\Users\DzpTe\OneDrive\Documents\harry\CCC_DATA\S2_Temp.csv"
volt_path = r"C:\Users\DzpTe\OneDrive\Documents\harry\CCC_DATA\S2_Volt.csv"

def ccc_processing(temp_path, volt_path,
                   i_incr=0.25, num_incr=12, width=10.3, depth=0.024,
                   alpha=0.8, threshold=8, distance=120):

    # extract csv data
    temp_diff, temp_times, temp_tot_raw = ccc_data_processing.load_temp_data(temp_path)
    volt, volt_times = ccc_data_processing.load_volt_data(volt_path)
    
    # filter all data to remove noise
    temp_diff = ccc_utils.simple_filter(temp_diff, alpha)
    temp_tot_raw = ccc_utils.simple_filter(temp_tot_raw, alpha)
    volt = ccc_utils.simple_filter(volt, alpha)
    
    # reinterpolate temp to volt timescale
    times = volt_times # rename - it is now a common scale
    
    
    # peak detection for volt
    volt_dot = ccc_utils.differentiator(times, volt)
    volt_dot = ccc_utils.simple_filter(volt_dot, alpha) # smooth it
    vd_sigma = np.std(volt_dot) # st dev
    peak_indices, _ = ccc_utils.detect_events(volt_dot, times, threshold, vd_sigma, distance)
    peak_indices = np.insert(peak_indices, 0, 0)
    peak_indices = np.append(peak_indices, len(times)-1)
    
    # generate current values
    current = np.array(0)
    if (len(peak_indices)-1) != num_incr+1:
        print("Error - number of increments doesn't line up with peak times")
    else:
        for i in range(len(peak_indices)-1):
            curr_value = i * i_incr
            current = np.append(current, np.full(peak_indices[i+1]-peak_indices[i], curr_value))
                
    # calculate resistance
    res = np.divide(volt, current, where=(current != 0), out=None)
    res = ccc_utils.simple_filter(res, alpha)
    
    # discretise data for T/I plots
    # find eqm temps - last 20% averaged
    temp = np.interp(times, temp_times, temp_diff)  # rename for lazy typing
    temp_tot = np.interp(times, temp_times, temp_tot_raw)
    discr_temp = ccc_utils.discretise(temp, peak_indices)
    discr_temp_tot = ccc_utils.discretise(temp_tot, peak_indices)
    # current
    discr_curr = np.linspace(0, (num_incr)*i_incr, num_incr+1)
    norm_curr = discr_curr / (width*depth) # normalised with area
                
    return discr_curr, norm_curr, discr_temp, discr_temp_tot, res, current, volt, times


if __name__ == '__main__':
    

    discr_curr, norm_curr, discr_temp, discr_temp_tot, res, current, volt, times = ccc_processing(temp_path, 
                                                                                                  volt_path)
    
    width=10.3
    depth=0.024
    
    # PLOTS
    plt.style.use('seaborn-v0_8-ticks')
    
    # show time series
    fig, ax1 = plt.subplots(layout="constrained")
    fig.canvas.manager.set_window_title(f"{sample} - CCC time series")
    line1, = ax1.plot(times, volt, color='lime', label='Voltage')
    line2, = ax1.plot(times, res, color='red', label='Resistance')
    ax1.set_title(f"{sample} - CCC time series")
    ax1.set_ylabel('Voltage (V) / Resistance (Ohms)')
    ax1.set_xlabel('Time (ms)')
    ax2 = ax1.twinx()
    line3, = ax2.plot(times, current, color='blue', label='Current')
    ax2.set_ylabel('Current (A)')
    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')
    
    # temperature vs current plot
    fig, ax3 = plt.subplots(layout="constrained")
    fig.canvas.manager.set_window_title(f"{sample} - Temperature vs current density")
    line4, = ax3.plot(norm_curr, discr_temp, color='red', label='Temperature above ambient')
    line5, = ax3.plot(norm_curr, discr_temp_tot, color='blue', label='Total temperature')
    ax3.set_title(f"{sample} - Temperature vs current density")
    ax3.set_ylabel('Temperature (deg C)')
    ax3.set_xlabel('Current density (A/mm^2)')
    lines2 = [line4, line5]
    labels2 = [l.get_label() for l in lines2]
    ax4 = ax3.secondary_xaxis('top', functions=(lambda x: x*width*depth, lambda x: x*width*depth))
    ax4.set_xlabel('Current (A)')
    ax3.legend(lines2, labels2, loc='best')
    
    plt.show()
        
            