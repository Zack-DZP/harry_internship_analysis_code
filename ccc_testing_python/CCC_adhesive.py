# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import ccc_utils

# DECIDE WHETHER AUTO PEAK FINDING IS VALID
peak_times_manual_SA2 = ["10:04:30", "10:13:04", "10:25:53", "10:48:44",
                         "11:02:36", "11:14:46", "11:34:16", "11:47:48",
                         "11:59:20", "12:13:19", "12:30:03", "12:43:27",
                         "12:57:35"]
peak_times_manual = peak_times_manual_SA2 # choose set of manual peak times

# expt info
sample = "SA4"


if sample == "SA2":    
    temp_path = r"C:\Users\Cookie\Documents\harry\CCC_DATA\SA2_Temp.csv"
    volt_path = r"C:\Users\Cookie\Documents\harry\CCC_DATA\SA2_Volt.csv"
    PEAK_FINDER_VALID = False
elif sample == "SA4":
    temp_path = r"C:\Users\Cookie\Documents\harry\CCC_DATA\SA4_Temp.csv"
    volt_path = r"C:\Users\Cookie\Documents\harry\CCC_DATA\SA4_Volt.csv"
    PEAK_FINDER_VALID = True
else:
    pass

# data in case data starts at nonzero current
start_time = 0  # in seconds
start_curr = 0
end_curr = 3.0
incr_size = 0.25 # all in amps

width = 10 # in mm
depth = 0.024 # in mm

alpha = 0.8 # filter parameter
threshold = 8 # standard deviation multiple for derivative peak finder
distance = 120 # distance used for peak finder to try eliminate multiple peaks

# extract temp data and exclude bad data
def ccc_adhesive_processing(temp_path=temp_path, volt_path=volt_path, 
                            sample=sample, PEAK_FINDER_VALID=PEAK_FINDER_VALID):
    
    temp_data = pd.read_csv(temp_path)
    temp_data = temp_data.dropna().reset_index(drop=True) # drop NaN values to avoid error
    
    time_parts = temp_data["Unnamed: 0"].str.split(":", expand=True)
    hours = pd.to_numeric(time_parts[0])
    minutes = pd.to_numeric(time_parts[1])
    seconds = pd.to_numeric(time_parts[2])
    
    total_ms = 1000 * (3600 * hours + 60 * minutes + seconds)
    temp_data['total_ms'] = total_ms - total_ms[0] # correct times in case times didnt start at 0
    
    temp_data = temp_data[temp_data['total_ms'] > start_time*1000 ] # mask in case of bad data
    
    temp_bond = temp_data["Bond temp Ave. (C)"].to_numpy()
    temp_amb = temp_data["Ambient temp Ave. (C)"].to_numpy()
    temp_ni = temp_data["Nickel temp Ave. (C)"].to_numpy()
    temp_electrode = temp_data["Electrode temp Ave. (C)"].to_numpy()
    temp_times = temp_data['total_ms'].to_numpy()
    
    # same for volt data
    
    volt_data = pd.read_csv(volt_path, skiprows=16)
    volt_data["Time (s)"] = pd.to_datetime(volt_data["Time (s)"])
    elapsed_series = (volt_data['Time (s)'] - volt_data['Time (s)'].iloc[0]).dt.total_seconds()
    volt_data['Time (ms)'] = 1000 * elapsed_series
    
    volt_data = volt_data[volt_data["Time (ms)"] > 1000 * start_time] # same mask
    
    volt = volt_data["Channel 1 (VDC)"].to_numpy()
    volt_times = volt_data['Time (ms)'].to_numpy()
    
    # main body
    num_incr = int((end_curr - start_curr) / incr_size)
    
    temp_bond = ccc_utils.simple_filter(temp_bond, alpha)
    temp_amb = ccc_utils.simple_filter(temp_amb, alpha)
    temp_ni = ccc_utils.simple_filter(temp_ni, alpha)
    temp_electrode = ccc_utils.simple_filter(temp_electrode, alpha)
    
    volt = ccc_utils.simple_filter(volt, alpha)
    
    # reinterpolate temp to volt timescale
    times = volt_times # rename - now a common scale between vars
    
    if PEAK_FINDER_VALID:
        # automatic peak detection for volt
        volt_dot = ccc_utils.differentiator(times, volt) # diff for impulse train
        volt_dot = ccc_utils.simple_filter(volt_dot, alpha) # smooth it
        vd_sigma = np.std(volt_dot) # st dev
        # use peak detection IF NECESSARY
        peak_indices, _ = ccc_utils.detect_events(volt_dot, times, threshold, vd_sigma, distance)
        peak_indices = np.insert(peak_indices, 0, 0.0)
        peak_indices = np.append(peak_indices, len(times)-1)
    else: # in the case of noisy data
        # insert peak times manually, determined using time series graph in temperature plot
        timedeltas = pd.to_timedelta(peak_times_manual)
        elapsed_series = (timedeltas - timedeltas[0]) // pd.Timedelta(milliseconds=1)
        elapsed_ms = elapsed_series.to_numpy()
        # find indices coreresponding to peak times
        peak_indices = [np.searchsorted(times, t) for t in elapsed_ms]
        peak_indices.append(len(times)-1) # add on final time to ensure they line up
        peak_indices = np.array(peak_indices) # convert to arr

    # generate current values
    current = np.array(0)
    if (len(peak_indices)-1) != num_incr+1:
        print("Error - number of increments doesn't line up with peak times")
    else:
        for i in range(len(peak_indices)-1):
            curr_value = i * incr_size + start_curr
            current = np.append(current, np.full(peak_indices[i+1]-peak_indices[i], curr_value))
            
    # calculate resistance
    res = np.divide(volt, current, where=(current != 0), out=None)
    res = ccc_utils.simple_filter(res, alpha)

    # discretise data for T/I plots
    # find eqm temps - last 20% averaged
    temp_bond2 = np.interp(times, temp_times, temp_bond) 
    temp_amb2 = np.interp(times, temp_times, temp_amb) 
    temp_ni2 = np.interp(times, temp_times, temp_ni) 
    temp_electrode2 = np.interp(times, temp_times, temp_electrode) 
    
    discr_temp_bond = ccc_utils.discretise(temp_bond2, peak_indices, ub=0.95)
    discr_temp_amb = ccc_utils.discretise(temp_amb2, peak_indices, ub=0.95)
    discr_temp_ni = ccc_utils.discretise(temp_ni2, peak_indices, ub=0.95)
    discr_temp_electrode = ccc_utils.discretise(temp_electrode2, peak_indices, ub=0.95)
    
    # currentemp_ni, di
    discr_curr = np.linspace(0, (num_incr)*incr_size, num_incr+1)
    norm_curr = discr_curr / (width*depth)

    return times, discr_temp_bond, discr_temp_amb, discr_temp_ni, discr_temp_electrode, norm_curr, current, volt, res

times, discr_temp_bond, discr_temp_amb, discr_temp_ni, discr_temp_electrode, norm_curr, current, volt, res = ccc_adhesive_processing()



# plots

if __name__ == '__main__':
    

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
    line4, = ax3.plot(norm_curr, discr_temp_bond, label='Bond temperature')
    line5, = ax3.plot(norm_curr, discr_temp_amb, label='Ambient temperature')
    line6, = ax3.plot(norm_curr, discr_temp_ni, label='Nickel temperature')
    line7, = ax3.plot(norm_curr, discr_temp_electrode, label='Printed silver temperature')
    ax3.set_title(f"{sample} - Temperature vs current density")
    ax3.set_ylabel('Temperature (deg C)')
    ax3.set_xlabel('Current density (A/mm^2)')
    lines2 = [line4, line5, line6, line7]
    labels2 = [l.get_label() for l in lines2]
    ax4 = ax3.secondary_xaxis('top', functions=(lambda x: x*width*depth, lambda x: x*width*depth))
    ax4.set_xlabel('Current (A)')
    ax3.legend(lines2, labels2, loc='best')
    
    plt.show()





 