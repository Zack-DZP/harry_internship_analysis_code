# -*- coding: utf-8 -*-

import pandas as pd

def load_temp_data(temp_path):
    temp_data = pd.read_csv(temp_path)
    temp_data = temp_data.dropna().reset_index(drop=True) # drop NaN values to avoid error
    temp_tot = temp_data["Channel 1 Ave. (C)"].to_numpy()
    temp_amb = temp_data["Channel 2 Ave. (C)"].to_numpy()
    temp_diff = temp_tot - temp_amb
    time_parts = temp_data["Unnamed: 0"].str.split(":", expand=True)
    hours = pd.to_numeric(time_parts[0])
    minutes = pd.to_numeric(time_parts[1])
    seconds = pd.to_numeric(time_parts[2])
    total_ms = 1000 * (3600 * hours + 60 * minutes + seconds)
    temp_data['total_ms'] = total_ms - total_ms[0] # correct times in case times didnt start at 0
    temp_times = temp_data['total_ms'].to_numpy()
    
    return temp_diff, temp_times, temp_tot

def load_volt_data(volt_path):
    header_row = 0
    with open(volt_path, 'r') as f:
        for idx, line in enumerate(f):
            if "Channel 1 (VDC)" in line:
                header_row = idx
                break
    volt_data = pd.read_csv(volt_path, skiprows=header_row)
    volt = volt_data["Channel 1 (VDC)"].to_numpy()
    volt_data["Time (s)"] = pd.to_datetime(volt_data["Time (s)"])
    elapsed_series = (volt_data['Time (s)'] - volt_data['Time (s)'].iloc[0]).dt.total_seconds()
    volt_time = 1000 * elapsed_series.to_numpy()
    
    return volt, volt_time

