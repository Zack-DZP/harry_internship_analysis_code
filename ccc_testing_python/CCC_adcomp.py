# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from CCC_adhesive import ccc_adhesive_processing

samples    = ("SA2", 
              "SA4")#
temp_paths = (r"C:\Users\Cookie\Documents\harry\CCC_DATA\SA2_Temp.csv",
              r"C:\Users\Cookie\Documents\harry\CCC_DATA\SA4_Temp.csv")
volt_paths = (r"C:\Users\Cookie\Documents\harry\CCC_DATA\SA2_Volt.csv",
              r"C:\Users\Cookie\Documents\harry\CCC_DATA\SA4_Volt.csv")
validity = [False, True]

discr_temp_bond_l, discr_temp_amb_l, discr_temp_ni_l, discr_temp_electrode_l = [], [], [], []
norm_curr_l = []


for i in range(len(samples)):
    _, discr_temp_bond, discr_temp_amb, discr_temp_ni, discr_temp_electrode, norm_curr, _, _, _ = ccc_adhesive_processing(temp_path=temp_paths[i], 
                                                                                                                          volt_path=volt_paths[i],
                                                                                                                          PEAK_FINDER_VALID=validity[i]) 
    norm_curr_l.append(norm_curr)
    discr_temp_bond_l.append(discr_temp_bond)
    discr_temp_amb_l.append(discr_temp_amb)
    discr_temp_ni_l.append(discr_temp_ni)
    discr_temp_electrode_l.append(discr_temp_electrode)

# comparison graph

plt.style.use('seaborn-v0_8-ticks')
style = ['solid', 'dashed']

plt.figure('Adhesive CCC Comparison', layout='constrained')
for i in range(len(samples)):
    plt.plot(norm_curr_l[i], discr_temp_bond_l[i], label=f'{samples[i]} - Bond temperature', color='r', linestyle=style[i])
    plt.plot(norm_curr_l[i], discr_temp_amb_l[i], label=f'{samples[i]} - Ambient temperature', color='b', linestyle=style[i])
    plt.plot(norm_curr_l[i], discr_temp_ni_l[i], label=f'{samples[i]} - Nickel temperature', color='lime', linestyle=style[i])
    plt.plot(norm_curr_l[i], discr_temp_electrode_l[i], label=f'{samples[i]} - Printed silver temperature', color='m', linestyle=style[i])
    
plt.xlabel('Current density (A / mm^2)')
plt.ylabel('Temperature (deg C)')
plt.legend()
plt.title('Adhesive CCC comparison')

plt.show