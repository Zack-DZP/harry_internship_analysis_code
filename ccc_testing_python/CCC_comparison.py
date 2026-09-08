# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from CCC_testing import ccc_processing

samples    = ("S2", 
              "S5")
temp_paths = (r"C:\Users\Cookie\Documents\harry\CCC_TESTS\S2_Temp.csv",
              r"C:\Users\Cookie\Documents\harry\CCC_TESTS\S5_Temp.csv")
volt_paths = (r"C:\Users\Cookie\Documents\harry\CCC_TESTS\S2_Volt.csv",
              r"C:\Users\Cookie\Documents\harry\CCC_TESTS\S5_Volt.csv")

norm_curr_list = []
discr_temp_list = []
discr_temp_tot_list = []

for i in range(len(samples)):
    _, norm_curr, discr_temp, discr_temp_tot,_,_,_,_ = ccc_processing(temp_paths[i], volt_paths[i]) 
    norm_curr_list.append(norm_curr)
    discr_temp_list.append(discr_temp)
    discr_temp_tot_list.append(discr_temp_tot)

# comparison graph

plt.figure('CCC Comparison')
for i in range(len(samples)):
    plt.plot(norm_curr_list[i], discr_temp_list[i], label=f'{samples[i]} - Temperature above ambient')
    plt.plot(norm_curr_list[i], discr_temp_tot_list[i], label=f'{samples[i]} - Temperature')
plt.xlabel('Current density (A / mm^2)')
plt.ylabel('Temperature (deg C)')
plt.legend()
plt.title('CCC comparison')

plt.show