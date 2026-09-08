    # -*- coding: utf-8 -*-

# functionality modules
import numpy as np
from statistics import stdev
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# my custom modules
import data_processing as processing
import signal_utils as utils
import curvefitting as curve

'''
NOTE - THIS CODE IS OUT OF DATE AND HASNT BEEN MODIFIED SINCE ITS USE IN JUNE
CAN BE MODIFIED IF REQUIRED - CORR MATRIX PART WORKS
PERHAPS IMPLEMENT SMARTER WAY OF LOADING MATERIAL DATA
'''

# user inputs

ftdelay = 5000.0
alpha_1 = 0.8
alpha_2 = 0.9
threshold = 0.4
all_plots = False
lower_bound = 25.0
refres_force = 50.0
show_fft_plots = False

# sample summary materials

sample_name = ['S6', 'S6_flipped', 'S14', 'S14_flipped', 'S25', 'S25_flipped',
               'S27', 'S27_flipped', 'S30', 'S30_flipped', 'S31', 'S31_flipped',
               'S35', 'S35_flipped', 'S36', 'S36_flipped']
graphene = ['GR94-04', 'GR94-04', 'GR94-04', 'GR94-04', 'GR113', 'GR113',
            'GR113B', 'GR113B', 'GR116', 'GR116', 'GR109', 'GR109',
            'GR116', 'GR116', 'GR116', 'GR116']
substrate = ['Panasonic', 'Panasonic', 'PET', 'PET', 'PET', 'PET', 
             'PET', 'PET', 'C81', 'C81', 'C81', 'C81',
             'PET', 'PET', 'C20', 'C20']
coating = ['Kapton', 'Kapton', 'C81', 'C81', 'CAD', 'CAD',
           'CAD', 'CAD', 'C81', 'C81', 'C81', 'C81',
           'CAD', 'CAD', 'C20', 'C20']

# file paths - NEED UPDATING BEFORE USING AGAIN

force_paths = [r"C:\Users\Cookie\Documents\harry\s6 v2\s6 v2 force data acquisition 1.csv",
               r"C:\Users\Cookie\Documents\harry\s6_flipped v2\s6_flipped v2 force data acquisition 1.csv",
               r"C:\Users\Cookie\Documents\harry\s14\s14 force data acquisition 1.csv",
               r"C:\Users\Cookie\Documents\harry\s14_flipped\s14_flipped force data acquisition 1.csv",
               r"C:\Users\Cookie\Documents\harry\s25\s25 force data acquisition 1.csv",
               r"C:\Users\Cookie\Documents\harry\s25_flipped\s25_flipped force data acquisition 1.csv",
               r"C:\Users\Cookie\Documents\harry\s27\force data acquisition 1.csv",
               r"C:\Users\Cookie\Documents\harry\s27_flipped\s27_flipped force data acquisition 1.csv",
               r"C:\Users\Cookie\Documents\harry\s30\s30 expt2 force data.csv",
               r"C:\Users\Cookie\Documents\harry\s30_flipped\s30_flipped force data acquisition 1.csv",
               r"C:\Users\Cookie\Documents\harry\s31\s31 force data acquisition 1.csv", 
               r"C:\Users\Cookie\Documents\harry\s31_flipped\s31_flipped force data acquisition 1.csv",
               r"C:\Users\Cookie\Documents\harry\s35\s35 force data acquisition 1.csv",
               r"C:\Users\Cookie\Documents\harry\s35_flipped\s35_flipped force data acquisition 1.csv",
               r"C:\Users\Cookie\Documents\harry\s36\s36 force data acquisition 1.csv",
               r"C:\Users\Cookie\Documents\harry\s36_flipped\s36_flipped force data acquisition 1.csv"
               ]

res_paths = [r"C:\Users\Cookie\Documents\harry\s6 v2\s6 v2 res data 2026-07-06.csv",
             r"C:\Users\Cookie\Documents\harry\s6_flipped v2\s6_flipped v2 res data 2026-07-06.csv",
             r"C:\Users\Cookie\Documents\harry\s14\s14 res data 2026-07-13.csv",
             r"C:\Users\Cookie\Documents\harry\s14_flipped\s14_flipped res data 2026-07-13.csv",
             r"C:\Users\Cookie\Documents\harry\s25\s25 res data 2026-07-13.csv",
             r"C:\Users\Cookie\Documents\harry\s25_flipped\s25_flipped res data 2026-07-13.csv",
             r"C:\Users\Cookie\Documents\harry\s27\res data acquisition 2026-07-02.csv",
             r"C:\Users\Cookie\Documents\harry\s27_flipped\s27_flipped res data 2026-07-03.csv",
             r"C:\Users\Cookie\Documents\harry\s30\s30 expt2 resis data 2026-07-01.csv",
             r"C:\Users\Cookie\Documents\harry\s30_flipped\s30_flipped res data 2026-07-03.csv",
             r"C:\Users\Cookie\Documents\harry\s31\s31 res data 2026-07-03.csv",
             r"C:\Users\Cookie\Documents\harry\s31_flipped\s31_flipped res data 2026-07-03.csv",
             r"C:\Users\Cookie\Documents\harry\s35\s35 res data 2026-07-13.csv",
             r"C:\Users\Cookie\Documents\harry\s35_flipped\s35_flipped res data 2026-07-13.csv",
             r"C:\Users\Cookie\Documents\harry\s36\s36 res data 2026-07-13.csv",
             r"C:\Users\Cookie\Documents\harry\s36_flipped\s36_flipped res data 2026-07-14.csv"
             ]

# dict
data = {"Sample Name": sample_name, "Graphene": graphene, "Substrate": substrate, "Coating": coating, 
        "Force File": force_paths, "Resistance File": res_paths}
#df
df = pd.DataFrame(data)

# calculate hyst ratios and sens

hysteresis_ratios = []
sensitivities_load_start = []
sensitivities_load_end = []
sensitivities_unload_start = []
sensitivities_unload_end = []

print("Starting batch processing for hysteresis ratios...\n")

# loop through each row of your dataframe using its indices
for index, row in df.iterrows():
    # Extract the file paths for the current sample channel
    sample = row['Sample Name']
    force_path = row['Force File']
    res_path = row['Resistance File']
    
    print(f"Processing {sample}...")
    
    try:
        # copy original code
        force, cftimes = processing.load_force_data(force_path, ftdelay)
        res, rtimes = processing.load_resistance_data(res_path)
        
        # relative resistance 
        peak_index = force.index(max(force))
        rising_time_force = cftimes[:peak_index + 1]
        rising_force = force[:peak_index + 1]
        
        target_time = np.interp(refres_force, rising_force, rising_time_force)
        res_reference = np.interp(target_time, rtimes, res)
        
        relres = [r / res_reference for r in res]

        # filter and differentiate
        filterforce = utils.simple_filter(force, alpha_1)
        ff_dot = utils.differentiator(cftimes, filterforce)
        rr_dot = utils.differentiator(rtimes, relres)

        # filter derivative
        filtered_ffdot = utils.simple_filter(ff_dot, alpha_2)
        filffdot_sigma = stdev(filtered_ffdot)

        # convert for event tracking 
        fil_ffdot_array = np.array(filtered_ffdot)
        cftimes_array = np.array(cftimes)

        all_peak_indices, all_peak_times = utils.detect_events(
            fil_ffdot_array, cftimes_array, threshold, filffdot_sigma
        )
            
        # find final values for hysteresis graph
        avg_force = processing.calculate_avg_force(all_peak_indices, filterforce)
        avg_relres = processing.calculate_avg_resistance(all_peak_times, rtimes, relres)
        hyst_ratio = processing.hysteresis_value(avg_force, avg_relres)
        
        # append hyst ratios
        hysteresis_ratios.append(hyst_ratio)
        
        # use lower force threshold to eliminate first regime
        f_arr_raw = np.array(avg_force)
        r_arr_raw = np.array(avg_relres)
        
        valid_mask = f_arr_raw >= lower_bound
        avg_force_2 = f_arr_raw[valid_mask]
        avg_relres_2 = r_arr_raw[valid_mask]
        
        # FIXED
        leaderboard_up, leaderboard_down, _ = curve.hyst_fit(avg_force_2, avg_relres_2)

        # each item inside the leaderboard is a tuple: (model_name, data_dict)
        best_model_up = leaderboard_up[0]    # Grabs ('Cubic', {...})
        best_model_down = leaderboard_down[0]  # Grabs ('Linear', {...})
        
        #loading sensitivities
        if best_model_up[1]['r2'] != -1:
            sens1, sens2 = curve.sens_values(best_model_up[0], best_model_up[1], avg_force_2)
        
        sensitivities_load_start.append(sens1)
        sensitivities_load_end.append(sens2)
        
        #unloading sensitivites
        if best_model_down[1]['r2'] != -1:
            sens3, sens4 = curve.sens_values(best_model_down[0], best_model_down[1], avg_force_2)
        
        sensitivities_unload_start.append(sens3)
        sensitivities_unload_end.append(sens4)
        
    except Exception as e:
        # If a specific file is corrupt or empty, log the error and record NaN 
        print(f"Error processing {sample}: {e}")
        hysteresis_ratios.append(np.nan)

# add to df
df['Hysteresis Ratio'] = hysteresis_ratios
df['Start sensitivity (loading)'] = sensitivities_load_start
df['End sensitivity (loading)'] = sensitivities_load_end
df['Start sensitivity (unloading)'] = sensitivities_unload_start
df['End sensitivity (unloading)'] = sensitivities_unload_end

def prod_final_df(name, dfx):
    dfx_gr = df[df['Graphene'] == name].copy()
    encoded_dfx_gr = pd.get_dummies(dfx_gr, columns = ['Substrate', 'Coating'])
    clean_encoded_dfx_gr = encoded_dfx_gr.drop(columns=['Sample Name', 'Graphene', 'Force File', 'Resistance File'])
    return clean_encoded_dfx_gr

df_9404_final = prod_final_df('GR94-04', df)
df_116_final = prod_final_df('GR116', df)

matrix_9404 = df_9404_final.corr()
matrix_116 = df_116_final.corr()

mask_9404 = np.zeros_like(matrix_9404, dtype=bool)
mask_9404[-4:, -4:] = True
mask_116 = np.zeros_like(matrix_116, dtype=bool)
mask_116[-6:,-6:] = True


plt.figure('GR94-04 correlation matrix', layout='constrained')
sns.heatmap(matrix_9404, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, mask=mask_9404)
plt.title("Correlation Heatmap - GR94-04")
plt.show()

plt.figure('GR116 correlation matrix', layout='constrained')
sns.heatmap(matrix_116, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, mask=mask_116)
plt.title("Correlation Heatmap - GR116")
plt.show()

# create table summarising results

df2 = df.drop(columns=['Force File', 'Resistance File'])
# build the matrix
def safe_to_4_sf(val):
    # Check specifically for numbers, excluding booleans (which are ints in Python)
    if isinstance(val, (int, float, np.number)) and not isinstance(val, bool):
        if not pd.isna(val):  # Ensure we don't break on missing data/NaNs
            return f"{val:.4g}"
    return str(val)  # Return non-numeric text exactly as-is


# apply the formatting across every cell safely
df_formatted = df2.map(safe_to_4_sf)

# initialise wide canvas window to fit 15 columns
fig, ax = plt.subplots(figsize=(24, 8))
ax.axis("off")

# generate the table (Omit 'rowLabels' to hide the index)
tbl = ax.table(
    cellText=df_formatted.values,
    colLabels=df_formatted.columns,
    loc="center",
    cellLoc="center",
)

# visual scaling & styling to prevent text compression
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)  # small text prevents character overlap
tbl.scale(1.0, 2.2)  # high vertical multiplier gives rows breathing room

# automatically adjust column widths based on cell text length
tbl.auto_set_column_width(col=list(range(len(df_formatted.columns))))

plt.tight_layout()

for col_idx in range(len(df_formatted.columns)):
    cell = tbl[0, col_idx]  # col header
    cell.set_facecolor("#0099CC")  
    cell.get_text().set_color("white")
    cell.get_text().set_weight("bold")
    
plt.show()