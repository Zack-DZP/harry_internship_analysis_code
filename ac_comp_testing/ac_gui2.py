# -*- coding: utf-8 -*-

# ui modules
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
# functionality modules
import numpy as np
import matplotlib.pyplot as plt
# my custom modules
import data_processing2 as processing
import signal_utils2 as utils
import plots2 as plots
import curvefitting2 as curve

# redirects print outputs

class PrintRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.insert("end", string)
        self.text_widget.see("end")  # scrolls to bottom automatically

    def flush(self):
        pass # needed to keep python happy

# gui window main class 

class ScriptGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AC graphene sensor data processing")
        self.root.geometry("850x700")

        # execution variables tracked
        self.force_path_var = tk.StringVar()
        self.rx_path_var = tk.StringVar()

        self.setup_ui()

    def setup_ui(self):
        
        # frame for file select
        file_frame = tk.LabelFrame(self.root, text=" File Inputs ")
        file_frame.pack(fill="x", padx=15, pady=10, ipady=5)

        # force data file
        tk.Label(file_frame, text="Force CSV Path:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
        tk.Entry(file_frame, textvariable=self.force_path_var, width=65).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(file_frame, text="Browse...", command=self.browse_force).grid(row=0, column=2, padx=10, pady=5)

        # res data file
        tk.Label(file_frame, text="RX CSV Path:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
        tk.Entry(file_frame, textvariable=self.rx_path_var, width=65).grid(row=1, column=1, padx=5, pady=5)
        tk.Button(file_frame, text="Browse...", command=self.browse_res).grid(row=1, column=2, padx=10, pady=5)
        
        # frame for parameters
        param_frame = tk.LabelFrame(self.root, text=" User Variables Configuration ")
        param_frame.pack(fill="x", padx=15, pady=5, ipady=5)

        # sample Name
        tk.Label(param_frame, text="Sample Name:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
        self.sample_name_entry = tk.Entry(param_frame, width=15)
        self.sample_name_entry.insert(0, "")
        self.sample_name_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # freq
        tk.Label(param_frame, text="Frequency (Hz):").grid(row=0, column=2, padx=15, pady=5, sticky="e")
        self.freq_entry = tk.Entry(param_frame, width=15)
        self.freq_entry.insert(0, "")
        self.freq_entry.grid(row=0, column=3, padx=5, pady=5, sticky="w")

        # alpha 1
        tk.Label(param_frame, text="Force filter alpha:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
        self.alpha1_entry = tk.Entry(param_frame, width=15)
        self.alpha1_entry.insert(0, "0.9")
        self.alpha1_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # alpha 2
        tk.Label(param_frame, text="Derivative filter alpha:").grid(row=1, column=2, padx=15, pady=5, sticky="e")
        self.alpha2_entry = tk.Entry(param_frame, width=15)
        self.alpha2_entry.insert(0, "0.9")
        self.alpha2_entry.grid(row=1, column=3, padx=5, pady=5, sticky="w")

        # threshold
        tk.Label(param_frame, text="Peak detection threshold:").grid(row=2, column=0, padx=10, pady=5, sticky="e")
        self.threshold_entry = tk.Entry(param_frame, width=15)
        self.threshold_entry.insert(0, "0.02")
        self.threshold_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        
        # force for reference r/x
        tk.Label(param_frame, text='Force for reference RX (g):').grid(row=2, column=2, padx=10, pady=5, sticky='e')
        self.refrx_entry = tk.Entry(param_frame, width=15)
        self.refrx_entry.insert(0, "0.0")
        self.refrx_entry.grid(row=2, column=3, padx=5, pady=5, sticky='w')
        
        # lower force bound
        tk.Label(param_frame, text="Lower Force Bound (g):").grid(row=3, column=0, padx=10, pady=5, sticky="e")
        self.lower_bound_entry = tk.Entry(param_frame, width=15)
        self.lower_bound_entry.insert(0, "0.0") # default choice
        self.lower_bound_entry.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        
        # regime changeover force bound
        tk.Label(param_frame, text="Changeover Force Bound (g):").grid(row=3, column=2, padx=10, pady=5, sticky="e")
        self.changeover_bound_entry = tk.Entry(param_frame, width=15)
        self.changeover_bound_entry.insert(0, "25.0") # default choice
        self.changeover_bound_entry.grid(row=3, column=3, padx=5, pady=5, sticky="w")
        
        # run button
        self.run_button = tk.Button(
            self.root, 
            text="Run", 
            font=("Arial", 12, "bold"), 
            bg="#0099CC", 
            fg="white", 
            command=self.execute_processing_pipeline
        )
        self.run_button.pack(pady=15)

        # text console window
        console_frame = tk.LabelFrame(self.root, text=" Output Log Console ")
        console_frame.pack(fill="both", expand=True, padx=15, pady=10)

        self.console_display = ScrolledText(console_frame, wrap="word", bg="white", fg="black", font=("Arial", 10))
        self.console_display.pack(fill="both", expand=True, padx=5, pady=5)

        # redirect print outputs using above class
        sys.stdout = PrintRedirector(self.console_display)

    # browse functions
    def browse_force(self):
        filename = filedialog.askopenfilename(title="Select Force Data File", filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if filename:
            self.force_path_var.set(filename)

    def browse_res(self):
        filename = filedialog.askopenfilename(title="Select RX Data File", filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if filename:
            self.rx_path_var.set(filename)

    # logic body (repeats main)
    
    def execute_processing_pipeline(self):
        plt.close('all')
        
        # clear previous log session details
        self.console_display.delete("1.0", tk.END)

        # extract paths and handle missing conditions up front
        force_path = self.force_path_var.get()
        rx_path = self.rx_path_var.get()       
       
        if not force_path or not rx_path:
            messagebox.showerror("Error", 
                                 "Both Force and RX data paths are required to process variables.")
            return 
        # get user input values
        try:
            sample_name = str(self.sample_name_entry.get())
            freq = str(self.freq_entry.get())
            alpha_1 = float(self.alpha1_entry.get())
            alpha_2 = float(self.alpha2_entry.get())
            threshold = float(self.threshold_entry.get())
            refrx_force = float(self.refrx_entry.get())
            lower_bound = float(self.lower_bound_entry.get())
            changeover_bound = float(self.changeover_bound_entry.get())

        except ValueError as err:
            messagebox.showerror("Validation Error", 
                                 f"Failed parsing numerical config variables. Double check numerical variables entries.\nDetails: {err}")
            return

        # copy main
        try:
            print("--- Executing  ---")
            print(f"\nSample: {sample_name} at {freq}Hz")
            
            # initial processing
            force, cftimes, res, react, rxtimes = processing.load_ac_data(force_path, rx_path)
            
            if res[0] < 10.0 or res[0] > 1000000:
                messagebox.showerror("Error", 
                                     "RX data was recorded incorrectly. Check the probes were connected properly.")
            # find mean reactance - used mostly for 100 Hz
            mean_react = np.mean(react)
            print(f"The mean reactance is {mean_react:.4g}")
            
            # relative resistance - CHANGED 13.7 - original method 
            # slice to only include first half
            peak_index = force.index(max(force))
            rising_time_force = cftimes[:(peak_index + 1)]
            rising_force = force[:(peak_index + 1)]
            # target time
            target_time = np.interp(refrx_force, rising_force, rising_time_force)
            r_reference = np.interp(target_time, rxtimes, res)
            x_reference = np.interp(target_time, rxtimes, react)
            
            print(f'Reference resistance at {refrx_force:.2f}: {r_reference:.5g}')
                
            # find relres with new reference
            relres = []
            for value in res:
                relres.append(value/r_reference)
                
            relx = []
            for value in react:
                relx.append(value/x_reference)
                    
            print(f"\nRelative resistance calculated relative to resistance at {refrx_force:.1f} g")
            
            # filter and differentiate
            filterforce = utils.simple_filter(force, alpha_1)
            ff_dot = utils.differentiator(cftimes, filterforce)

            # filter derivative
            filtered_ffdot = utils.simple_filter(ff_dot, alpha_2)
            filffdot_sigma = np.std(filtered_ffdot)

            # convert for event tracking 
            fil_ffdot_array = np.array(filtered_ffdot)
            cftimes_array = np.array(cftimes)

            all_peak_indices, all_peak_times = utils.detect_events(
                fil_ffdot_array, cftimes_array, threshold, filffdot_sigma
            )
                
            # find final values for hysteresis graph
            avg_force = processing.calculate_avg_force(all_peak_indices, filterforce)
            avg_relres = processing.calculate_avg_resistance(all_peak_times, rxtimes, relres)
            avg_relx = processing.calculate_avg_resistance(all_peak_times, rxtimes, relx)
            hyst_ratio_r = processing.hysteresis_value(avg_force, avg_relres)
            hyst_ratio_x = processing.hysteresis_value(avg_force, avg_relx)

            # print out results 
            print(f"Resistance hysteresis ratio: {hyst_ratio_r:.4f}") # hyst ratio r
            print(f"Reactance hysteresis ratio: {hyst_ratio_x:.4f}") # hyst ratio x

            # using lower force threshold to eliminate first regime        
            # NEW - use changeover bound to split the two regimes
            f_arr_raw = np.array(avg_force)
            r_arr_raw = np.array(avg_relres)
            x_arr_raw = np.array(avg_relx)
            
            # regime 1
            valid_mask1 = (f_arr_raw >= lower_bound) & (f_arr_raw <= changeover_bound) 
            avg_force_reg1 = f_arr_raw[valid_mask1]
            avg_relres_reg1 = r_arr_raw[valid_mask1]
            avg_relx_reg1 = x_arr_raw[valid_mask1]
            
            # regime 2
            valid_mask2 = f_arr_raw >= changeover_bound
            avg_force_reg2 = f_arr_raw[valid_mask2]
            avg_relres_reg2 = r_arr_raw[valid_mask2]
            avg_relx_reg2 = x_arr_raw[valid_mask2]
            
            # filter for time series graph
            fil_relr = utils.simple_filter(relres, 0.9)
            fil_relx = utils.simple_filter(relx, 0.9)

            fit_subjects = [(avg_force_reg1, avg_relres_reg1, 'Resistance (lower regime)'),
                            (avg_force_reg2, avg_relres_reg2, 'Resistance (upper regime)'),
                            (avg_force_reg1, avg_relx_reg1, 'Reactance (lower regime)'),
                            (avg_force_reg2, avg_relx_reg2, 'Reactance (upper regime)')]
            
            # loop through 2x regimes for R and X (4 total)
            
            for sub_force, rx, name in fit_subjects:
                leaderboard_up, leaderboard_down, _ = curve.hyst_fit(sub_force, rx)
                
                best_model_up = leaderboard_up[0]
                best_model_down = leaderboard_down[0]
                
                print(f"\n{name} loading curve fit:")
                for rank, (model, data) in enumerate(leaderboard_up, start=1):
                    if data['r2'] != -1:
                        print(f" {rank}. {model} (R^2 = {data['r2']:.4g}) -> {data['formula']}")
                    else:
                        print(f" {rank}. {model} -> Fit failed.")
                print(f" Best Loading Fit: {best_model_up[0]} (R^2 = {best_model_up[1]['r2']:.4g})")
                        
                if best_model_up[1]['r2'] != -1:
                    sens_log_up = curve.calculate_sensitivity(best_model_up[0], best_model_up[1], sub_force)
                    print(sens_log_up)
                            
                print(f"\n{name} unloading curve fit:")
                for rank, (model, data) in enumerate(leaderboard_down, start=1):
                    if data['r2'] != -1:
                        print(f" {rank}. {model} (R^2 = {data['r2']:.4g}) -> {data['formula']}")
                    else:
                        print(f" {rank}. {model} -> Fit failed.")
                print(f" Best Unloading Fit: {best_model_down[0]} (R^2 = {best_model_down[1]['r2']:.4g})")
                                        
                if best_model_down[1]['r2'] != -1:
                    sens_log_down = curve.calculate_sensitivity(best_model_down[0], best_model_down[1], sub_force)
                    print(sens_log_down)
  
            print("\n--- Execution finished ---")
            
            
            
            leaderboards_up = []
            leaderboards_down = []
            regime_equations = {}

            for sub_force, rx, name in fit_subjects:
                leaderboard_up, leaderboard_down, _ = curve.hyst_fit(sub_force, rx)
                leaderboards_up.append((name, leaderboard_up))
                leaderboards_down.append((name, leaderboard_down))
    
                # Extract top model formulas
                _, best_up_data = leaderboard_up[0]
                _, best_down_data = leaderboard_down[0]
                
                formula_up = best_up_data['formula'] if best_up_data['r2'] != -1 else "Fit failed"
                formula_down = best_down_data['formula'] if best_down_data['r2'] != -1 else "Fit failed"
                
                # Store formatted equation strings
                regime_equations[f"\n{name} (Loading)"] = f"{formula_up}"
                regime_equations[f"{name} (Unloading)"] = f"{formula_down}"
            
            # plots

            plots.plot_time_series(sample_name, freq, cftimes, force, rxtimes, fil_relr, fil_relx)
            
            plots.plot_hysteresis_loop(sample_name, freq, 'resistance', avg_force, 
                                       avg_relres, leaderboards_up, leaderboards_down, lower_bound, 
                                       changeover_bound, hyst_ratio_r)
            plots.plot_hysteresis_loop(sample_name, freq, 'reactance', avg_force, 
                                       avg_relx, leaderboards_up, leaderboards_down, lower_bound, 
                                       changeover_bound, hyst_ratio_x)
            
            # create data summary dictionary    
            data_summary = {'Sample name': sample_name,
                            'Frequency': f"{freq}Hz",
                            '\nMean reactance': mean_react,
                            '\nResistance hysteresis ratio': hyst_ratio_r,
                            'Reactance hysteresis ratio': hyst_ratio_x,
                            }
            data_summary.update(regime_equations)

            # format and create o/p text window
            output_text = ""
            # format to 4dp
            for name, value in data_summary.items():
                if isinstance(value, (int, float)):
                    formatted_val = f"{value:.4g}"
                else:
                    formatted_val = str(value)
                output_text += f"{name}: {formatted_val}\n"
        
            # popup text window
            pop_window = tk.Toplevel(self.root) 
            pop_window.title("Processing Summary")
            pop_window.geometry("800x500")
            pop_window.configure(bg="#F5F5F7")  # nice grey background

            # make it pretty
            text_area = tk.Text(
                pop_window, 
                padx=15, 
                pady=15, 
                font=("Arial", 10),      
                bg="white",              
                fg="black",              
                relief="flat",              # remove retro borders
                highlightthickness=3,       # borderline
                highlightbackground="#0099CC" # dzp blue
                )
            text_area.pack(expand=True, fill="both", padx=15, pady=15)
            
            text_area.tag_configure("blue_title", foreground="#0099CC", font=("Arial", 14, "bold"))
            
            text_area.insert(tk.END, "Summary of analysis \n\n", "blue_title")
            text_area.insert(tk.END, output_text)
            
            # read only
            text_area.configure(state="disabled")
            
        # should cover errors with preamble etc
        except Exception as pipeline_error:
            print(f"\n[FATAL SCRIPT CRASH]: {pipeline_error}")
            messagebox.showerror("Pipeline Process Failure", f"An anomaly broke execution:\n{pipeline_error}")

# initialise framework
if __name__ == "__main__":
    root = tk.Tk()
    app = ScriptGUI(root)
    root.mainloop()