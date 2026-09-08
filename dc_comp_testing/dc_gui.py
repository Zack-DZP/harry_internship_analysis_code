# -*- coding: utf-8 -*-

# gui modules
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
# functionality modules
import numpy as np
import matplotlib.pyplot as plt
# my custom modules
import data_processing as processing
import signal_utils as utils
import plots
import curvefitting as curve


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
        self.root.title("DC graphene sensor data processing")
        self.root.geometry("850x700")

        # execution variables tracked
        self.force_path_var = tk.StringVar()
        self.res_path_var = tk.StringVar()
        self.fft_plots_var = tk.BooleanVar(value=False)

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
        tk.Label(file_frame, text="Resistance CSV Path:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
        tk.Entry(file_frame, textvariable=self.res_path_var, width=65).grid(row=1, column=1, padx=5, pady=5)
        tk.Button(file_frame, text="Browse...", command=self.browse_res).grid(row=1, column=2, padx=10, pady=5)
        
        # frame for parameters
        param_frame = tk.LabelFrame(self.root, text=" User Variables Configuration ")
        param_frame.pack(fill="x", padx=15, pady=5, ipady=5)

        # sample Name
        tk.Label(param_frame, text="Sample Name:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
        self.sample_name_entry = tk.Entry(param_frame, width=15)
        self.sample_name_entry.insert(0, "")
        self.sample_name_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # ftdelay
        tk.Label(param_frame, text="Force recording delay (ms):").grid(row=0, column=2, padx=15, pady=5, sticky="e")
        self.ftdelay_entry = tk.Entry(param_frame, width=15)
        self.ftdelay_entry.insert(0, "5000")
        self.ftdelay_entry.grid(row=0, column=3, padx=5, pady=5, sticky="w")

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
        self.threshold_entry.insert(0, "0.2")
        self.threshold_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        
        # lower force bound
        tk.Label(param_frame, text="Lower Force Bound (g):").grid(row=3, column=0, padx=10, pady=5, sticky="e")
        self.lower_bound_entry = tk.Entry(param_frame, width=15)
        self.lower_bound_entry.insert(0, "0.0")  # Sets 25g as your standard startup default choice
        self.lower_bound_entry.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        
        # force for reference resistance
        tk.Label(param_frame, text='Force for reference resistance (g):').grid(row=2, column=2, padx=10, pady=5, sticky='e')
        self.refres_entry = tk.Entry(param_frame, width=15)
        self.refres_entry.insert(0, "50.0")
        self.refres_entry.grid(row=2, column=3, padx=5, pady=5, sticky='w')
        
        # FFT
        self.fft_check = tk.Checkbutton(param_frame, text="Show FFT analysis:", variable=self.fft_plots_var)
        self.fft_check.grid(row=4, column=0, columnspan=2, padx=15, pady=5, sticky="")
        
        # changeover force bound
        tk.Label(param_frame, text="Changeover Force Bound (g):").grid(row=3, column=2, padx=10, pady=5, sticky="e")
        self.upper_bound_entry = tk.Entry(param_frame, width=15)
        self.upper_bound_entry.insert(0, "25.0")  # Sets 25g as your standard startup default choice
        self.upper_bound_entry.grid(row=3, column=3, padx=5, pady=5, sticky="w")
        
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
        filename = filedialog.askopenfilename(title="Select Resistance Data File", filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if filename:
            self.res_path_var.set(filename)

    # logic body (repeats main)
    
    def execute_processing_pipeline(self):
        # clear previous log session details
        self.console_display.delete("1.0", tk.END)
        plt.close('all')

        # extract paths and handle missing conditions up front
        force_path = self.force_path_var.get()
        res_path = self.res_path_var.get()
        
       
        
        if not force_path or not res_path:
            messagebox.showerror("Error", "Both Force and Resistance data paths are required to process variables.")
            return

        # get user input values
        try:
            sample_name = str(self.sample_name_entry.get())
            ftdelay = float(self.ftdelay_entry.get())
            alpha_1 = float(self.alpha1_entry.get())
            alpha_2 = float(self.alpha2_entry.get())
            threshold = float(self.threshold_entry.get())
            lower_bound = float(self.lower_bound_entry.get())
            upper_bound = float(self.upper_bound_entry.get())
            refres_force = float(self.refres_entry.get())
            show_fft_plots = self.fft_plots_var.get()

        except ValueError as err:
            messagebox.showerror("Validation Error", f"Failed parsing numerical config variables. Double check numerical variables entries.\nDetails: {err}")
            return

        # copy main
        try:
            print("--- Executing  ---")
            print(f"\nSample: {sample_name}")
            
            # initial processing
            force, cftimes = processing.load_force_data(force_path, ftdelay)
            res, rtimes = processing.load_resistance_data(res_path)
            
            # relres
            # slice to only include first half
            peak_index = force.index(max(force))
            rising_time_force = cftimes[:peak_index + 1]
            rising_force = force[:peak_index + 1]
            # target time
            target_time = np.interp(refres_force, rising_force, rising_time_force)
            res_reference = np.interp(target_time, rtimes, res)
            # find relres with reference force
            relres = []
            for i in range(len(res)):
                relres.append(res[i]/res_reference)
            print(f"\nRelative resistance calculated relative to resistance at {refres_force:.1f} g")

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
            avg_relres = processing.calculate_avg_resistance(all_peak_times, rtimes, relres)
            hyst_ratio = processing.hysteresis_value(avg_force, avg_relres)

            # print out results 
            print(f"Hysteresis ratio: {hyst_ratio:.4f}") # hyst ratio
            
            # convert to np array for fancy slicing      
            f_arr_raw = np.array(avg_force)
            r_arr_raw = np.array(avg_relres)
            
            valid_mask1 = (f_arr_raw >= lower_bound) & (f_arr_raw <= upper_bound) 
            avg_force_reg1 = f_arr_raw[valid_mask1]
            avg_relres_reg1 = r_arr_raw[valid_mask1]
            
            # regime 2
            valid_mask2 = f_arr_raw >= upper_bound
            avg_force_reg2 = f_arr_raw[valid_mask2]
            avg_relres_reg2 = r_arr_raw[valid_mask2]
            
            fit_subjects = [(avg_force_reg1, avg_relres_reg1, 'Resistance (lower regime)'),
                            (avg_force_reg2, avg_relres_reg2, 'Resistance (upper regime)')]
            
            # loop through 2x regimes for R and X (4 total)
            
            leaderboards_up = []
            leaderboards_down = []
            regime_equations = {}
            
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
  
    
            print(f'Resistance at {refres_force:.2f} g: {res_reference:.5g}')
            print("\n--- Execution finished ---")
            
            
            # plots

            plots.plot_time_series(sample_name, cftimes, force, rtimes, relres)
            
            plots.plot_hysteresis_loop(sample_name, 'resistance', avg_force, 
                                       avg_relres, leaderboards_up, leaderboards_down, lower_bound, 
                                       upper_bound, hyst_ratio)

            if show_fft_plots is True:
                plots.fft_plots(cftimes, rtimes, filterforce, force, relres)
            else:
                pass

        # should cover errors with preamble etc
        except Exception as pipeline_error:
            print(f"\n[FATAL SCRIPT CRASH]: {pipeline_error}")
            messagebox.showerror("Pipeline Process Failure", f"An anomaly broke execution:\n{pipeline_error}")

# NEED TO CREATE SOME KIND OF FUNCTION THAT ALLOWS EXPORTING FOR COMPARISON / CORR MATRIX

# initialise framework
if __name__ == "__main__":
    root = tk.Tk()
    app = ScriptGUI(root)
    root.mainloop()
