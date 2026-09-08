# -*- coding: utf-8 -*-

import traceback
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import numpy as np
import matplotlib.pyplot as plt

import w_processing as processing
import w_plots as plots
import w_curve as curve

class PrintRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.insert("end", string)
        self.text_widget.see("end")  # scrolls to bottom automatically

    def flush(self):
        pass # needed to keep python happy

class ScriptGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Wire stretching data analysis")
        self.root.geometry("615x750")

        self.original_stdout = sys.stdout
        # execution variables tracked
        self.rx_path_var = tk.StringVar()
        self.calc_mode_var = tk.StringVar(value="Resistance")

        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        # frame for file select
        file_frame = tk.LabelFrame(self.root, text=" File Input ")
        file_frame.pack(fill="x", padx=15, pady=10, ipady=5)

        # res data file
        tk.Label(file_frame, text="RX CSV Path:").grid(row=1, column=0, padx=10, pady=5,
                                                       sticky="e")
        tk.Entry(file_frame, textvariable=self.rx_path_var, width=45).grid(row=1, column=1,
                                                                           padx=5, pady=5)
        tk.Button(file_frame, text="Browse...", command=self.browse_res).grid(row=1, column=2,
                                                                              padx=10, pady=5)

        # frame for parameters
        param_frame = tk.LabelFrame(self.root, text=" User Variables Configuration ")
        param_frame.pack(fill="x", padx=15, pady=5, ipady=5)

        # pack outer cols with empty space to centre inputs
        param_frame.grid_columnconfigure(0, weight=1)
        param_frame.grid_columnconfigure(3, weight=1)

        # active length
        tk.Label(param_frame, text="Active length (mm):").grid(row=0, column=1, padx=10, pady=5,
                                                               sticky="e")
        self.active_length_entry = tk.Entry(param_frame, width=15)
        self.active_length_entry.insert(0, "100")
        self.active_length_entry.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        # variable to be analysed
        tk.Label(param_frame, text="Calculation Type:").grid(row=1, column=1, padx=10, pady=5,
                                                             sticky="e")
        self.radio_resistance = tk.Radiobutton(
            param_frame, text="Resistance", variable=self.calc_mode_var, value="Resistance")
        self.radio_resistance.grid(row=1, column=2, padx=5, pady=5, sticky="w")
        self.radio_reactance = tk.Radiobutton(
            param_frame, text="Reactance", variable=self.calc_mode_var, value="Reactance")
        self.radio_reactance.grid(row=2, column=2, padx=5, pady=5, sticky="w")
        self.radio_para_res = tk.Radiobutton(
            param_frame, text="Parallel resistance", variable=self.calc_mode_var, 
            value="Parallel resistance")
        self.radio_para_res.grid(row=3, column=2, padx=5, pady=5, sticky="w")
        self.radio_inductance = tk.Radiobutton(
            param_frame, text="Parallel inductance", variable=self.calc_mode_var, 
            value="Parallel inductance")
        self.radio_inductance.grid(row=4, column=2, padx=5, pady=5, sticky="w")
        self.radio_capacitance = tk.Radiobutton(
            param_frame, text="Parallel capacitance", variable=self.calc_mode_var, 
            value="Parallel capacitance")
        self.radio_capacitance.grid(row=5, column=2, padx=5, pady=5, sticky="w")
        
        # run button
        self.run_button = tk.Button(
            self.root,
            text="Run analysis",
            font=("Arial", 12, "bold"),
            bg="#0099CC",
            fg="white",
            command=self.execute_processing_pipeline,
            width=15
        )
        self.run_button.pack(pady=15)

        # text console window
        console_frame = tk.LabelFrame(self.root, text=" Output Log Console ")
        console_frame.pack(fill="both", expand=True, padx=15, pady=10)
        self.console_display = ScrolledText(console_frame, wrap="word",
                                            bg="white", fg="black", font=("Arial", 10))
        self.console_display.pack(fill="both", expand=True, padx=5, pady=5)
        # redirect print outputs using above class
        sys.stdout = PrintRedirector(self.console_display)
        
    def browse_res(self):
        filename = filedialog.askopenfilename(title="Select RX Data File",
                                              filetypes=[("CSV Files", "*.csv"),
                                                         ("All Files", "*.*")])
        if filename:
            self.rx_path_var.set(filename)

    def execute_processing_pipeline(self):
        plt.close('all')
        # clear previous log session details
        self.console_display.delete("1.0", tk.END)

        # get user input values
        ACTIVE_LENGTH = float(self.active_length_entry.get())
        CALC_MODE = str(self.calc_mode_var.get())
        RX_PATH = self.rx_path_var.get()
        mode_map = {
            'Resistance':            (r"R(OHM)", 'Ohms'),
            'Reactance':             (r"X(OHM)", 'Ohms'),
            'Parallel resistance':   (r"Rp(OHM)", 'Ohms'),
            'Parallel inductance':   (r"Lp(H)", 'H'),
            'Parallel capacitance':  (r"Cp(F)", 'F')
            }
        
        try:
            # look up mode
            header, unit = mode_map.get(CALC_MODE)
            # get data
            data = processing.load_rx_data(RX_PATH, header)
            
            plot_info = processing.load_names(RX_PATH) # sample name, freq, extension
            strain = float(plot_info[2]) / ACTIVE_LENGTH
            plot_info[3] = strain
            plot_info[4] = CALC_MODE # creating list w test info for plots
            plot_info[5] = unit

            # peak detection
            data_event_indices = processing.find_peaks_troughs_detrended(data)
            data0 = np.mean(data[:(data_event_indices[0] - 20)]) # avg before first peak

            # fom1 - ptp avg values
            ptp_data = processing.find_ptp_res(data_event_indices, data)
            fom1a = np.mean(ptp_data)
            fom1r = fom1a / data0
            cycle_nums = 0.5 * np.arange(len(data_event_indices)-1)
            # fom2
            midline_vals = processing.find_midline(data_event_indices, data)
            try:
                midline_exp_p, midline_exp_func, midline_exp_r2 = curve.fit_exp(cycle_nums,
                                                                                midline_vals)
                ml_exp_x, ml_exp_y = curve.gen_trendline(cycle_nums, midline_exp_func)

                midline_ln_p, midline_ln_func, midline_ln_r2 = curve.fit_ln(cycle_nums,
                                                                            midline_vals)
                ml_ln_x, ml_ln_y = curve.gen_trendline(cycle_nums, midline_ln_func)
            except Exception as e:
                print('Midline:', e)
            # fom3
            try:
                amp_exp_p, amp_exp_func, amp_exp_r2 = curve.fit_exp(cycle_nums, ptp_data)
                amp_exp_x, amp_exp_y = curve.gen_trendline(cycle_nums, amp_exp_func)

                amp_ln_p, amp_ln_func, amp_ln_r2 = curve.fit_ln(cycle_nums, ptp_data)
                amp_ln_x, amp_ln_y = curve.gen_trendline(cycle_nums, amp_ln_func)
            except Exception as e:
                print('\nAmplitude:', e)

            # FOM o/ps
            print(f'\nMean initial value ({unit}): ', processing.to_4sf(data0))
            print('\nFOM1A: ', processing.to_4sf(fom1a))
            print('FOM1R: ', processing.to_4sf(fom1r))
            print('\nFOM2 log coefficients (a,b,c):',
                  *(processing.to_4sf(item) for item in midline_ln_p),
                  f"\n(R2 = {midline_ln_r2:.4g})")
            print('FOM2 exp coefficients (a,b,c):',
                  *(processing.to_4sf(item) for item in midline_exp_p),
                  f"\n(R2 = {midline_exp_r2:.4g})")
            print('\nFOM3 log coefficients (a,b,c):',
                  *(processing.to_4sf(item) for item in amp_ln_p),
                  f"\n(R2 = {amp_ln_r2:.4g})")
            print('FOM3 exp coefficients (a,b,c):',
                  *(processing.to_4sf(item) for item in amp_exp_p),
                  f"\n(R2 = {amp_exp_r2:.4g})")

            # plots
            plt.style.use("seaborn-v0_8-bright")
            plots.plot_res_series(data, data_event_indices, plot_info)
            plots.plot_midline_series(cycle_nums, midline_vals, data, data_event_indices,
                                      ml_exp_x, ml_exp_y, ml_ln_x, ml_ln_y, plot_info)
            plots.plot_amplitude(cycle_nums, ptp_data, amp_exp_x, amp_exp_y,
                                 amp_ln_x, amp_ln_y, plot_info)
            plt.show()

        except Exception as pipeline_error:
            # produce traceback and error text
            full_traceback = traceback.format_exc()
            error_message = str(pipeline_error)
            # print error string to log
            print(f"\n[FATAL SCRIPT CRASH]:\n{full_traceback}")
            # pass error string into lambda arg
            self.root.after(0, lambda err=error_message: messagebox.showerror(
                "Pipeline Process Failure",
                f"An anomaly broke execution:\n\n{err}\n\nCheck the Log Console for full details."
            ))

    def on_closing(self):
        # reset stdout
        sys.stdout = self.original_stdout
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ScriptGUI(root)
    root.mainloop()
