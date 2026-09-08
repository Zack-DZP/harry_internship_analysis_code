# -*- coding: utf-8 -*-

import traceback
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from temptest_utils import simple_filter, to_4sf

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
        self.root.title("Temperature testing data analysis")
        self.root.geometry("650x500")

        self.original_stdout = sys.stdout
        # execution variables tracked
        self.rx_path_var = tk.StringVar()
        self.temp_path_var = tk.StringVar()

        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        # frame for file select
        file_frame = tk.LabelFrame(self.root, text=" File Input ")
        file_frame.pack(fill="x", padx=15, pady=10, ipady=5)

        # res data file
        tk.Label(file_frame, text="RX CSV Path:").grid(row=0, column=0, padx=10, pady=5,
                                                       sticky="e")
        tk.Entry(file_frame, textvariable=self.rx_path_var, width=40).grid(row=0, column=1,
                                                                           padx=5, pady=5)
        tk.Button(file_frame, text="Browse...", command=self.browse_res).grid(row=0, column=2,
                                                                              padx=10, pady=5)

        # temp data file
        tk.Label(file_frame, text="Temperature CSV Path:").grid(row=1, column=0, padx=10, pady=5,
                                                       sticky="e")
        tk.Entry(file_frame, textvariable=self.temp_path_var, width=40).grid(row=1, column=1,
                                                                           padx=5, pady=5)
        tk.Button(file_frame, text="Browse...", command=self.browse_temp).grid(row=1, column=2,
                                                                              padx=10, pady=5)

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

    def browse_temp(self):
        filename = filedialog.askopenfilename(title="Select Temperature Data FIle",
                                              filetypes=[("CSV Files", "*.csv"),
                                                         ("All Files", "*.*")])
        if filename:
            self.temp_path_var.set(filename)

    def execute_processing_pipeline(self):
        plt.close('all')
        # clear previous log session details
        self.console_display.delete("1.0", tk.END)

        # get user input values
        TEMP_PATH = str(self.temp_path_var.get())
        RX_PATH = str(self.rx_path_var.get())

        try:
            # might be able to auto find freq and sample name
            filename = Path(RX_PATH).stem
            # Split by underscore
            parts = filename.split("_")
            # Sample is always first, Frequency is always last
            sample_name = parts[0]
            freq = parts[-1]

            # extract rx data
            header_row = 0
            with open(RX_PATH, 'r') as f:
                for idx, line in enumerate(f):
                    if r"R(OHM)" in line:
                        header_row = idx
                        break
            rx_data = pd.read_csv(RX_PATH, skiprows=header_row) # avoids preamble
            res = rx_data[r"R(OHM)"].to_numpy()
            react = rx_data[r"X(OHM)"].to_numpy()

            # extract temp data
            temp_data = pd.read_csv(TEMP_PATH)
            temp_data = temp_data.dropna().reset_index(drop=True) # drop NaN values
            temp = temp_data["Channel 1 Ave. (C)"].to_numpy()
            # convert time from hhmmss to ms
            time_parts = temp_data["Unnamed: 0"].str.split(":", expand=True)
            hours = pd.to_numeric(time_parts[0])
            minutes = pd.to_numeric(time_parts[1])
            seconds = pd.to_numeric(time_parts[2])
            total_ms = 1000 * (3600 * hours + 60 * minutes + seconds)
            # correct times in case times didnt start at 0
            temp_data['total_ms'] = total_ms - total_ms[0]
            t_times = temp_data['total_ms'].to_numpy()

            # generate synthetic rx times
            rx_times = np.linspace(0, t_times[-1], len(rx_data))

            # find rel rx
            rel_r = res/res[0]
            rel_x = react/react[0]

            # filter
            fil_rel_r = np.array(simple_filter(rel_r, 0.9))
            fil_rel_x = np.array(simple_filter(rel_x, 0.9))

            # interpolate to match time scale to t times:
            interp_fil_rel_r = np.interp(t_times, rx_times, fil_rel_r)
            interp_fil_rel_x = np.interp(t_times, rx_times, fil_rel_x)

            # curve fitting - quadratic
            pr = np.polyfit(temp, interp_fil_rel_r, 2) # resistance
            r_smooth = np.polyval(pr, temp)
            px = np.polyfit(temp, interp_fil_rel_x, 2) # reactance
            x_smooth = np.polyval(px, temp)

            pr_formatted = [float(to_4sf(p)) for p in pr]
            px_formatted = [float(to_4sf(p)) for p in px]

            # give equations
            print(f"{sample_name} - {freq}Hz")
            res_eqn = f"{pr[0]:.4g}x^2 + {pr[1]:.4g}x + {pr[2]:.4g}"
            x_eqn = f"{px[0]:.4g}x^2 + {px[1]:.4g}x + {px[2]:.4g}"
            print("\nResistance equation: ", res_eqn)
            print("Reactance equation: ", x_eqn, "\n", pr_formatted[0:3],
                  "\n", px_formatted[0:3])

            # give sensitivity values at start/end
            sens_r_start = 2 * pr[0] * temp[0] + pr[1]
            sens_r_end = 2 * pr[0] * temp[-1] + pr[1]
            sens_x_start = 2 * px[0] * temp[0] + px[1]
            sens_x_end = 2 * px[0] * temp[-1] + px[1]

            print(f"\nResistance temperature sensitivity (/deg C): Start: {sens_r_start:.4g}"
                  f"\nEnd: {sens_r_end:.4g}")
            print(f"\nReactance temperature sensitivity (/deg C): Start: {sens_x_start:.4g}"
                  f"\nEnd: {sens_x_end:.4g}")

            # use seaborn to make plots pretty
            plt.style.use('seaborn-v0_8-ticks')

            # temp vs rx graph
            fig, ax1 = plt.subplots(layout='constrained')
            fig.canvas.manager.set_window_title(f"{sample_name} - "
                                                f"{freq}Hz - Filtered RX vs temperature")
            line1, = ax1.plot(temp, interp_fil_rel_r, color='blue',
                              label='Relative resistance', alpha=0.7)
            ax1.set_title(f"{sample_name} - {freq}Hz - Filtered RX vs temperature")
            ax1.set_ylabel('Relative resistance')
            ax1.set_xlabel('Temperature (deg C)')
            line3, = ax1.plot(temp, r_smooth, color='deepskyblue', linestyle='dashed',
                              label='Relative resistance best fit line')
            ax2 = ax1.twinx()
            line2, = ax2.plot(temp, interp_fil_rel_x, color='red',
                              label='Relative reactance', alpha=0.7)
            ax2.set_ylabel('Relative Reactance')
            line4, = ax2.plot(temp, x_smooth, color='orange', linestyle='dashed',
                              label='Relative reactance best fit line')
            lines = [line1, line2, line3, line4]
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='best')

            # time series graph
            fig, ax1 = plt.subplots(layout="constrained")
            fig.canvas.manager.set_window_title(f"{sample_name} - {freq}Hz - Temperature "
                                                "and relative resistance/reactance vs time"
                                                )
            line1, = ax1.plot(t_times, temp, color='lime', label='Temperature (deg C)')
            ax1.set_title(f"{sample_name} - {freq}Hz - Temperature and RX time series")
            ax1.set_ylabel('Temperature (deg C)')
            ax1.set_xlabel('Time (ms)')
            ax2 = ax1.twinx()
            line2, = ax2.plot(rx_times, fil_rel_r, color='blue',
                              label='Filtered relative resistance', alpha=0.5)
            ax2.set_ylabel('Relative Resistance')
            ax3 = ax1.twinx()
            ax3.spines["right"].set_position(("axes", 1.2))
            line3, = ax3.plot(rx_times, fil_rel_x, color='red',
                              label='Filtered relative reactance', alpha=0.5)
            ax3.set_ylabel('Relative Reactance')
            lines = [line1, line2, line3]
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='best')

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
