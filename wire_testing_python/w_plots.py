# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

def plot_res_series(res, res_event_indices, plot_info):
    plt.figure(f'{plot_info[4]} - {plot_info[0]}, {plot_info[1]}Hz, '
               f'{(plot_info[3]):.1%} strain', layout='constrained')
    plt.title(f'{plot_info[4]} - {plot_info[0]} \n{plot_info[1]}Hz, '
              f'{(plot_info[3]):.1%} strain')
    plt.plot(res, label=plot_info[4])
    plt.plot(res_event_indices, res[res_event_indices], 'x',
             label='Peaks detected')
    plt.xlabel('Measurement number')
    plt.ylabel(f'{plot_info[4]} ({plot_info[5]})')
    plt.legend()
    
def plot_midline_series(cycle_nums, midline_vals, res, res_event_indices,
                        ml_exp_x, ml_exp_y, ml_ln_x, ml_ln_y,
                        plot_info):
    plt.figure(f'Midline {plot_info[4].lower()} - {plot_info[0]}, {plot_info[1]}Hz, '
               f'{(plot_info[3]):.1%} strain', layout='constrained')
    plt.title(f'Midline {plot_info[4].lower()} - {plot_info[0]} \n{plot_info[1]}Hz, '
              f'{(plot_info[3]):.1%} strain')
    plt.plot(cycle_nums, midline_vals, label=f'Midline {plot_info[4].lower()}')
    plt.plot(cycle_nums, res[res_event_indices[:-1]], 'x', label='Peak values')
    plt.plot(ml_exp_x, ml_exp_y, label='Exponential fit')
    plt.plot(ml_ln_x, ml_ln_y, label='Logarithmic fit', color='orange')
    plt.xlabel('Cycle number')
    plt.ylabel(f'{plot_info[4]} ({plot_info[5]})')
    plt.legend()
    
def plot_amplitude(cycle_nums, ptp_res, amp_exp_x, amp_exp_y,
                   amp_ln_x, amp_ln_y, plot_info):
    plt.figure(f'{plot_info[4]} oscillation amplitude - {plot_info[0]}, '
               f'{plot_info[1]}Hz, {(plot_info[3]):.1%} strain',
               layout='constrained')
    plt.title(f'{plot_info[4]} oscillation amplitude - {plot_info[0]}' 
              f'\n{plot_info[1]}Hz, {(plot_info[3]):.1%} strain')
    plt.plot(cycle_nums, ptp_res, label='Amplitude')
    plt.plot(amp_exp_x, amp_exp_y, label='Exponential fit')
    plt.plot(amp_ln_x, amp_ln_y, label='Logarithmic fit')
    plt.xlabel('Cycle number')
    plt.ylabel(f'{plot_info[4]} oscillation amplitude ({plot_info[5]})')
    plt.legend()
    
