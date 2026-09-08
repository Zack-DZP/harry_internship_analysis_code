# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from data_processing2 import split_curve

def gen_trendline(xdata, modelname, data):
    # generate x values for evaluation
    x_spaced = np.linspace(np.min(xdata), np.max(xdata), 100)

    # in case of earlier curve fitting complete failure
    if data['r2'] == -1:
        return None, None

    # fit
    try:
        if modelname == 'Linear':
            p = np.polyfit(xdata, data['_raw_y'], 1)
            y_smooth = np.polyval(p, x_spaced)
        elif modelname == 'Quadratic':
            p = np.polyfit(xdata, data['_raw_y'], 2)
            y_smooth = np.polyval(p, x_spaced)
        else:
            return None, None
        return x_spaced, y_smooth
    except Exception:
        return None, None

def plot_time_series(sample, freq, ftimes, force, rxtimes, relr, relx):

    fig, ax1 = plt.subplots(layout="constrained")
    fig.canvas.manager.set_window_title(f"{sample} - {freq}Hz - Force and (filtered)"
                                        " relative resistance/reactance vs time")
    line1, = ax1.plot(ftimes, force, color='blue', label='Force (g)')
    ax1.set_title(f"{sample} - {freq}Hz - Force and relative resistance/reactance vs time")
    ax1.set_ylabel('Force (g)')
    ax1.set_xlabel('Time (ms)')
    ax2 = ax1.twinx()
    line2, = ax2.plot(rxtimes, relr, color='red', label='Filtered relative resistance', alpha=0.5)
    ax2.set_ylabel('Relative Resistance')

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.2))
    line3, = ax3.plot(rxtimes, relx, color='lime', label='Filtered relative reactance', alpha=0.5)
    ax3.set_ylabel('Relative Reactance')
    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')

def plot_hysteresis_loop(sample, freq, channel, avg_force, avg_relx,
                         leaderboards_up, leaderboards_down,
                         lower_bound, changeover_bound, hyst_ratio_x):

    plt.figure(f"{sample} - {freq}Hz - Relative {channel} vs force", layout="constrained")
    plt.plot(avg_force, avg_relx, marker='x', linestyle='-',
             color='blue', alpha=0.5, label='Actual Loop Data')
    plt.ylabel(f"Relative {channel}")
    plt.xlabel("Force (g)")
    plt.grid(True)

    if channel == 'resistance':
        ranked_up_x1 = leaderboards_up[0][1]
        ranked_up_x2 = leaderboards_up[1][1]
        ranked_down_x1 = leaderboards_down[0][1]
        ranked_down_x2 = leaderboards_down[1][1]
    elif channel == 'reactance':
        ranked_up_x1 = leaderboards_up[2][1]
        ranked_up_x2 = leaderboards_up[3][1]
        ranked_down_x1 = leaderboards_down[2][1]
        ranked_down_x2 = leaderboards_down[3][1]

    # Extract curve split components
    f_up, f_down, x_up, x_down, _ = split_curve(avg_force, avg_relx)

    # slice to find single best winner
    if ranked_up_x1 and len(ranked_up_x1) > 0:
        best_up_name, best_up_data = ranked_up_x1[0]
        mask_up = (f_up >= lower_bound) & (f_up <= changeover_bound)
        f_up_filtered = f_up[mask_up]
        r_up_filtered = x_up[mask_up]
        # Extracts top model tuple cleanly
        best_up_data['_raw_y'] = r_up_filtered
        x_fit_up, y_fit_up = gen_trendline(f_up_filtered, best_up_name, best_up_data)
        if x_fit_up is not None:
            plt.plot(x_fit_up, y_fit_up, color='limegreen', linestyle='--', linewidth=2,
                     label=f'Loading Fit (1): {best_up_name} (R²={best_up_data["r2"]:.3f})')

    if ranked_down_x1 and len(ranked_down_x1) > 0:
        best_down_name, best_down_data = ranked_down_x1[0]
        mask_down = (f_down >= lower_bound) & (f_down <= changeover_bound)
        f_down_filtered = f_down[mask_down]
        r_down_filtered = x_down[mask_down]
        best_down_data['_raw_y'] = r_down_filtered # Extracts top model tuple cleanly
        x_fit_down, y_fit_down = gen_trendline(f_down_filtered, best_down_name, best_down_data)
        if x_fit_down is not None:
            plt.plot(x_fit_down, y_fit_down, color='firebrick', linestyle='--', linewidth=2,
                     label=f'Unloading Fit (1): {best_down_name} (R²={best_down_data["r2"]:.3f})')

    if ranked_up_x2 and len(ranked_up_x2) > 0:
        best_up_name, best_up_data = ranked_up_x2[0]
        mask_up = f_up >= changeover_bound
        f_up_filtered = f_up[mask_up]
        r_up_filtered = x_up[mask_up]
        # Extracts top model tuple cleanly
        best_up_data['_raw_y'] = r_up_filtered
        x_fit_up, y_fit_up = gen_trendline(f_up_filtered, best_up_name, best_up_data)
        if x_fit_up is not None:
            plt.plot(x_fit_up, y_fit_up, 'g--', linewidth=2,
                     label=f'Loading Fit (2): {best_up_name} (R²={best_up_data["r2"]:.3f})')

    if ranked_down_x2 and len(ranked_down_x2) > 0:
        best_down_name, best_down_data = ranked_down_x2[0]
        mask_down = f_down >= changeover_bound
        f_down_filtered = f_down[mask_down]
        r_down_filtered = x_down[mask_down]
        best_down_data['_raw_y'] = r_down_filtered # Extracts top model tuple cleanly
        x_fit_down, y_fit_down = gen_trendline(f_down_filtered, best_down_name, best_down_data)
        if x_fit_down is not None:
            plt.plot(x_fit_down, y_fit_down, 'r--', linewidth=2,
                     label=f'Unloading Fit (2): {best_down_name} (R²={best_down_data["r2"]:.3f})')

    # text boxes
    plt.text(avg_force[0], avg_relx[0], ' Start ', ha='left', va='center',
             fontsize=12, color='red')
    plt.text(avg_force[-1], avg_relx[-1], ' End', ha='left', va='center',
             fontsize=12, color='red')
    plt.title(f"{sample} - {freq}Hz - Relative {channel} vs force")
    plt.legend(loc='lower right', fancybox=True, framealpha=0.5)

    box_text = f"Hysteresis ratio: {hyst_ratio_x:.4f}"
    plt.text(
        0.95, 0.95, box_text,
        transform=plt.gca().transAxes,
        ha='right', va='top',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9)
    )
