# -*- coding: utf-8 -*-

import numpy as np
from data_processing2 import split_curve
import math

def evaluate_fit(x, y):
    # allows R2 evaluation for a single phase at a time
    if len(x) < 3:  # in case of too few points - very unlikely to ever occur
        return [('Fit Failed', {'r2': -1, 'formula': "Insufficient Data", 'params': np.array([])})]

    y_mean = np.mean(y)
    tss = np.sum((y_mean - y) ** 2)
    results = {}
    
    # linear 
    try:
        p = np.polyfit(x, y, 1)
        r2 = 1 - (np.sum((y - np.polyval(p, x)) ** 2) / tss) if tss != 0 else 0
        results['Linear'] = {'r2': r2, 'formula': f"y={p[0]:.4g}x + {p[1]:.4g}", 'params': p}
    except Exception:
        results['Linear'] = {'r2': -1, 'formula': "Failure", 'params': np.array([])}
        
    # quadratic 
    try:
        p = np.polyfit(x, y, 2)
        r2 = 1 - (np.sum((y - np.polyval(p, x)) ** 2) / tss) if tss != 0 else 0
        results['Quadratic'] = {'r2': r2, 'formula': f"y={p[0]:.4g}x^2 + {p[1]:.4g}x + {p[2]:.4g}", 'params': p}
    except Exception:
        results['Quadratic'] = {'r2': -1, 'formula': "Failure", 'params': np.array([])}
    
    format_p = [(to_4sf(x)) for x in p]
    print(format_p)
    # Returns the complete ranked leaderboard list of all fitted items
    return sorted(results.items(), key=lambda item: item[1]['r2'], reverse=True)

# creates R2 values
def hyst_fit(avg_force, avg_relres):
    f_up, f_down, r_up, r_down, fmax = split_curve(avg_force, avg_relres)
    
    ranked_up = evaluate_fit(f_up, r_up)
    ranked_down = evaluate_fit(f_down, r_down)
    
    return ranked_up, ranked_down, fmax
def sens_values(model_name, data_dict, x_data):
    
    params = data_dict.get('params', None)
    
    if params is None or len(params) == 0 or data_dict.get('r2', -1) == -1: 
        return None
    
    x_min = float(np.min(x_data))
    x_max = float(np.max(x_data))
    
    if model_name == 'Quadratic':
        a = params[0]
        b = params[1]
        
        # calculate sensitivity at start and end for reference
        sens_start = 2 * a * x_min + b
        sens_end = 2 * a * x_max + b
        
        return sens_start, sens_end
    
    elif model_name == 'Linear':
        slope = params[0]
        
        return slope, slope
    
    else:
        return False

def calculate_sensitivity(model_name, data_dict, x_data):
    
    # use above functions
    
    result = sens_values(model_name, data_dict, x_data)
    
    if result is None:
        return "  - Sensitivity Equation: Could not calculate (Fit Failed)"
    if result is False:
        return "  - Sensitivity: Model type not configured."
        
    sens_start, sens_end = result
    x_min, x_max = float(np.min(x_data)), float(np.max(x_data))
    params = data_dict['params']

    if model_name == 'Linear':
        return (
            f"  - Sensitivity Equation: dR/dF = {sens_start:.4g}\n"
            f"  - Constant Sensitivity across range: {sens_start:.4g} /g"
        )
        
    elif model_name == 'Quadratic':
        a, b = params[0], params[1]
        return (
            f"  - Sensitivity Equation: dR/dF = {2*a:.4g}*F + ({b:.4g})\n"
            f"  - Sensitivity at Start ({x_min:.1f}g): {sens_start:.4g} /g\n"
            f"  - Sensitivity at End ({x_max:.1f}g): {sens_end:.4g} /g"
        )
    
def to_4sf(num):
    if num == 0:
        return "0.000"
    
    # Calculate the scale / order of magnitude of the number
    magnitude = math.floor(math.log10(abs(num)))
    decimals = 3 - magnitude
    
    # Force fixed-point format using the calculated decimals
    if decimals < 0:
        return f"{num:.0f}"  # Large integers don't need trailing decimals
    return f"{num:.{decimals}f}"
        

