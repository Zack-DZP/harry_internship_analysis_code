# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import curve_fit

def gen_trendline(domain, func):
    x_dummy = np.linspace(domain[0], domain[-1], 300)
    y_dummy = func(x_dummy)
    return x_dummy, y_dummy

def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c

def fit_exp(x, y):
    # initial guesses using data
    c_guess = y[-1]
    a_guess = y[0] - c_guess if (y[0] - c_guess) != 0 else 1.0
    b_guess = 0.1 if y[-1] > y[0] else -0.1 # decide decay or growth
    initial_guess = [a_guess, b_guess, c_guess]
    try:
        popt, _ = curve_fit(exp_func, x, y, p0=initial_guess, maxfev=20000)
    
    except RuntimeError as e:
        raise RuntimeError("The optimization failed to converge.") from e
        
    # helper func using parameters
    fit_fn = lambda x_val: exp_func(x_val, *popt)
    
    # calculate r2 values
    y_pred = exp_func(x, *popt)
    ss_res = np.sum((y - y_pred) ** 2)          # residual sum of squares
    ss_tot = np.sum((y - np.mean(y)) ** 2)      # total sum of squares
    r2 = 1 - (ss_res / ss_tot)
    
    return popt, fit_fn, r2

def ln_func(x, a, b, c):
    return a * np.log(x - b) + c

def fit_ln(x, y):
    
    # establish bounds for the optimiser (ln cannot be negative)
    buffer = 1.0e-5
    max_b = np.min(x) - buffer
    bounds = ([-np.inf, -np.inf, -np.inf], [np.inf, max_b, np.inf])
    
    # initial guesses
    b_guess = np.min(x) - 1
    # Guard against b_guess becoming invalid if min(x) is small/negative
    if b_guess >= np.min(x):
        b_guess = np.min(x) - buffer
    c_guess = np.mean(y)
    a_guess = 1.0
    initial_guess = [a_guess, b_guess, c_guess]
    
    # fit curves
    try:
        popt, _ = curve_fit(ln_func, x, y, p0=initial_guess, 
                            bounds=bounds, maxfev=20000)
    except RuntimeError as e:
        raise RuntimeError("The optimization failed to converge.") from e
    
    # calculate r2
    y_pred = ln_func(x, *popt)
    ss_res = np.sum((y - y_pred) ** 2)          # residual sum of squares
    ss_tot = np.sum((y - np.mean(y)) ** 2)      # total sum of squares
    r2 = 1 - (ss_res / ss_tot)
    
    # create lambda func
    fit_fn = lambda x_val: ln_func(x_val, *popt)

    return popt, fit_fn, r2