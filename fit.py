# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 16:25:52 2025

@author: Alejandro
"""

import numpy as np
import os
from PyQt5.QtWidgets import QFileDialog
from scipy import special as _special
from scipy.optimize import least_squares

#FUNCION CON DATOS SIN NORMALIZAR (DA PROBLEMAS AL FITEAR)
# def load_npy(parent=None): 
#     """
#     Open a file dialog to select treated .npy data file and load it.
#     Returns: data_c, TD, WL, base_dir
#     """
#     file_path, _ = QFileDialog.getOpenFileName(parent, "Select treated data file", "", "NumPy files (*.npy)")
#     if not file_path:
#         raise ValueError("No file selected")
    
#     data = np.load(file_path, allow_pickle=True).item()
#     data_c = data['data_c']
#     data_c = data_c - np.mean(data_c[:, :10], axis=1, keepdims=True)  # baseline
#     data_c /= np.max(np.abs(data_c))  # normalize

#     WL = data['WL'].flatten()
#     TD = data['TD'].flatten()
#     base_dir = os.path.dirname(file_path)
    
#     return data_c, TD, WL, base_dir

def load_npy(parent=None, normalize_per_wl=True):
    """
    Open a file dialog to select treated .npy data file and load it.
    Returns: data_c, TD, WL, base_dir
    """
    file_path, _ = QFileDialog.getOpenFileName(parent, "Select treated data file", "", "NumPy files (*.npy)")
    if not file_path:
        raise ValueError("No file selected")
    
    data = np.load(file_path, allow_pickle=True).item()
    data_c = data['data_c'].astype(float)

    # 1) baseline correction per row
    data_c = data_c - np.mean(data_c[:, :10], axis=1, keepdims=True)

    # # 2) normalization
    # if normalize_per_wl:
    #     # normalize each wavelength (row) individually
    #     max_vals = np.max(np.abs(data_c), axis=1, keepdims=True)
    #     max_vals[max_vals == 0] = 1  # evitar división por cero
    #     data_c /= max_vals
    # else:
    #     # normalize globally
    #     data_c /= np.max(np.abs(data_c))
    
    WL = data['WL'].flatten()
    TD = data['TD'].flatten()
    base_dir = os.path.dirname(file_path)
    
    return data_c, TD, WL, base_dir
def crop_spectrum(data_c, WL, WLmin, WLmax):
    mask = (WL >= WLmin) & (WL <= WLmax)
    return data_c[mask, :], WL[mask]

def crop_kinetics(data_c, TD, TDmin, TDmax):
    mask = (TD >= TDmin) & (TD <= TDmax)
    return data_c[:, mask], TD[mask]

def binning(data_c, WL, bin_size):
    numWL = len(WL) // bin_size
    datacAVG = np.zeros((numWL, data_c.shape[1]))
    WLAVG = np.zeros(numWL)
    for i in range(numWL):
        datacAVG[i, :] = np.mean(data_c[i*bin_size:(i+1)*bin_size, :], axis=0)
        WLAVG[i] = np.mean(WL[i*bin_size:(i+1)*bin_size])
    return datacAVG, WLAVG

def convolved_exp(t, t0, tau, w):
    """
    Analytical expression for convolution of single-exponential decay
    with Gaussian instrument response (normalized, vectorized).
    Returns same shape as t.
    """
    # ensure arrays
    t = np.asarray(t)
    # avoid division by zero
    tau = np.maximum(tau, 1e-12)
    w = np.maximum(w, 1e-12)
    arg1 = (w**2 - 2 * tau * (t - t0)) / (2 * tau**2)
    arg2 = (w**2 - tau * (t - t0)) / (np.sqrt(2) * w * tau)
    # use safe erf from scipy.special
    return 0.5 * np.exp(arg1) * (1 - _special.erf(arg2))


def eval_global_model(x, t, numExp, numWL, t0_choice_str):
    F = np.zeros((len(t), numWL))
    if t0_choice_str == 'Yes':
        w = x[0]
        taus = x[1:1+numExp]
        base_idx = 1 + numExp
        for j in range(numWL):
            t0 = x[base_idx + j*(numExp+1)]
            A_start = base_idx + 1 + j*(numExp+1)
            for n in range(numExp):
                A = x[A_start + n]
                tau_n = taus[n]
                F[:, j] += A * convolved_exp(t, t0, tau_n, w)
    else:
        w = x[0]
        t0 = x[1]
        taus = x[2:2+numExp]
        A_base = 2 + numExp
        for j in range(numWL):
            A_start = A_base + j*numExp
            for n in range(numExp):
                A = x[A_start + n]
                tau_n = taus[n]
                F[:, j] += A * convolved_exp(t, t0, tau_n, w)
    return F


def run_fit(data_c, TD, numExp, numWL, ini, limi, lims, t0_choice):
    def residuals(x):
        F = eval_global_model(x, TD, numExp, numWL, t0_choice)
        return F.flatten() - data_c.T.flatten()
    
    result = least_squares(residuals, ini, bounds=(limi, lims), jac='2-point', verbose=2)
    
    x = result.x
    fitres = eval_global_model(x, TD, numExp, numWL, t0_choice).T
    resid = data_c - fitres
    return x, fitres, resid, result

