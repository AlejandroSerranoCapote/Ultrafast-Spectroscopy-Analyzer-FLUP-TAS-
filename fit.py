# -*- coding: utf-8 -*-
import numpy as np
import os
from PyQt5.QtWidgets import QFileDialog
from scipy import special as _special
from scipy.optimize import least_squares

def load_npy(parent=None, normalize_per_wl=True):
    """
    Carga el archivo .npy, corrige baseline y devuelve matrices limpias.
    """
    file_path, _ = QFileDialog.getOpenFileName(parent, "Select treated data file", "", "NumPy files (*.npy)")
    if not file_path:
        raise ValueError("No file selected")
    
    data = np.load(file_path, allow_pickle=True).item()
    data_c = data['data_c'].astype(float) # Shape habitual: [Time, Wavelength]

    # 1) CORRECCIÓN DE LÍNEA BASE (Vital para el ajuste)
    # Asumimos que los primeros 10 puntos son antes del pulso láser (t < 0)
    # data_c = data_c - np.mean(data_c[:, :5], axis=1, keepdims=True)

    # 2) Normalización (Opcional, ayuda a la convergencia del fit)
    # Si no normalizas, los valores de A serán muy pequeños (mOD) y least_squares puede sufrir.
    # Es mejor ajustar datos en escala de OD o normalizados a 1.
    
    WL = data['WL'].flatten()
    TD = data['TD'].flatten()
    base_dir = os.path.dirname(file_path)
    
    return data_c, TD, WL, base_dir

def crop_spectrum(data_c, WL, WLmin, WLmax):
    mask = (WL >= WLmin) & (WL <= WLmax)
    return data_c[:, mask], WL[mask] # Ojo: data_c suele ser [WL, Time] o [Time, WL]. Revisa tu shape.

def crop_kinetics(data_c, TD, TDmin, TDmax):
    mask = (TD >= TDmin) & (TD <= TDmax)
    # Asumiendo data_c es [WL, Time], si es al revés cambia a data_c[mask, :]
    return data_c[:, mask], TD[mask] 

def binning(data_c, WL, bin_size):
    numWL = len(WL) // bin_size
    # Asumiendo data_c shape: [Num_WL, Num_Time] basado en tu bucle original
    # Si tu data_c es [Time, WL], ajusta los axis.
    datacAVG = np.zeros((numWL, data_c.shape[1]))
    WLAVG = np.zeros(numWL)
    for i in range(numWL):
        datacAVG[i, :] = np.mean(data_c[i*bin_size:(i+1)*bin_size, :], axis=0)
        WLAVG[i] = np.mean(WL[i*bin_size:(i+1)*bin_size])
    return datacAVG, WLAVG

def convolved_exp(t, t0, tau, w):
    """
    Analytical expression for convolution of single-exponential decay
    with Gaussian IRF.
    """
    t = np.asarray(t)
    tau = np.maximum(tau, 1e-12) # Evitar división por cero
    w = np.maximum(w, 1e-12)
    
    # La fórmula estándar
    arg1 = (w**2 - 2 * tau * (t - t0)) / (2 * tau**2)
    arg2 = (w**2 - tau * (t - t0)) / (np.sqrt(2) * w * tau)
    
    return 0.5 * np.exp(arg1) * (1 - _special.erf(arg2))

def eval_global_model(x, t, numExp, numWL, t0_choice_str):
    """
    Calcula la matriz del modelo.
    Estructura de x:
    - Si t0 variable (Chirp): [w, tau_1..tau_n,  (t0_wl1, A1_wl1..An_wl1), (t0_wl2...)...]
    - Si t0 fijo (Global):    [w, t0, tau_1..tau_n, (A1_wl1..An_wl1), (A1_wl2...)...]
    """
    F = np.zeros((len(t), numWL))
    
    if t0_choice_str == 'Yes': # CHIRP CORRECTION MODE (t0 varía por WL)
        w = x[0]
        taus = x[1:1+numExp]
        base_idx = 1 + numExp
        
        # Este bucle es necesario porque t0 cambia por columna, 
        # pero optimizado con numpy operations dentro
        for j in range(numWL):
            idx = base_idx + j*(numExp+1)
            t0 = x[idx]
            Amps = x[idx+1 : idx+1+numExp]
            
            # Suma de exponenciales para esta longitud de onda
            kinetics = np.zeros_like(t)
            for n in range(numExp):
                kinetics += Amps[n] * convolved_exp(t, t0, taus[n], w)
            F[:, j] = kinetics

    else: # STANDARD GLOBAL FIT (t0 único global) -> ¡VERSION VECTORIZADA RAPIDA!
        w = x[0]
        t0 = x[1]
        taus = x[2:2+numExp]
        A_base = 2 + numExp
        
        # 1. Pre-calcular las bases cinéticas (Time x NumExp)
        # Esto se hace una sola vez para todas las WLs
        basis_functions = np.zeros((len(t), numExp))
        for n in range(numExp):
            basis_functions[:, n] = convolved_exp(t, t0, taus[n], w)
            
        # 2. Extraer todas las amplitudes en una matriz (NumExp x NumWL)
        # x tiene [A1_wl1, A2_wl1, ..., A1_wl2, A2_wl2...]
        all_Amps = x[A_base:].reshape(numWL, numExp).T 
        
        # 3. Multiplicación matricial: [T, Exp] @ [Exp, WL] -> [T, WL]
        F = basis_functions @ all_Amps
        
    return F

def residuals(x, t, data, numExp, numWL, t0_choice_str):
    """Función de error para least_squares"""
    model = eval_global_model(x, t, numExp, numWL, t0_choice_str)
    # Aplanamos la diferencia para que sea un vector 1D, como pide least_squares
    return (model - data).flatten()

