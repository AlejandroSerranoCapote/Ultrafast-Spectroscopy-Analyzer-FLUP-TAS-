# -*- coding: utf-8 -*-
import numpy as np
import os
from PyQt5.QtWidgets import QFileDialog
from scipy import special as _special

def load_npy(parent=None, normalize_per_wl=True):
    """
    Carga el archivo .npy, corrige baseline y devuelve matrices limpias.
    """
    file_path, _ = QFileDialog.getOpenFileName(parent, "Select treated data file", "", "NumPy files (*.npy)")
    if not file_path:
        raise ValueError("No file selected")
    
    data = np.load(file_path, allow_pickle=True).item()
    data_c = data['data_c'].astype(float) 
    
    WL = data['WL'].flatten()
    TD = data['TD'].flatten()
    base_dir = os.path.dirname(file_path)
    
    return data_c, TD, WL, base_dir

def crop_spectrum(data_c, WL, WLmin, WLmax):
    mask = (WL >= WLmin) & (WL <= WLmax)
    return data_c[:, mask], WL[mask] 

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
    with Gaussian IRF.
    """
    t = np.asarray(t)
    tau = np.maximum(tau, 1e-12) # Evitar división por cero
    w = np.maximum(w, 1e-12)
    
    arg1 = (w**2 - 2 * tau * (t - t0)) / (2 * tau**2)
    arg2 = (w**2 - tau * (t - t0)) / (np.sqrt(2) * w * tau)
    # --- PROTECCIÓN CONTRA OVERFLOW ---

    arg1 = np.clip(arg1, -700, 700)
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
        
        for j in range(numWL):
            idx = base_idx + j*(numExp+1)
            t0 = x[idx]
            Amps = x[idx+1 : idx+1+numExp]
            
            # Suma de exponenciales para esta longitud de onda
            kinetics = np.zeros_like(t)
            for n in range(numExp):
                kinetics += Amps[n] * convolved_exp(t, t0, taus[n], w)
            F[:, j] = kinetics

    else: # STANDARD GLOBAL FIT (t0 único global) 
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

def get_sequential_populations(t, t0, w, taus):
    """ Calculates populations for a sequential model A -> B -> C... """
    k = 1.0 / np.asarray(taus) # Rates
    pops = []
    
    E = [convolved_exp(t, t0, tau, w) for tau in taus]
    
    # --- Species 1 ---
    pops.append(E[0])
    
    # --- Species 2 ---
    if len(taus) >= 2:
        denom = k[1] - k[0]
        if abs(denom) < 1e-9: denom = 1e-9 
        factor = k[0] / denom
        p2 = factor * (E[0] - E[1])
        pops.append(p2)
        
    # --- Species 3 ---
    if len(taus) >= 3:
        k0, k1, k2 = k[0], k[1], k[2]
        d0 = (k[1]-k[0]) * (k[2]-k[0])
        d1 = (k[0]-k[1]) * (k[2]-k[1])
        d2 = (k[0]-k[2]) * (k[1]-k[2])
        if abs(d0) < 1e-9: d0 = 1e-9
        if abs(d1) < 1e-9: d1 = 1e-9
        if abs(d2) < 1e-9: d2 = 1e-9
        p3 = (k0 * k1) * ( (E[0]/d0) + (E[1]/d1) + (E[2]/d2) )
        pops.append(p3)

    return pops


def eval_sequential_model(x, t, numExp, numWL, t0_choice_str):
    """
    Calcula la matriz del modelo SECUENCIAL (A -> B -> C...).
    
    Interpretación de los parámetros 'Amps' en x:
    En modelo paralelo: Amps = DAS (Decay Associated Spectra)
    En modelo secuencial: Amps = SAS (Species Associated Spectra) o EADS
    
    Estructura de x (idéntica a tu modelo global):
    - Chirp (Yes): [w, tau_1..n, (t0_wl1, SAS1_wl1..SASn_wl1), ...]
    - Global (No): [w, t0, tau_1..n, (SAS1_wl1..SASn_wl1), (SAS1_wl2...)...]
    """
    F = np.zeros((len(t), numWL))
    
    # --- MODO CHIRP CORRECTION (t0 varía por WL) ---
    if t0_choice_str == 'Yes': 
        w = x[0]
        taus = x[1:1+numExp]
        base_idx = 1 + numExp
        
        for j in range(numWL):
            idx = base_idx + j*(numExp+1)
            t0 = x[idx]
            # Aquí 'Amps' son los coeficientes espectrales de las especies para esta WL
            sas_coeffs = x[idx+1 : idx+1+numExp] 
            
            # 1. Calcular las poblaciones para este t0 específico
            # Devuelve una lista [PopA, PopB, PopC...]
            pops_list = get_sequential_populations(t, t0, w, taus)
            
            # 2. Combinación lineal: Suma(Poblacion_i * Espectro_i)
            kinetics = np.zeros_like(t)
            for n in range(numExp):
                kinetics += sas_coeffs[n] * pops_list[n]
            
            F[:, j] = kinetics

    # --- MODO GLOBAL STANDARD (t0 fijo)
    else: 
        w = x[0]
        t0 = x[1]
        taus = x[2:2+numExp]
        A_base = 2 + numExp
        
        # 1. Calcular la base cinética (Poblaciones) UNA SOLA VEZ
        # get_sequential_populations devuelve lista de arrays 1D.
        pops_list = get_sequential_populations(t, t0, w, taus)
        basis_functions = np.column_stack(pops_list) 
        
        # 2. Extraer todos los espectros (SAS) en una matriz (NumSpecies x NumWL)
        # x contiene [S1_wl1, S2_wl1, ..., S1_wl2, S2_wl2...]
        all_SAS = x[A_base:].reshape(numWL, numExp).T 
        
        # 3. Multiplicación matricial:
        # [T, Species] @ [Species, WL] -> [T, WL]
        F = basis_functions @ all_SAS
        
    return F


