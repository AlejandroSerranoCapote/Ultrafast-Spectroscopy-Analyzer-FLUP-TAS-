# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 11:17:39 2025

@author: Alejandro
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def eV_a_nm(E_eV):
    """Convierte energía (eV) a longitud de onda (nm)."""
    E_eV_safe = np.where(E_eV == 0, np.inf, E_eV)
    return 1239.841984 / E_eV_safe

# === CARGAR DATOS ===
ruta_base = "C:/Users/Alejandro/OneDrive - Universidad de Oviedo/Escritorio/PhD/Software/UltrafastSpectroscopyAnalyzer/TASImdea/"

#Lectura datos
data = np.loadtxt(ruta_base + "muestra.dat")
data_sol = np.loadtxt(ruta_base + "disolvente.dat")

# === PROCESAMIENTO ===
wl = eV_a_nm(data[:, 0])    # Longitudes de onda (nm)
t = data[0] * 1e-3          # Tiempos (ps)

# === REEMPLAZAR DENTRO DE LAS MISMAS MATRICES ===
data[:, 0] = wl         # Sustituir primera columna por wl
data[0, :] = t           # Sustituir primera fila por t

data_sol[:, 0] = wl
data_sol[0, :] = t


