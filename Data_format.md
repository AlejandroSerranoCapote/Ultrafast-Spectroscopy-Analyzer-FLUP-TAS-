# 📁 Formatos de datos admitidos y archivos generados

Este documento describe los formatos de archivo que el **Ultrafast Spectroscopy Analyzer** puede importar y exportar.

---

## 🧩 Datos de entrada

### 🔹 FLUPS (*Fluorescence Up-Conversion Spectroscopy*)
Se admite un único archivo `.csv` con el siguiente formato:

| Fila / Columna | Contenido                |
|----------------|--------------------------|
| Primera fila   | Valores de *delay* (ps)  |
| Primera columna| Longitudes de onda (nm)  |
| Resto          | Matriz ΔA(λ, t)          |

Ejemplo:
```math
\text{ΔA(λ,t)} =
\begin{bmatrix}
λ \setminus t & -1.00 & -0.50 & 0.00 & 0.50 & 1.00 \\
400 & 0.002 & 0.005 & 0.010 & 0.004 & 0.001 \\
410 & 0.001 & 0.004 & 0.008 & 0.003 & 0.000
\end{bmatrix}
```
También se admiten 3 archivos `.txt` de la siguiente forma:
```math
\text{ΔA(λ,t)} =
\begin{bmatrix}
 0.002 & 0.005 & 0.010 & 0.004 & 0.001 \\
0.001 & 0.004 & 0.008 & 0.003 & 0.000
\end{bmatrix}
```
```math
\text{λ} =
\begin{bmatrix}
  450& -475 & 500 & 525 & 550 & ... 
\end{bmatrix}
```
```math
\text{t} =
\begin{bmatrix}
  -1.00 & -0.50 & 0.00 & 0.50 & 1.00  & ... 
\end{bmatrix}
```
---

### 🔹 TAS (*Transient Absorption Spectroscopy*)
Se requieren **dos archivos**:

1. **Medida experimental** (`sample.csv`)  
2. **Medida del solvente** (`solvent.csv`)  

Ambos deben tener la misma estructura que en FLUPS (`.csv`):
- Fila 1 → delays  
- Columna 1 → longitudes de onda  
- Celdas → ΔA(λ, t)

Ejemplo:
```math
\text{ΔA(λ,t)} =
\begin{bmatrix}
λ \setminus t & -1.00 & -0.50 & 0.00 & 0.50 & 1.00 \\
400 & 0.002 & 0.005 & 0.010 & 0.004 & 0.001 \\
410 & 0.001 & 0.004 & 0.008 & 0.003 & 0.000
\end{bmatrix}
```

El programa combina ambas matrices, resta el solvente, y aplica las correcciones definidas por el usuario.

---

## 📦 Archivos generados automáticamente

Tras ejecutar un ajuste de *t₀* y un análisis global, se crea una carpeta:

```text
<nombre_archivo>_Results/
│
├── WL.txt                 → Longitudes de onda (nm)
├── TD.txt                 → Delays (ps)
├── treated_data.npy       → Datos corregidos en formato NumPy
├── t0_fit.txt             → Curva de ajuste t₀(λ)
├── fit_params.txt         → Parámetros del modelo de ajuste
├── kin.txt                → Cinéticas (ΔA vs tiempo)
├── spec.txt               → Espectros (ΔA vs λ)
│
├── Fit/                   → Carpeta con los resultados del global fit
│   ├── Amplitudes.txt     → Amplitudes del Decay Associates Spectra 
│   ├── GFit_resid.txt     → Residuals del ajuste de la cinética
│   ├── GFit.txt           → Ajuste de la cinética para todas las longitudes de onda
│   ├── GFitResults.npy    → .npy diccionario de NumPy con todos los datos
│   ├── TD.txt             → Delays (ps)
│   └── WL.txt             → Longitudes de onda (nm)
│
└── Plots/                 → Carpeta con los plots del ajuste
    ├── DAS.png            → Plot del Decay Associated Spectra (DAS)
    ├── Fit_xxxnm.png      → Plot del ajuste a la cinética de una λ determinada
    ├── Fit_xxxnm.txt      → Resultados del ajuste a la cinética de una λ determinada
    └── Residual.png       → Plot de los residuals del ajuste
```

## 🧠 Notas adicionales

- Los archivos `.npy` pueden cargarse directamente en Python con `numpy.load()`.  
- Las versiones en texto (`.txt`, `.csv`) están normalizadas para compatibilidad con **Origin**, **Igor Pro**, **MATLAB** y **Python**.  
- Los nombres de las carpetas se generan automáticamente según el archivo de entrada.

---
