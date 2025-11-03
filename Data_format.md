# 📁 Formatos de datos admitidos y archivos generados

Este documento describe los formatos de archivo que el **Ultrafast Spectroscopy Analyzer** puede importar y exportar.

---

## 🧩 Datos de entrada

### 🔹 FLUPS (*Fluorescence Up-Conversion Spectroscopy*)
Se admite un único archivo `.csv` o `.txt` con el siguiente formato:

| Fila / Columna | Contenido                |
|----------------|--------------------------|
| Primera fila   | Valores de *delay* (ps)  |
| Primera columna| Longitudes de onda (nm)  |
| Resto          | Matriz ΔA(λ, t)          |

Ejemplo:

---

### 🔹 TAS (*Transient Absorption Spectroscopy*)
Se requieren **dos archivos**:

1. **Medida experimental** (`sample.csv` o `.txt`)  
2. **Medida del solvente** (`solvent.csv` o `.txt`)  

Ambos deben tener la misma estructura que en FLUPS:
- Fila 1 → delays  
- Columna 1 → longitudes de onda  
- Celdas → ΔA(λ, t)

El programa combina ambas matrices, resta el solvente, y aplica las correcciones definidas por el usuario.

---

## 📦 Archivos generados automáticamente

Tras ejecutar un ajuste de *t₀* o un análisis global, se crea una carpeta:

## 🧠 Notas adicionales

- Los archivos `.npy` pueden cargarse directamente en Python con `numpy.load()`.  
- Las versiones en texto (`.txt`, `.csv`) están normalizadas para compatibilidad con **Origin**, **Igor Pro**, **MATLAB** y **Python**.  
- Los nombres de las carpetas se generan automáticamente según el archivo de entrada.

---
