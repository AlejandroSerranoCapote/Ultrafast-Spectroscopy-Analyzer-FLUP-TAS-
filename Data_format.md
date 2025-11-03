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
