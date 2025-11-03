# Ultrafast Spectroscopy Analyzer ⚡🔬

**Ultrafast Spectroscopy Analyzer** es un software gratuito y de código abierto diseñado para el procesamiento y análisis de datos de espectroscopía ultrarrápida.  
Permite trabajar con dos técnicas experimentales complementarias:

- **FLUPS** — *Fluorescence Up-Conversion Spectroscopy*  
- **TAS** — *Transient Absorption Spectroscopy*

La aplicación integra un entorno gráfico interactivo que facilita la corrección del chirp temporal (*t₀*), la sustracción del solvente, la eliminación del *pump scattering* y la visualización en tiempo real del mapa espectro-temporal.  
Además, permite realizar análisis globales ajustando los datos a un modelo multiexponencial, obteniendo los **DAS (Decay-Associated Spectra)** correspondientes a cada componente cinética.

---

## ✨ Características principales

✅ Interfaz gráfica intuitiva basada en **PyQt5**  
✅ Visualización dinámica del mapa ΔA(λ, t)  
✅ Selección interactiva de puntos *t₀* sobre el mapa  
✅ Ajuste de *t₀* mediante modelos **polinómicos** o **no lineales**  
✅ Corrección automática del solvente (modo TAS)  
✅ Eliminación del *pump fringe* o dispersión del láser de bombeo  
✅ Visualización simultánea de **cinéticas y espectros** bajo el cursor  
✅ Ajuste global multiexponencial con generación de **DAS**  
✅ Exportación automática y estructurada de todos los resultados corregidos  

---

> 📘 Consulta también: [Formatos de datos admitidos →](./Data_format.md)

## 🖥️ Capturas de pantalla

> *GUI FLUPS*
<img width="1394" height="932" alt="Foto1" src="https://github.com/user-attachments/assets/ab6397c5-5751-4c59-858c-83ba9da74b67" />

> *GUI TAS*
<img width="1399" height="930" alt="image" src="https://github.com/user-attachments/assets/989b08e7-b13c-4fc4-bcf5-53e976dd429b" />

> *GUI Global Fit*
<p align="center">
   <img src="https://github.com/user-attachments/assets/7effdce7-a700-4892-be37-54eac1b0866c" width="48%">
   <img src="https://github.com/user-attachments/assets/b103c26c-9a2b-42e3-977e-83fe45f9ab6e" width="48%">
 </p>

> *Decay Associated Spectra*
<img width="788" height="666" alt="image" src="https://github.com/user-attachments/assets/b84d6776-b94d-4424-9ddf-70cdac77e1dc" />

> *Kinetics Fit*
<img width="891" height="464" alt="image" src="https://github.com/user-attachments/assets/28caddd6-b46c-4981-b36c-5d3dd7228ea0" />


