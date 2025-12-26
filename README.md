# Ultrafast Spectroscopy Analyzer 
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**Ultrafast Spectroscopy Analyzer** es un software gratuito y de código abierto diseñado para el procesamiento y análisis de datos de espectroscopía ultrarrápida.  
Permite trabajar con dos técnicas experimentales:

- **FLUPS** — *Fluorescence Up-Conversion Spectroscopy*  
- **TAS** — *Transient Absorption Spectroscopy*

La aplicación integra un entorno gráfico interactivo que facilita la corrección del chirp temporal (*t₀*), la sustracción del solvente, la eliminación del *pump scattering* y la visualización en tiempo real del mapa espectro-temporal.  

##  Modelos Matemáticos

El software ajusta la señal experimental $\Delta A(t, \lambda)$ utilizando dos aproximaciones principales, ambas convolucionadas con la Función de Respuesta del Instrumento (IRF).

### 1. Modelo Paralelo: Decay-Associated Spectra (DAS)
Asume que los componentes decaen de forma independiente, ideal para mezclas de especies no acopladas.

$$\Delta A(t, \lambda) = IRF(t) \otimes \sum_{i=1}^{n} A_i(\lambda) e^{-t/\tau_i}$$

Donde cada $A_i(\lambda)$ representa el **DAS** de la componente con tiempo de vida $\tau_i$.

### 2. Modelo Secuencial: Species-Associated Spectra (SAS)
Describe una cascada de energía o reacción consecutiva: $1 \xrightarrow{k_1} 2 \xrightarrow{k_2} \dots \xrightarrow{k_n} n$.  
Las poblaciones de cada especie se rigen por las **Ecuaciones de Bateman**.

Para una cadena de decaimiento donde $k_i = 1/\tau_i$, la concentración $C_n(t)$ de la especie $n$ se define como:

$$C_n(t) = \left( \prod_{j=1}^{n-1} k_j \right) \sum_{j=1}^{n} \frac{e^{-k_j t}}{\prod_{p=1, p \neq j}^{n} (k_p - k_j)}$$

La señal total es la suma de las contribuciones de cada estado excitado (SAS):

$$\Delta A(t, \lambda) = IRF(t) \otimes \sum_{i=1}^{n} SAS_i(\lambda) C_i(t)$$

###  Función de Respuesta del Instrumento (IRF)
La resolución temporal se modela mediante una Gaussiana de ancho $w$ (FWHM) centrada en $t_0$:

$$IRF(t) = \frac{1}{w \sqrt{\pi}} \exp\left( -\left( \frac{t - t_0}{w} \right)^2 \right)$$

---
>  **Instala las dependencias necesarias con el siguiente comando (dentro de la carpeta donde está el script):**
> ```bash
> pip install -r requirements.txt
> ```
>  **Para ejecutarlo, en la consola de comandos escribimos lo siguiente (dentro de la carpeta donde está el script):**
> ```bash
> python "Ultrafast Spectroscopy Analyzer.py"
> ```
> 
>  **Crear un .exe del script (Ejecutar en el cmd dentro de la carpeta donde está el script)**:
> ```bash
> pyinstaller --onefile --noconsole --icon=icon.ico --exclude-module PyQt6 "Ultrafast Spectroscopy Analyzer.py"
> ```

---
##  Características principales

 Interfaz gráfica intuitiva basada en **PyQt5**  
 Visualización dinámica del mapa ΔA(λ, t)  
 Selección interactiva de puntos *t₀* sobre el mapa  
 Ajuste de *t₀* mediante modelos **polinómicos** o **no lineales**  
 Corrección automática del solvente (modo TAS)  
 Eliminación del *pump fringe* o dispersión del láser de bombeo  
 Visualización simultánea de **cinéticas y espectros** bajo el cursor  
 Ajuste global multiexponencial con generación de **DAS**  
 Exportación automática y estructurada de todos los resultados corregidos  

---

>  Consulta también: [Formatos de datos admitidos →](./Data_format.md)

##  Capturas de pantalla

> *GUI FLUPS*
<img width="1394" height="932" alt="Foto1" src="https://github.com/user-attachments/assets/ab6397c5-5751-4c59-858c-83ba9da74b67" />

> *GUI TAS*
<img width="1381" height="925" alt="image" src="https://github.com/user-attachments/assets/fb28d525-57a1-464f-994e-8829048f7ac9" />


> *GUI Global Fit*
<p align="center">
   <img src="https://github.com/user-attachments/assets/7effdce7-a700-4892-be37-54eac1b0866c" width="48%">
   <img src="https://github.com/user-attachments/assets/b103c26c-9a2b-42e3-977e-83fe45f9ab6e" width="48%">
 </p>

> *Decay Associated Spectra*
<img width="788" height="666" alt="image" src="https://github.com/user-attachments/assets/b84d6776-b94d-4424-9ddf-70cdac77e1dc" />

> *Kinetics Fit*
<img width="891" height="464" alt="image" src="https://github.com/user-attachments/assets/28caddd6-b46c-4981-b36c-5d3dd7228ea0" />


