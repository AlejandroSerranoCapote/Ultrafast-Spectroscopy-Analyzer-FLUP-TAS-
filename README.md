# Ultrafast Spectroscopy Analyzer 
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**Ultrafast Spectroscopy Analyzer** is a comprehensive, open-source software suite designed for the advanced processing and analysis of ultrafast spectroscopy data. It provides an intuitive graphical environment to transform raw experimental data into publication-quality results.

---

##  Supported Techniques

The application is optimized for two main experimental methods:
* **TAS** — *Transient Absorption Spectroscopy*
* **FLUPS** — *Fluorescence Up-Conversion Spectroscopy* 

## Mathematical Models

The software fits the experimental signal $\Delta A(t, \lambda)$ using two main approaches, both convolved with the Instrument Response Function (IRF).

---

### 1. Parallel Model: Decay-Associated Spectra (DAS)
Assumes that the components decay independently, which is ideal for mixtures of uncoupled species.

$$\Delta A(t, \lambda) = IRF(t) \otimes \sum_{i=1}^{n} A_i(\lambda) e^{-t/\tau_i}$$

Where each $A_i(\lambda)$ represents the **DAS** of the component with lifetime $\tau_i$.

---

### 2. Sequential Model: Species-Associated Spectra (SAS)
Describes an energy cascade or consecutive reaction: $1 \xrightarrow{k_1} 2 \xrightarrow{k_2} \dots \xrightarrow{k_n} n$.  
The populations of each species are governed by the **Bateman Equations**.

For a decay chain where $k_i = 1/\tau_i$, the concentration $C_n(t)$ of species $n$ is defined as:

$$C_n(t) = \left( \prod_{j=1}^{n-1} k_j \right) \sum_{j=1}^{n} \frac{e^{-k_j t}}{\prod_{p=1, p \neq j}^{n} (k_p - k_j)}$$

The total signal is the sum of the contributions of each excited state (SAS):

$$\Delta A(t, \lambda) = IRF(t) \otimes \sum_{i=1}^{n} SAS_i(\lambda) C_i(t)$$

---

### Instrument Response Function (IRF)
The time resolution is modeled using a Gaussian of width $w$ (FWHM) centered at $t_0$:

$$IRF(t) = \frac{1}{w \sqrt{\pi}} \exp\left( -\left( \frac{t - t_0}{w} \right)^2 \right)$$

---
> **Install the required dependencies** (run this command in the folder containing the script):
> ```bash
> pip install -r requirements.txt
> ```
>  **Run the application by typing the following in your terminal (inside the script folder):**
> ```bash
> python "Ultrafast Spectroscopy Analyzer.py"
> ```
> 
>  **Create a Standalone Executable (.exe) (Run by typing in your terminal inside the script folder)**:
> ```bash
> pyinstaller --onefile --noconsole --icon=icon.ico --exclude-module PyQt6 "Ultrafast Spectroscopy Analyzer.py"
> ```

---
##  Main Features

###  Multi-Technique Support
* **Dual Analysis:** Fully compatible with **TAS** (Transient Absorption Spectroscopy) and **FLUP** (Fluorescence Upconversion) data.
* **TCSPC Ready:** Support for Time-Correlated Single Photon Counting data processing.

###  Advanced Pre-processing
* **Chirp Correction:** Automated and manual $t_0$ adjustment per wavelength to correct Group Velocity Dispersion (GVD).
* **Data Cleaning:** Integrated tools for baseline subtraction, spectral/temporal binning, and dynamic data cropping.
* **Flexible Scaling:** Support for Linear and SymLog (Symmetric Logarithmic) time axes for better visualization of ultrafast dynamics.

###  Advanced Mathematical Analysis
* **SVD Diagnosis:** Built-in **Singular Value Decomposition** to determine the number of photo-active species (matrix rank) and spectral components.
* **Global Fitting:** Multiexponential analysis (up to 6 components) using two physical models:
    * **Parallel Model (DAS):** Extraction of Decay Associated Spectra.
    * **Sequential Model (SAS):** Species Associated Spectra modeling for successive population transfer.
* **Error Estimation:** Reliability analysis using covariance matrices and Jacobian-based confidence intervals.

###  Scientific Visualization
* **3D Surface Explorer:** Interactive 3D rendering of data surfaces to identify global trends.
* **Trace Checker:** Real-time inspection of individual wavelength kinetics with dual Linear/Log views.
* **Residual Mapping:** Automated generation of 2D error maps to evaluate fit quality across the entire dataset.

###  Export & Integration
* **Publication-Ready:** Export high-resolution plots (300 DPI) in PNG/PDF formats.
* **Open Data Formats:** Save results as `.txt` (compatible with Origin, Excel, or Python) and binary `.npy` files for fast reloading.


---

> See also: [Supported Data Formats →](./Data_format.md)

##  Screenshots

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


