# core_analysis.py
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

def read_csv_file(path):
    """Lee y limpia los datos del CSV.
    Devuelve: WL (1D numpy), TD (1D numpy), data (2D numpy shape (n_wl, n_td))
    """
    df = pd.read_csv(path)
    WL_raw = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    valid_rows = WL_raw.notna()
    WL = WL_raw[valid_rows].to_numpy()
    TD = []
    valid_cols = []
    for col in df.columns[1:]:
        try:
            TD.append(float(col))
            valid_cols.append(col)
        except Exception:
            continue
    TD = np.array(TD)
    data = df.loc[valid_rows, valid_cols].apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy()
    # Restar background simple (usa primeros 20 delays o todos si menos)
    nbac = 20 if data.shape[1] > 20 else data.shape[1]
    data = data - np.mean(data[:, :nbac], axis=1, keepdims=True)
    return WL, TD, data


def load_from_paths(data_path, wl_path, td_path):
    """
    Carga datos desde tres archivos separados: data, wl, td.
    - data: matriz 2D
    - wl: vector de longitudes de onda
    - td: vector de tiempos de retardo
    
    La función detecta automáticamente la orientación de la matriz y ajusta
    filas/columnas según wl y td. Si las dimensiones no coinciden, rellena
    con ceros o trunca según sea necesario.
    
    Devuelve:
        data_arr: np.array shape (n_wl, n_td)
        wl: np.array 1D
        td: np.array 1D
    """
    
    # ---- Cargar archivos ----
    # Intentar leer como CSV o txt
    try:
        df = pd.read_csv(data_path, header=None, sep=None, engine='python')
        data = df.values
    except Exception:
        data = np.loadtxt(data_path)
    
    wl = np.loadtxt(wl_path)
    td = np.loadtxt(td_path)
    
    # Reemplazar NaNs con 0
    data = np.nan_to_num(data, nan=0.0)
    wl = np.nan_to_num(wl, nan=0.0)
    td = np.nan_to_num(td, nan=0.0)
    
    # ---- Detectar orientación ----
    nwl, ntd = wl.size, td.size
    if data.shape == (nwl, ntd):
        data_arr = data.copy()
    elif data.shape == (ntd, nwl):
        data_arr = data.T.copy()
    else:
        # Intentar reshaping si coincide el número total de elementos
        if data.size == nwl * ntd:
            data_arr = data.reshape((nwl, ntd))
        else:
            # Rellenar/truncar según sea necesario
            data_arr = np.zeros((nwl, ntd))
            r = min(data.shape[0], nwl)
            c = min(data.shape[1], ntd)
            data_arr[:r, :c] = data[:r, :c]
            print(f"Warning: data shape {data.shape} no coincide con (n_wl, n_td); se rellenó/truncó a {(nwl, ntd)}.")
    
    # ---- Restar background simple ----
    nbac = 20 if data_arr.shape[1] > 20 else data_arr.shape[1]
    data_arr = data_arr - np.mean(data_arr[:, :nbac], axis=1, keepdims=True)
    
    return data_arr, wl, td

def load_data(auto_path=None, data_path=None, wl_path=None, td_path=None):
    """
    Carga datos desde un CSV único o desde tres archivos separados.
    - auto_path: path al CSV completo
    - data_path, wl_path, td_path: paths a los tres archivos
    Devuelve: data (2D), WL (1D), TD (1D)
    """
    import os

    if auto_path is not None and os.path.isfile(auto_path):
        try:
            WL, TD, data = read_csv_file(auto_path)
            return data, WL, TD
        except Exception:
            pass  # fall back a tres archivos

    if data_path and wl_path and td_path:
        data, WL, TD = load_from_paths(data_path, wl_path, td_path)
        return data, WL, TD

    raise ValueError("No se proporcionaron archivos válidos o no se pudieron leer.")

# ---------------------------------------------------------------------
# Modelos y funciones de corrección
# ---------------------------------------------------------------------
def t0_model(w, a, b, c, d):
    """
    Modelo no lineal propuesto:
    t0 = a * sqrt((b*w^2 - 1) / (c*w^2 - 1)) + d
    Devuelve nan donde la expresión no es válida.
    """
    w = np.asarray(w, dtype=float)
    num = b * w**2 - 1.0
    den = c * w**2 - 1.0
    out = np.full_like(w, np.nan, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = num / den
        valid = (den != 0) & (ratio >= 0)
        out[valid] = a * np.sqrt(ratio[valid]) + d
    return out

def apply_t0_correction_poly(popt, WL, TD, data):
    """Corrige datos usando un polinomio de grado 4.
    popt puede ser array-like con 5 coeficientes [c4,c3,c2,c1,c0] (como np.polyfit devuelve).
    Devuelve: corrected (same shape as data), t0_lambda (1D over WL)
    """
    # Acepta tanto coeficientes de np.polyfit (length 5) como exactamente (c4..c0)
    coeffs = np.asarray(popt)
    if coeffs.size != 5:
        raise ValueError("Polynomial coefficients must have length 5.")
    # np.polyval expects highest-first, so if user passed as [c4,c3,c2,c1,c0] that's ok
    t0_lambda = np.polyval(coeffs, WL)
    corrected = np.zeros_like(data)
    for i, wl in enumerate(WL):
        delay_corr = TD - t0_lambda[i]
        f = interp1d(delay_corr, data[i, :], kind='cubic', bounds_error=False, fill_value='extrapolate')
        corrected[i, :] = f(TD)
    return corrected, t0_lambda

def apply_t0_correction_nonlinear(popt, WL, TD, data):
    """Corrige datos usando los parámetros popt del modelo no lineal t0_model.
    Donde t0_model(WL) devuelve NaN, los datos se mantienen sin corregir.
    Devuelve: corrected, t0_lambda
    """
    t0_lambda = t0_model(WL, *popt)  # puede contener NaNs
    corrected = data.copy()
    for i, t0_val in enumerate(t0_lambda):
        if np.isfinite(t0_val):
            delay_corr = TD - t0_val
            f = interp1d(delay_corr, data[i, :], kind='cubic', bounds_error=False, fill_value='extrapolate')
            corrected[i, :] = f(TD)
        else:
            corrected[i, :] = data[i, :]
    return corrected, t0_lambda

# ---------------------------------------------------------------------
# Función de ajuste (intento non-linear, si falla -> polynomial)
# ---------------------------------------------------------------------
def fit_t0(w_points, t0_points, WL, TD, data, min_points_nonlinear=4):
    """
    Ajusta t0 a partir de puntos (w_points,t0_points) seleccionados por el usuario.
    Intentará ajustar el modelo no lineal (t0_model) si hay suficientes puntos; si falla,
    cae a un ajuste polinómico de grado 4 (requiere >=5 puntos). 
    Parámetros:
      - w_points: array-like de longitudes de onda (nm) de los puntos elegidos
      - t0_points: array-like de retardos (ps) correspondientes
      - WL, TD, data: arrays tal como devuelve read_csv_file
    Devuelve:
      dict con claves:
        - 'method': 'nonlinear' o 'poly4'
        - 'popt': parámetros del ajuste
        - 'fit_x', 'fit_y': arrays con curva a graficar (x ordenadas sobre rango de w_points)
        - 'corrected': matriz corregida (shape igual a data)
        - 't0_lambda': t0 por WL (1D)
    """
    w = np.asarray(w_points, dtype=float)
    t0 = np.asarray(t0_points, dtype=float)

    if w.size < 2:
        raise ValueError("Se necesitan al menos 2 puntos para ajustar (mejor >=4 para modelo non-linear).")

    fit_x = np.linspace(np.min(w), np.max(w), 400)
    # --- Intento 1: modelo no lineal ---
    try_nl = (w.size >= min_points_nonlinear)
    if try_nl:
        fit_ok = False
        try:
            wmin = np.min(w)
            # estimaciones iniciales robustas
            a0 = (np.nanmax(t0) - np.nanmin(t0)) / 2.0 if np.isfinite(t0).any() else 0.0
            d0 = np.nanmedian(t0)
            # garantizar que b,c > 1/(wmin^2) para que denominadores no sean absurdos
            min_required = 1.0 / (wmin**2) if wmin != 0 else 1e-8
            b0 = min_required * 1.1
            c0 = min_required * 1.2
            p0 = [a0, b0, c0, d0]
            bounds = ([-np.inf, min_required, min_required, -np.inf],
                      [np.inf, np.inf, np.inf, np.inf])

            popt, pcov = curve_fit(
                t0_model, w, t0,
                p0=p0, bounds=bounds,
                maxfev=20000, method="trf"
            )
            fit_y = t0_model(fit_x, *popt)
            # comprobar que la curva resultante es finita en el rango de interés
            if np.all(np.isfinite(fit_y)):
                corrected, t0_lambda = apply_t0_correction_nonlinear(popt, WL, TD, data)
                return {
                    'method': 'nonlinear',
                    'popt': popt,
                    'fit_x': fit_x,
                    'fit_y': fit_y,
                    'corrected': corrected,
                    't0_lambda': t0_lambda
                }
            else:
                # si la curva no es finita, hacemos fallback
                pass
        except Exception:
            # intento no lineal falló -> fallback
            pass

    # --- Fallback: polinomio grado 4 ---
    if w.size < 5:
        # no hay suficientes puntos para polinomio de grado 4 -> intentar polinomio con grado menor
        deg = min(3, max(1, w.size - 1))
    else:
        deg = 4

    try:
        coeffs = np.polyfit(w, t0, deg)
        # si deg<4, pad con ceros para que apply_t0_correction_poly espere 5 coef.
        if coeffs.size < 5:
            coeffs = np.concatenate([np.zeros(5 - coeffs.size), coeffs])
        fit_y = np.polyval(coeffs, fit_x)
        corrected, t0_lambda = apply_t0_correction_poly(coeffs, WL, TD, data)
        return {
            'method': 'poly4' if deg == 4 else f'poly{deg}',
            'popt': coeffs,
            'fit_x': fit_x,
            'fit_y': fit_y,
            'corrected': corrected,
            't0_lambda': t0_lambda
        }
    except Exception as e:
        raise RuntimeError(f"Both nonlinear and polynomial fits failed: {e}")

