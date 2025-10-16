# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 16:25:52 2025

@author: Alejandro
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec
from scipy.optimize import least_squares
from scipy.interpolate import RegularGridInterpolator, interp1d
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QSlider, QInputDialog,
    QDialog, QTabWidget, QProgressBar, QTableWidget, QTableWidgetItem,
    QHeaderView, QComboBox, QDoubleSpinBox, QFrame
)
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QLineEdit, QLabel, QHBoxLayout
import fit
from core_analysis import fit_t0, load_data
from PyQt5.QtWidgets import QLineEdit, QLabel, QHBoxLayout


class MainApp(QMainWindow):
    '''
    VENTANA PRINCIPAL (FLUPS/TAS)
    '''
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Analyzer Selector")
        self.setMinimumSize(500, 300)

        # --- Paleta de colores (coherente con FLUPS/TAS) ---
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("#1e1e1e"))
        palette.setColor(QPalette.WindowText, QColor("#f0f0f0"))
        self.setPalette(palette)

        # --- Widget central ---
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.setAlignment(Qt.AlignCenter)

        # --- Título principal ---
        title = QLabel("Select Analysis Mode")
        title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        title.setStyleSheet("color: #00bfff; margin-bottom: 20px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # --- Botones estilizados ---
        btn_flups = QPushButton("FLUPS Analyzer")
        btn_tas = QPushButton("TAS Analyzer  (Coming soon)")
        
        for btn in [btn_flups, btn_tas]:
            btn.setFont(QFont("Segoe UI", 12))
            btn.setFixedHeight(45)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #2d2d2d;
                    color: white;
                    border-radius: 10px;
                    border: 2px solid #00bfff;
                    padding: 8px 16px;
                }
                QPushButton:hover {
                    background-color: #00bfff;
                    color: black;
                }
                QPushButton:pressed {
                    background-color: #007acc;
                    color: white;
                }
            """)
            layout.addWidget(btn)
        
        # 🔒 Desactivar el TAS Analyzer (tachado y gris)
        font_tas = QFont("Segoe UI", 12)
        font_tas.setStrikeOut(True)
        btn_tas.setFont(font_tas)
        btn_tas.setEnabled(False)
        btn_tas.setStyleSheet("""
            QPushButton {
                background-color: #3a3a3a;
                color: gray;
                border-radius: 10px;
                border: 2px dashed #555;
                padding: 8px 16px;
            }
        """)

        # --- Conexiones ---
        btn_flups.clicked.connect(self.launch_flups)
        # btn_tas.clicked.connect(self.launch_tas)

        self.setCentralWidget(central_widget)
        # --- Descripción informativa ---
        description = QLabel(
            "Welcome! This free and open-source software allows you to analyze "
            "ultrafast spectroscopy data from experiments such as "
            "<b>FLUPS</b> (Fluorescence Upconversion Spectroscopy) "
            "and <b>TAS</b> (Transient Absorption Spectroscopy).<br><br>"
            "For any questions or feedback, please contact: "
            "<span style='color:#00bfff; font-weight:bold;'>alejandro.serrano1610@gmail.com</span>"
        )
        description.setWordWrap(True)
        description.setAlignment(Qt.AlignCenter)
        description.setStyleSheet("""
            color: #c0c0c0;
            font-size: 11pt;
            margin-top: 25px;
        """)
        layout.addWidget(description)
    # --- Lanzar FLUPS ---
    def launch_flups(self):
        self.analyzer = FLUPSAnalyzer()
        self.analyzer.show()
        self.close()

    # --- Lanzar TAS ---
    def launch_tas(self):
        self.analyzer = TASAnalyzer()
        self.analyzer.show()
        self.close()


class FLUPSAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FLUPS Analyzer — PyQt5 Edition")
        self.setMinimumSize(1400, 900)
    
        # estado
        self.WL = None
        self.TD = None
        self.data = None
        self.file_path = None
        self.data_corrected = None
        self.result_fit = None
    
        # widgets
        self.btn_load = QPushButton("Load CSV")
        self.btn_load.clicked.connect(self.load_file)

        self.btn_plot = QPushButton("Show Map")
        self.btn_plot.clicked.connect(self.plot_map)
        self.btn_plot.setEnabled(False)
        self.btn_remove_fringe = QPushButton("Remove Pump Fringe")
        self.btn_remove_fringe.clicked.connect(self.remove_pump_fringe)
        self.btn_remove_fringe.setEnabled(True)

        self.label_status = QLabel("No file loaded")
    
        self.btn_select = QPushButton("Select t₀ points")
        self.btn_select.clicked.connect(self.enable_point_selection)
        self.btn_select.setEnabled(False)
    
        self.btn_fit = QPushButton("Fit t₀")
        self.btn_fit.clicked.connect(self.fit_t0_points)
        self.btn_fit.setEnabled(False)
    
        self.btn_show_corr = QPushButton("Show Corrected Map")
        self.btn_show_corr.clicked.connect(self.toggle_corrected_map)
        self.btn_show_corr.setEnabled(False)
        self.showing_corrected = False
        self.btn_global_fit = QPushButton("Global Fit")
        self.btn_global_fit.clicked.connect(self.open_global_fit)

        # sliders
        self.slider_min = QSlider(Qt.Horizontal)
        self.slider_max = QSlider(Qt.Horizontal)
        self.slider_min.valueChanged.connect(self.update_wl_range)
        self.slider_max.valueChanged.connect(self.update_wl_range)
    
        # matplotlib canvas con gridspec (mapa arriba, cinética y espectro abajo)
        self.figure = Figure(figsize=(12, 8))
        self.gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[1, 1], hspace=0.25, wspace=0.35)
        
        # mapa principal ocupa las dos columnas de la primera fila
        self.ax_map = self.figure.add_subplot(self.gs[0, :])
        
        # segunda fila: dos subplots lado a lado
        self.ax_time_small = self.figure.add_subplot(self.gs[1, 0])
        self.ax_spec_small = self.figure.add_subplot(self.gs[1, 1])
        self.canvas = FigureCanvas(self.figure)
    
        # --- Inicializar variables relacionadas con eventos / interacción ---
        self.clicked_points = []   # almacenar puntos y artistas
        self.cid_click = None      # id de conexión para clicks
        self.cid_move = None  # <<< inicializar aquí
    
        # conectar evento de movimiento (una sola vez)
        self.cid_move = self.canvas.mpl_connect("motion_notify_event", self.on_move_map)
            
        # elementos interactivos
        self.pcm = None
        self.cbar = None
        self.marker_map = None
        self.vline_map = None
        self.hline_map = None
        self.fit_line_artist = None
    
        # inicializar small plots
        self._init_small_plots()
    
        # layout
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.btn_load)
        top_layout.addWidget(self.btn_plot)
        top_layout.addWidget(self.btn_select)
        top_layout.addWidget(self.btn_fit)
        top_layout.addWidget(self.label_status)
        top_layout.addWidget(self.btn_show_corr)
        top_layout.addWidget(self.btn_global_fit)
        top_layout.addWidget(self.btn_remove_fringe)
        
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("λ min"))
        slider_layout.addWidget(self.slider_min)
        slider_layout.addWidget(QLabel("λ max"))
        slider_layout.addWidget(self.slider_max)
    
        layout = QVBoxLayout()
        layout.addLayout(top_layout)
        layout.addWidget(self.canvas)
        layout.addLayout(slider_layout)
    
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
    
        # estilo
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121212;
            }
            QWidget {
                background-color: #121212;
                color: #E0E0E0;
            }
            QPushButton {
                background-color: #1E88E5;
                color: white;
                font-weight: bold;
                padding: 6px;
                border-radius: 8px;
                
            }
            QPushButton:hover {
                background-color: #42A5F5;
            }
            QPushButton:disabled {
                background-color: #555;
            }
            QLabel {
                font-size: 14px;
                color: #E0E0E0;
            }
        """)


        # --- Controles para límites del eje X ---
        range_layout = QHBoxLayout()
        
        # Reducir espacio entre widgets
        range_layout.setSpacing(5)
        
        # Centrar horizontalmente
        range_layout.setAlignment(Qt.AlignCenter)
        
        range_layout.addWidget(QLabel("Delay min (ps):"))
        
        self.xmin_edit = QLineEdit("-1")
        self.xmin_edit.setFixedWidth(50)
        range_layout.addWidget(self.xmin_edit)
        
        range_layout.addWidget(QLabel("Delay max (ps):"))
        
        self.xmax_edit = QLineEdit("3")
        self.xmax_edit.setFixedWidth(50)
        range_layout.addWidget(self.xmax_edit)
        
        self.btn_apply_xlim = QPushButton("Apply X limits")
        self.btn_apply_xlim.setFixedWidth(120)  # más compacto
        range_layout.addWidget(self.btn_apply_xlim)
        
        self.btn_apply_xlim.clicked.connect(self.apply_x_limits)
        
        # 🔹 Agregarlo dentro de otro layout contenedor para centrarlo mejor
        range_container = QWidget()
        range_container.setLayout(range_layout)
        range_container.setMaximumWidth(600)  # controla el ancho total del bloque
        
        layout.addWidget(range_container, alignment=Qt.AlignLeft)


        
        # --- fit de colores de los ejes principales y colorbars ---
        # Ejes del mapa principal (fondo blanco)
        self.ax_map.tick_params(colors="black")
        self.ax_map.xaxis.label.set_color("black")
        self.ax_map.yaxis.label.set_color("black")
        self.ax_map.title.set_color("black")
        for spine in self.ax_map.spines.values():
            spine.set_color("black")
        
        # Si la colorbar ya existe, ajusta su estilo también
        if self.cbar is not None:
            self.cbar.ax.yaxis.set_tick_params(color="black", labelcolor="black")
            self.cbar.ax.yaxis.label.set_color("black")
            for spine in self.cbar.ax.spines.values():
                spine.set_color("black")
        # Asegura que textos y ticks sean visibles sobre blanco
        for ax in [self.ax_time_small, self.ax_spec_small]:
            ax.tick_params(colors="black")
            ax.xaxis.label.set_color("black")
            ax.yaxis.label.set_color("black")
            ax.title.set_color("black")


    def open_global_fit(self):
        dlg = GlobalFitPanel(self)
        dlg.exec_()
    def _init_small_plots(self):
        # cinética pequeña
        self.ax_time_small.set_xlabel("Delay (ps)")
        self.ax_time_small.set_ylabel("ΔA")
        self.ax_time_small.set_title("Kinetics (cursor)")
        self.ax_time_small.set_xlim(-1, 3)
        self.cut_time_small, = self.ax_time_small.plot([], [], '-', lw=1.5)
    
        # línea vertical de tiempo (inicialmente en None)
        self.vline_time_small = self.ax_time_small.axvline(
            x=0, color='k', ls='--', lw=1, visible=False, zorder=5
        )
    
        # espectro pequeño
        self.ax_spec_small.set_xlabel("Wavelength (nm)")
        self.ax_spec_small.set_ylabel("ΔA")
        self.ax_spec_small.set_title("Spectra (cursor)")
        self.cut_spec_small, = self.ax_spec_small.plot([], [], '-', lw=1.5)
        
    def apply_x_limits(self):
        """Aplica los límites del eje X (Delay) escritos por el usuario."""
        try:
            x_min = float(self.xmin_edit.text())
            x_max = float(self.xmax_edit.text())
            if x_min >= x_max:
                raise ValueError("x_min debe ser menor que x_max")
            
            # Aplica los límites al subplot de cinética
            self.ax_time_small.set_xlim(x_min, x_max)
            self.canvas.draw_idle()
    
        except ValueError:
            QMessageBox.warning(self, "Error", "Introduce valores numéricos válidos para los límites de Delay.")

    def remove_pump_fringe(self):
        """Quita la franja de bombeo directamente sobre los datos actuales."""
        if self.data is None:
            QMessageBox.warning(self, "No data", "Load data first.")
            return
    
        # pedir al usuario la longitud de onda del pump y el ancho de la franja
        sWl, ok1 = QInputDialog.getDouble(
            self, "Pump wavelength", "Pump wavelength (nm):", min=0.0
        )
        if not ok1:
            return
        wisWL, ok2 = QInputDialog.getDouble(
            self, "Width of scattering", "Width of pump scattering (nm):", min=0.0
        )
        if not ok2:
            return
    
        # decidir sobre qué conjunto de datos aplicar
        if getattr(self, "showing_corrected", False) and self.data_corrected is not None:
            data_target = self.data_corrected
        else:
            data_target = self.data
    
        # índices de la franja
        posl1 = np.argmin(np.abs(self.WL - (sWl - wisWL / 2)))
        posl2 = np.argmin(np.abs(self.WL - (sWl + wisWL / 2)))
    
        # modificar los datos directamente
        data_target[posl1:posl2, :] = 1e-10
    
        # refrescar el mapa para ver el efecto
        if getattr(self, "showing_corrected", False):
            self.toggle_corrected_map()  # volver a mostrar mapa corregido
        else:
            self.plot_map()  # volver a mostrar mapa original
    
        QMessageBox.information(
            self, "Pump fringe removed",
            f"Fringe at {sWl} ± {wisWL/2} nm has been set to near-zero."
        )

    def load_file(self):
        """Carga archivo de datos y normaliza ΔA automáticamente."""
        # Seleccionar CSV o data.txt
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV or Data File", "", 
            "CSV Files (*.csv);;Data Files (*.txt *.dat)"
        )
        if not file_path:
            return
    
        try:
            # Cargar datos
            if file_path.endswith(".csv"):
                data, wl, td = load_data(auto_path=file_path)
            else:
                # Solicitar WL y TD por separado
                wl_path, _ = QFileDialog.getOpenFileName(self, "Select Wavelength File", "", "Text Files (*.txt)")
                td_path, _ = QFileDialog.getOpenFileName(self, "Select Delay File", "", "Text Files (*.txt)")
                if not wl_path or not td_path:
                    QMessageBox.warning(self, "Files missing", "You must select both WL and TD files.")
                    return
                data, wl, td = load_data(data_path=file_path, wl_path=wl_path, td_path=td_path)
    
            # Ordenar WL ascendente
            if not np.all(np.diff(wl) > 0):
                order = np.argsort(wl)
                wl = wl[order]
                data = data[order, :]
    
            # --- Normalización ---
            # Opción 1: Normalización global [-1, 1]
            max_val = np.nanmax(np.abs(data))
            if max_val != 0:
                data = data / max_val
    
            # Opción 2: Normalización por fila (cada WL)
            # data = data / np.nanmax(np.abs(data), axis=1)[:, np.newaxis]
    
            self.WL, self.TD, self.data = wl, td, data
            self.file_path = file_path

            # 🔧 Guarda también la ruta y el directorio base del CSV
            self.csv_path = file_path
            self.base_dir = os.path.dirname(file_path)
            
            self.label_status.setText(f"Loaded & Normalized: {os.path.basename(file_path)}")
            self.btn_plot.setEnabled(True)
            self.btn_select.setEnabled(True)
            self.btn_fit.setEnabled(True)
    
            # Actualizar sliders
            nwl = len(wl)
            self.slider_min.setMinimum(0)
            self.slider_min.setMaximum(nwl - 1)
            self.slider_max.setMinimum(0)
            self.slider_max.setMaximum(nwl - 1)
            self.slider_min.setValue(0)
            self.slider_max.setValue(nwl - 1)
    
        except Exception as e:
            QMessageBox.critical(self, "Error loading file", str(e))


    def plot_map(self):
        """Dibuja el mapa principal con subplots integrados y activa hover."""
        if self.data is None:
            return
    
        self.ax_map.clear()
        if self.cbar:
            try:
                self.cbar.remove()
            except Exception:
                pass
            self.cbar = None
    
        # pcolormesh usando todos los datos, evita problemas de dimensiones
        self.pcm = self.ax_map.pcolormesh(self.WL, self.TD, self.data.T, shading="auto", cmap="jet")
        self.ax_map.set_xlabel("Wavelength (nm)")
        self.ax_map.set_ylabel("Delay (ps)")
        self.ax_map.set_yscale("symlog")
    
        divider = make_axes_locatable(self.ax_map)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        self.cbar = self.figure.colorbar(self.pcm, cax=cax, label="ΔA")
    
        # crucetas y marcador iniciales
        self.vline_map = self.ax_map.axvline(self.WL[0], color='k', ls='--', lw=1, zorder=6)
        self.hline_map = self.ax_map.axhline(self.TD[0], color='k', ls='--', lw=1, zorder=6)
        self.marker_map, = self.ax_map.plot([self.WL[0]], [self.TD[0]], 'wx', markersize=8, markeredgewidth=2, zorder=7)
    
        # limpiar small plots
        self.cut_time_small.set_data([], [])
        self.cut_spec_small.set_data([], [])
        self.ax_time_small.relim(); self.ax_time_small.autoscale_view()
        self.ax_spec_small.relim(); self.ax_spec_small.autoscale_view()
    
        # conectar eventos
        if self.cid_click is None:
            self.cid_click = self.canvas.mpl_connect("button_press_event", self.on_click_map)

    
        self.canvas.draw_idle()
    
    
    def update_wl_range(self):
        """Actualizar el mapa y subplots pequeños según el rango de λ de los sliders."""
        if self.data is None:
            return
    
        wl_min_idx = self.slider_min.value()
        wl_max_idx = self.slider_max.value()
        if wl_min_idx >= wl_max_idx:
            wl_max_idx = wl_min_idx + 1
        wl_max_idx = min(wl_max_idx + 1, len(self.WL))
    
        WL_sel = self.WL[wl_min_idx:wl_max_idx]
        data_sel = self.data[wl_min_idx:wl_max_idx, :]  # (N_wl, N_td)
    
        # --- limpiar eje y colorbar ---
        self.ax_map.clear()
        if self.cbar:
            try:
                self.cbar.remove()
            except Exception:
                pass
            self.cbar = None
    
        # reset referencias de cruces
        self.vline_map = None
        self.hline_map = None
        self.marker_map = None
    
        # dibujar mapa filtrado
        self.pcm = self.ax_map.pcolormesh(WL_sel, self.TD, data_sel.T, shading="auto", cmap="jet")
        self.ax_map.set_xlabel("Wavelength (nm)")
        self.ax_map.set_ylabel("Delay (ps)")
        self.ax_map.set_title("ΔA Map (FLUPS)")
        self.ax_map.set_yscale("symlog")
    
        divider = make_axes_locatable(self.ax_map)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        self.cbar = self.figure.colorbar(self.pcm, cax=cax, label="ΔA")
    
        # redibujar puntos clicados
        for p in getattr(self, "clicked_points", []):
            try:
                self.ax_map.plot(p['x'], p['y'], 'wo', markeredgecolor='k', markersize=6, zorder=6)
            except Exception:
                pass
    
        # crear nuevas cruces iniciales (posición central del rango)
        if WL_sel.size > 0:
            x0 = np.median(WL_sel)
        else:
            x0 = self.WL[0]
        y0 = np.median(self.TD)
    
        self.vline_map = self.ax_map.axvline(x0, color='k', ls='--', lw=1, zorder=6)
        self.hline_map = self.ax_map.axhline(y0, color='k', ls='--', lw=1, zorder=6)
        self.marker_map, = self.ax_map.plot([x0], [y0], 'wx', markersize=8, markeredgewidth=2, zorder=7)
    
        # --- ACTUALIZAR SUBPLOTS PEQUEÑOS ---
        self.update_small_cuts(x0, y0, WL_sel=WL_sel, data_sel=data_sel)
    
        self.canvas.draw_idle()

    # interacción: selección de puntos
    def enable_point_selection(self):
        self.clicked_points = []
        if self.cid_click is None:
            self.cid_click = self.canvas.mpl_connect("button_press_event", self.on_click_map)
        QMessageBox.information(self, "Mode: Select points",
                                "Click izquierdo: añadir punto\nClick derecho: borrar último punto.\nLuego pulsa 'Fit t₀'.")

    def update_small_cuts(self, x, y, WL_sel=None, data_sel=None):
        """Actualiza los subplots pequeños (cinética + espectro) para la posición (x,y)
           usando los datos filtrados por sliders si se pasan."""
        if x is None or y is None:
            return
    
        # usar datos filtrados si se pasan, sino usar todos
        if WL_sel is None or data_sel is None:
            WL_vis = self.WL
            data_vis = self.data
        else:
            WL_vis = WL_sel
            data_vis = data_sel
    
        if WL_vis.size == 0:
            return
    
        # índices más cercanos
        idx_wl = int(np.argmin(np.abs(WL_vis - x)))
        idx_td = int(np.argmin(np.abs(self.TD - y)))
    
        # cinética (fila idx_wl)
        y_time = data_vis[idx_wl, :].ravel()
        self.cut_time_small.set_data(self.TD, y_time)
        self.ax_time_small.relim(); self.ax_time_small.autoscale_view()
        self.ax_time_small.set_title(f"Kinetics at {WL_vis[idx_wl]:.1f} nm")
    
        # espectro (columna idx_td)
        y_spec = data_vis[:, idx_td].ravel()
        self.cut_spec_small.set_data(WL_vis, y_spec)
        self.ax_spec_small.relim(); self.ax_spec_small.autoscale_view()
        self.ax_spec_small.set_title(f"Spectra at {self.TD[idx_td]:.2f} ps")
    def on_click_map(self, event):
        """Registrar puntos sobre el mapa (izq añade, derecha borra último) y actualizar cortes."""
        if event.inaxes != self.ax_map:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        if event.button == 1:  # left click -> añadir punto
            artist, = self.ax_map.plot(x, y, 'wo', markeredgecolor='k', markersize=6, zorder=6)
            self.clicked_points.append({'x': x, 'y': y, 'artist': artist})
        elif event.button == 3 and self.clicked_points:  # right click -> borrar último
            last = self.clicked_points.pop()
            try:
                last['artist'].remove()
            except Exception:
                pass

        # actualizar marker que sigue al cursor (opcional: mover el marker principal)
        if self.marker_map is None:
            self.marker_map, = self.ax_map.plot([x], [y], 'wx', markersize=8, markeredgewidth=2)
        else:
            self.marker_map.set_data([x], [y])


        # actualizar referencias visuales de la línea vertical
        if self.vline_map is None:
            # si no existe, crearla
            self.vline_map = self.ax_map.axvline(x, color='k', ls='--', lw=1)
        else:
            # si ya existe, solo actualizar su posición y asegurar que sea visible
            self.vline_map.set_xdata([x, x])
            self.vline_map.set_visible(True)
        
        # actualizar referencias visuales de la línea horizontal
        if self.hline_map is None:
            self.hline_map = self.ax_map.axhline(y, color='k', ls='--', lw=1)
        else:
            self.hline_map.set_ydata([y, y])
            self.hline_map.set_visible(True)

        # --- aquí está la diferencia: actualizar los subplots pequeños ---
        self.update_small_cuts(x, y)

        self.canvas.draw_idle()

    def on_move_map(self, event):
        """Actualizar líneas y subplots pequeños al mover el cursor sobre el mapa."""
        if event.inaxes != self.ax_map or self.data is None:
            return
    
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
    
        # Crear o actualizar líneas cruzadas
        if self.vline_map is None:
            self.vline_map = self.ax_map.axvline(x, color='k', ls='--', lw=1, zorder=6)
        else:
            self.vline_map.set_xdata([x, x])
    
        if self.hline_map is None:
            self.hline_map = self.ax_map.axhline(y, color='k', ls='--', lw=1, zorder=6)
        else:
            self.hline_map.set_ydata([y, y])
    
        # Crear o actualizar el marcador
        if self.marker_map is None:
            self.marker_map, = self.ax_map.plot([x], [y], 'wx', markersize=8,
                                                markeredgewidth=2, zorder=7)
        else:
            self.marker_map.set_data([x], [y])
    
        # Actualizar los subplots pequeños (cinética y espectro)
        self.update_small_cuts(x, y)
        
        # --- actualizar línea vertical en la cinética ---
        try:
            if self.vline_time_small is None:
                self.vline_time_small = self.ax_time_small.axvline(
                    x=y, color='k', ls='--', lw=1, zorder=5)
            else:
                self.vline_time_small.set_xdata([y, y])
                self.vline_time_small.set_visible(True)
        except Exception:
            pass
    
        # Refrescar el dibujo (sin sobrecargar CPU)
        self.canvas.draw_idle()


    def fit_t0_points(self):
        if not getattr(self, "clicked_points", None) or len(self.clicked_points) < 2:
            QMessageBox.warning(self, "Not enough points", "Select at least 2 points on the map.")
            return

        w_points = np.array([p['x'] for p in self.clicked_points])
        t0_points = np.array([p['y'] for p in self.clicked_points])

        try:
            result = fit_t0(w_points, t0_points, self.WL, self.TD, self.data)
        except Exception as e:
            QMessageBox.critical(self, "Fit error", str(e))
            return

        self.result_fit = result
        self.data_corrected = result['corrected']

        # dibujar curva del fit sobre mapa principal
        if self.fit_line_artist is not None:
            try:
                self.fit_line_artist.remove()
            except Exception:
                pass
        self.fit_line_artist, = self.ax_map.plot(result['fit_x'], result['fit_y'], 'r-', lw=2, label="t₀ fit")
        self.ax_map.legend()
        self.canvas.draw_idle()

        # guardado automático (idéntico a tu comportamiento actual)
        self.btn_show_corr.setEnabled(True)
        import os
        base_dir = os.path.dirname(self.file_path)
        base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        save_dir = os.path.join(base_dir, f"{base_name}_Results")
        os.makedirs(save_dir, exist_ok=True)

        data_corr = result['corrected']
        WL = self.WL
        TD = self.TD

        np.save(os.path.join(save_dir, f"{base_name}_treated_data.npy"),
                {'data_c': data_corr, 'WL': WL, 'TD': TD})

        np.savetxt(os.path.join(save_dir, f"{base_name}_WL.txt"), WL,
                   fmt='%.6f', header='Wavelength (nm)', comments='')
        np.savetxt(os.path.join(save_dir, f"{base_name}_TD.txt"), TD,
                   fmt='%.6f', header='Delay (ps)', comments='')

        with open(os.path.join(save_dir, f"{base_name}_kin.txt"), 'w') as f:
            f.write("\t".join([f"{base_name}_kin_{round(wl,1)}nm" for wl in WL]) + "\n")
            np.savetxt(f, data_corr.T, fmt='%.6e', delimiter='\t')

        with open(os.path.join(save_dir, f"{base_name}_spec.txt"), 'w') as f:
            f.write("\t".join([f"{base_name}_spec_{td:.2f}ps" for td in TD]) + "\n")
            np.savetxt(f, data_corr, fmt='%.6e', delimiter='\t')

        t0_lambda = result['t0_lambda']
        popt = result['popt']
        method = result['method']

        t0_file = os.path.join(save_dir, f"{base_name}_t0_fit.txt")
        np.savetxt(t0_file, np.column_stack((WL, t0_lambda)),
                   fmt='%.6f', header='Wavelength (nm)\t t0 (ps)', comments='')

        params_file = os.path.join(save_dir, f"{base_name}_fit_params.txt")
        with open(params_file, 'w') as f:
            f.write(f"Fit method: {method}\n")
            f.write("Fit parameters:\n")
            if method.startswith('poly'):
                names = ['c4', 'c3', 'c2', 'c1', 'c0']
            else:
                names = ['a', 'b', 'c', 'd']
            for name, val in zip(names, popt):
                f.write(f"  {name} = {val:.6g}\n")

        QMessageBox.information(self, "Files saved",
                                f"Results saved in:\n{save_dir}")
        QMessageBox.information(self, "t₀ Fit Result",
                                f"Fit completed using {method} model.\nParameters: {np.round(popt,4)}")


    def toggle_corrected_map(self):
        """Alterna entre mapa original y corregido dentro de la misma ventana,
           mostrando el mapa limpio y manteniendo crucetas."""
        
        if self.data_corrected is None:
            QMessageBox.warning(self, "No corrected data", "Run 'Fit t₀' first to generate corrected data.")
            return
    
        # decidir qué mapa mostrar
        if getattr(self, "showing_corrected", False):
            # mostrar original
            data_to_plot = self.data
            self.showing_corrected = False
            self.btn_show_corr.setText("Show Corrected Map")
        else:
            # mostrar corregido
            data_to_plot = np.copy(self.data_corrected)
            self.showing_corrected = True
            self.btn_show_corr.setText("Show Original Map")
    
        # limpiar eje y colorbar
        self.ax_map.clear()
        if self.cbar:
            try: self.cbar.remove()
            except: pass
            self.cbar = None
    
        # BORRAR puntos seleccionados y línea de fit
        for p in getattr(self, "clicked_points", []):
            try:
                p['artist'].remove()
            except Exception:
                pass
        self.clicked_points = []
    
        if self.fit_line_artist is not None:
            try: self.fit_line_artist.remove()
            except Exception:
                pass
        self.fit_line_artist = None
        # decidir rango de color según modo
        if getattr(self, "is_TAS_mode", False):
            # usar rango real de los datos
            vmin = np.nanmin(data_to_plot)
            vmax = np.nanmax(data_to_plot)
            # margen opcional
            margin = 0.05 * (vmax - vmin)
            vmin -= margin
            vmax += margin
        else:
            # FLUPS siempre [-1, 1]
            vmin, vmax = -1, 1
        
        self.pcm = self.ax_map.pcolormesh(
            self.WL, self.TD, data_to_plot.T,
            shading="auto",
            cmap="jet",
            vmin=vmin,
            vmax=vmax
        )
        self.ax_map.set_xlabel("Wavelength (nm)")
        self.ax_map.set_ylabel("Delay (ps)")
        self.ax_map.set_title("ΔA Map (FLUPS)")
        self.ax_map.set_yscale("symlog")
    
        # colorbar
        divider = make_axes_locatable(self.ax_map)
        cax = divider.append_axes("right", size="3%", pad=0.02)
        self.cbar = self.figure.colorbar(self.pcm, cax=cax, label="ΔA")
        self.cbar.ax.yaxis.set_tick_params(color="black", labelcolor="black")
        self.cbar.ax.yaxis.label.set_color("black")
        for spine in self.cbar.ax.spines.values():
            spine.set_color("black")
    
        # restaurar estilo de ejes
        self.ax_map.set_facecolor("white")
        self.ax_map.tick_params(colors="black")
        self.ax_map.xaxis.label.set_color("black")
        self.ax_map.yaxis.label.set_color("black")
        self.ax_map.title.set_color("black")
        for spine in self.ax_map.spines.values():
            spine.set_color("black")
    
        # posición actual de crucetas
        if self.vline_map is not None:
            x0 = self.vline_map.get_xdata()[0]
        else:
            x0 = self.WL[0]
    
        if self.hline_map is not None:
            y0 = self.hline_map.get_ydata()[0]
        else:
            y0 = self.TD[0]
    
        # crucetas y marcador
        self.vline_map = self.ax_map.axvline(x0, color='k', ls='--', lw=1)
        self.hline_map = self.ax_map.axhline(y0, color='k', ls='--', lw=1)
        self.marker_map, = self.ax_map.plot([x0], [y0], 'wx', markersize=8, markeredgewidth=2)
    
        # limpiar subplots pequeños
        self.cut_time_small.set_data([], [])
        self.cut_spec_small.set_data([], [])
        self.ax_time_small.relim(); self.ax_time_small.autoscale_view()
        self.ax_spec_small.relim(); self.ax_spec_small.autoscale_view()
    
        # conectar eventos para crucetas y subplots
        def update_small_cuts(x, y):
            if x is None or y is None:
                return
            idx_wl = int(np.argmin(np.abs(self.WL - x)))
            idx_td = int(np.argmin(np.abs(self.TD - y)))
            y_time = data_to_plot[idx_wl, :].ravel()
            self.cut_time_small.set_data(self.TD, y_time)
            self.ax_time_small.relim(); self.ax_time_small.autoscale_view()
            self.ax_time_small.set_title(f"Kinetics at {self.WL[idx_wl]:.1f} nm")
            y_spec = data_to_plot[:, idx_td].ravel()
            self.cut_spec_small.set_data(self.WL, y_spec)
            self.ax_spec_small.relim(); self.ax_spec_small.autoscale_view()
            self.ax_spec_small.set_title(f"Spectra at {self.TD[idx_td]:.2f} ps")
    
        def onclick(event):
            if event.inaxes != self.ax_map: return
            x, y = event.xdata, event.ydata
            if x is None or y is None: return
            self.vline_map.set_xdata([x, x])
            self.hline_map.set_ydata([y, y])
            self.marker_map.set_data([x], [y])
            update_small_cuts(x, y)
            self.canvas.draw_idle()
    
        def onmove(event):
            if event.inaxes != self.ax_map: return
            x, y = event.xdata, event.ydata
            if x is None or y is None: return
            self.vline_map.set_xdata([x, x])
            self.hline_map.set_ydata([y, y])
            self.marker_map.set_data([x], [y])
            update_small_cuts(x, y)
            self.canvas.draw_idle()
    
        # desconectar eventos previos de mapa corregido
        if hasattr(self, "cid_corr_click") and self.cid_corr_click is not None:
            self.canvas.mpl_disconnect(self.cid_corr_click)
        if hasattr(self, "cid_corr_move") and self.cid_corr_move is not None:
            self.canvas.mpl_disconnect(self.cid_corr_move)
    
        # conectar nuevos eventos
        self.cid_corr_click = self.canvas.mpl_connect("button_press_event", onclick)
        self.cid_corr_move = self.canvas.mpl_connect("motion_notify_event", onmove)
    
        self.canvas.draw_idle()
    


class TASAnalyzer(FLUPSAnalyzer):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TAS Analyzer — PyQt5 Edition")
        self.label_status.setText("TAS mode active")
        
        # --- Datos ---
        self.medida = None
        self.solvente = None
        self.pump_mask = None  # nueva variable para eliminar pump
        self.TDSol = None
        self.WLSol = None
        self.is_TAS_mode = True


       # --- Sliders extra para AM (amplitud) y SF (shift temporal) ---
        self.slider_am = QSlider(Qt.Horizontal)
        self.slider_am.setMinimum(0)
        self.slider_am.setMaximum(200)
        self.slider_am.setValue(100)  # 100% por defecto
        self.slider_am.valueChanged.connect(self.update_am_sf)
        
        # Slider para shift temporal
        self.slider_sf = QSlider(Qt.Horizontal)
        # Aumentamos resolución ×100 → precisión de 0.01 ps
        self.slider_sf.setMinimum(-20000)
        self.slider_sf.setMaximum(20000)
        self.slider_sf.setValue(0)
        
        # Spinbox para shift temporal (permite decimales)
        self.spin_sf = QDoubleSpinBox()
        self.spin_sf.setDecimals(3)
        self.spin_sf.setRange(-200.0, 200.0)  # en ps
        self.spin_sf.setSingleStep(0.01)
        self.spin_sf.setValue(0.0)
        
        # Sincronizar slider y spinbox
        def sync_slider_to_spin(value):
            # slider tiene resolución ×100 → divide entre 100
            self.spin_sf.blockSignals(True)
            self.spin_sf.setValue(value / 100.0)
            self.spin_sf.blockSignals(False)
            self.update_am_sf()
        
        def sync_spin_to_slider(value):
            self.slider_sf.blockSignals(True)
            self.slider_sf.setValue(int(value * 100))
            self.slider_sf.blockSignals(False)
            self.update_am_sf()
        
        self.slider_sf.valueChanged.connect(sync_slider_to_spin)
        self.spin_sf.valueChanged.connect(sync_spin_to_slider)
        
        # Layout
        slider_layout_extra = QHBoxLayout()
        slider_layout_extra.addWidget(QLabel("Amplitude (%)"))
        slider_layout_extra.addWidget(self.slider_am)
        slider_layout_extra.addWidget(QLabel("Shift (ps)"))
        slider_layout_extra.addWidget(self.slider_sf)
        slider_layout_extra.addWidget(self.spin_sf)
        
        # Añadir sliders adicionales al layout principal
        self.centralWidget().layout().addLayout(slider_layout_extra)

    def remove_pump_fringe(self):
        if self.data is None:
            QMessageBox.warning(self, "No data", "Load TAS data first.")
            return
    
        sWl, ok1 = QInputDialog.getDouble(self, "Pump wavelength", "Pump wavelength (nm):", min=0.0)
        if not ok1: return
        wisWL, ok2 = QInputDialog.getDouble(self, "Width of scattering", "Width of pump scattering (nm):", min=0.0)
        if not ok2: return
    
        posl1 = np.argmin(np.abs(self.WL - (sWl - wisWL / 2)))
        posl2 = np.argmin(np.abs(self.WL - (sWl + wisWL / 2)))
    
        # crear o actualizar máscara
        if self.pump_mask is None:
            self.pump_mask = np.zeros_like(self.medida, dtype=bool)
        self.pump_mask[posl1:posl2, :] = True
    
        # aplicar máscara sobre self.data
        self.update_am_sf()
    
        QMessageBox.information(self, "Pump fringe removed",
                                f"Fringe at {sWl} ± {wisWL/2} nm will be zeroed.")

    # ------------------------------------------------------------------
    # CARGA DE ARCHIVOS
    # ------------------------------------------------------------------
    def load_file(self):
        """Carga los ficheros de medida y solvente TAS (CSV o TXT)."""
        import os
    
        # --- Seleccionar archivo de medida ---
        file_path_medida, _ = QFileDialog.getOpenFileName(
            self,
            "Select Measurement CSV",
            "",
            "CSV Files (*.csv);;Data Files (*.txt *.dat)"
        )
        if not file_path_medida or not os.path.exists(file_path_medida):
            self.label_status.setText("❌ No measurement file selected.")
            return
    
        raw = pd.read_csv(file_path_medida, header=None)
        raw = raw.apply(pd.to_numeric, errors="coerce").dropna(how="any")
        raw = raw.values.astype(float)
        self.TD = raw[0, 1:]        # delay (ps)
        self.WL = raw[1:, 0]        # wavelength (nm)
        self.medida = raw[1:, 1:]   # ΔA(λ, t)
        self.medida[np.isnan(self.medida)] = 0
    
        # --- Seleccionar archivo de solvente ---
        file_path_solvente, _ = QFileDialog.getOpenFileName(
            self,
            "Select Solvent CSV",
            "",
            "CSV Files (*.csv);;Data Files (*.txt *.dat)"
        )
        if not file_path_solvente or not os.path.exists(file_path_solvente):
            self.label_status.setText("❌ No solvent file selected.")
            return
    
        rawSol = pd.read_csv(file_path_solvente, header=None)
        rawSol = rawSol.apply(pd.to_numeric, errors="coerce").dropna(how="any")
        rawSol = rawSol.values.astype(float)
        self.TDSol = rawSol[0, 1:]
        self.WLSol = rawSol[1:, 0]
        self.solvente = rawSol[1:, 1:]
        self.solvente[np.isnan(self.solvente)] = 0
    
        # --- Configurar sliders de λ ---
        nwl = len(self.WL)
        self.slider_min.setMinimum(0)
        self.slider_min.setMaximum(nwl - 1)
        self.slider_max.setMinimum(0)
        self.slider_max.setMaximum(nwl - 1)
        self.slider_min.setValue(0)
        self.slider_max.setValue(nwl - 1)
    
        # --- Calcular mapa inicial ---
        self.label_status.setText("✅ TAS data loaded")
        self.update_am_sf()
    
        # --- Definir ruta base para compatibilidad con FLUPSAnalyzer ---
        self.file_path = file_path_medida
    
        # --- Asegurar que los botones t₀ funcionan ---
        if hasattr(self, "btn_plot"):
            self.btn_plot.setEnabled(True)
        if hasattr(self, "btn_select"):
            self.btn_select.setEnabled(True)
        if hasattr(self, "btn_fit"):
            self.btn_fit.setEnabled(True)

    def fit_t0_points(self):
        if not getattr(self, "clicked_points", None) or len(self.clicked_points) < 2:
            QMessageBox.warning(self, "Not enough points", "Select at least 2 points on the map.")
            return
    
        w_points = np.array([p['x'] for p in self.clicked_points])
        t0_points = np.array([p['y'] for p in self.clicked_points])
    
        try:
            # aplicar corrección t0 sobre self.data ya con fringe eliminado
            self.update_am_sf()  # asegura que pump_mask se aplique antes del fit
            result = fit_t0(w_points, t0_points, self.WL, self.TD, self.data)
        except Exception as e:
            QMessageBox.critical(self, "Fit error", str(e))
            return
    
        self.result_fit = result
        self.data_corrected = result['corrected']
    
        # dibujar curva del fit sobre mapa
        if self.fit_line_artist is not None:
            try: self.fit_line_artist.remove()
            except Exception: pass
        self.fit_line_artist, = self.ax_map.plot(result['fit_x'], result['fit_y'], 'r-', lw=2, label="t₀ fit")
        self.ax_map.legend()
        self.canvas.draw_idle()
    
        # guardar datos
        self.btn_show_corr.setEnabled(True)
        import os
        base_dir = os.path.dirname(self.file_path)
        base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        save_dir = os.path.join(base_dir, f"{base_name}_Results")
        os.makedirs(save_dir, exist_ok=True)
    
        # asegurarse de guardar la data con fringe eliminado
        self.update_am_sf()  # ✅ aplicamos pump_mask
        data_corr = np.copy(self.data)  # guardamos copia segura
        WL = self.WL
        TD = self.TD
    
        np.save(os.path.join(save_dir, f"{base_name}_treated_data.npy"),
                {'data_c': data_corr, 'WL': WL, 'TD': TD})
    
        np.savetxt(os.path.join(save_dir, f"{base_name}_WL.txt"), WL,
                   fmt='%.6f', header='Wavelength (nm)', comments='')
        np.savetxt(os.path.join(save_dir, f"{base_name}_TD.txt"), TD,
                   fmt='%.6f', header='Delay (ps)', comments='')
    
        with open(os.path.join(save_dir, f"{base_name}_kin.txt"), 'w') as f:
            f.write("\t".join([f"{base_name}_kin_{round(wl,1)}nm" for wl in WL]) + "\n")
            np.savetxt(f, data_corr.T, fmt='%.6e', delimiter='\t')
    
        with open(os.path.join(save_dir, f"{base_name}_spec.txt"), 'w') as f:
            f.write("\t".join([f"{base_name}_spec_{td:.2f}ps" for td in TD]) + "\n")
            np.savetxt(f, data_corr, fmt='%.6e', delimiter='\t')
    
        t0_lambda = result['t0_lambda']
        popt = result['popt']
        method = result['method']
    
        t0_file = os.path.join(save_dir, f"{base_name}_t0_fit.txt")
        np.savetxt(t0_file, np.column_stack((WL, t0_lambda)),
                   fmt='%.6f', header='Wavelength (nm)\t t0 (ps)', comments='')
    
        params_file = os.path.join(save_dir, f"{base_name}_fit_params.txt")
        with open(params_file, 'w') as f:
            f.write(f"Fit method: {method}\n")
            f.write("Fit parameters:\n")
            if method.startswith('poly'):
                names = ['c4', 'c3', 'c2', 'c1', 'c0']
            else:
                names = ['a', 'b', 'c', 'd']
            for name, val in zip(names, popt):
                f.write(f"  {name} = {val:.6g}\n")
    
        QMessageBox.information(self, "Files saved",
                                f"Results saved in:\n{save_dir}")
        QMessageBox.information(self, "t₀ Fit Result",
                                f"Fit completed using {method} model.\nParameters: {np.round(popt,4)}")

    # ------------------------------------------------------------------
    # ACTUALIZACIÓN DE MAPA TRAS SLIDERS
    # ------------------------------------------------------------------
    def update_am_sf(self):
        if self.medida is None or self.solvente is None:
            return
    
        if hasattr(self, "_updating_am_sf") and self._updating_am_sf:
            return
        self._updating_am_sf = True
    
        am = self.slider_am.value() / 100.0
        sf = self.spin_sf.value()
        interpSol = RegularGridInterpolator(
            (self.WLSol, self.TDSol),
            self.solvente,
            bounds_error=False,
            fill_value=0
        )
    
        WL_grid, TD_grid = np.meshgrid(self.WL, self.TD, indexing="ij")
        points = np.column_stack([WL_grid.ravel(), (TD_grid - sf).ravel()])
        solvente_interp = interpSol(points).reshape(len(self.WL), len(self.TD)) * am
    
        self.data = self.medida - solvente_interp
    
        # aplicar máscara si existe
        if self.pump_mask is not None:
            self.data[self.pump_mask] = 1e-10
    
        self.update_wl_range()
    
        if hasattr(self, "global_fit_panel") and self.global_fit_panel is not None:
            self.global_fit_panel.update_from_parent()
    
        self._updating_am_sf = False
    # ------------------------------------------------------------------
    # DIBUJAR MAPA ΔA
    # ------------------------------------------------------------------
    def plot_map(self, show_fit=False):
        """Dibuja el mapa principal TAS sin normalizar y con subplots ajustados."""
        if self.data is None:
            return
    
        self.ax_map.clear()
        if self.cbar:
            try:
                self.cbar.remove()
            except Exception:
                pass
            self.cbar = None
    
        # Escala real de los datos con pequeño margen
        vmin = np.nanmin(self.data)
        vmax = np.nanmax(self.data)
        margin = 0.05 * (vmax - vmin)
        vmin -= margin
        vmax += margin
    
        # Dibujar pcolormesh
        self.pcm = self.ax_map.pcolormesh(
            self.WL, self.TD, self.data.T,
            shading="auto",
            cmap="jet",
            vmin=vmin,
            vmax=vmax
        )
    
        self.ax_map.set_xlabel("Wavelength (nm)")
        self.ax_map.set_ylabel("Delay (ps)")
        self.ax_map.set_title("ΔA Map (TAS)")
        self.ax_map.set_yscale("symlog")
    
        # Colorbar
        divider = make_axes_locatable(self.ax_map)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        self.cbar = self.figure.colorbar(self.pcm, cax=cax, label="ΔA")
    
        # Líneas cruzadas y marcador iniciales
        x0 = np.median(self.WL)
        y0 = np.median(self.TD)
        self.vline_map = self.ax_map.axvline(x0, color='k', ls='--', lw=1, zorder=6)
        self.hline_map = self.ax_map.axhline(y0, color='k', ls='--', lw=1, zorder=6)
        self.marker_map, = self.ax_map.plot([x0], [y0], 'wx', markersize=8, markeredgewidth=2, zorder=7)
    
        # Mostrar fit si existe
        if show_fit and hasattr(self, 'result_fit') and self.result_fit is not None:
            fit_x = self.result_fit.get('fit_x', None)
            fit_y = self.result_fit.get('fit_y', None)
            if fit_x is not None and fit_y is not None:
                if hasattr(self, 'fit_line_artist') and self.fit_line_artist is not None:
                    try: self.fit_line_artist.remove()
                    except: pass
                self.fit_line_artist, = self.ax_map.plot(fit_x, fit_y, 'r-', lw=2, label="t₀ fit")
                self.ax_map.legend()
    
        # Ajustar subplots pequeños al rango real
        self.update_small_cuts(x0, y0)
    
        # Conectar eventos si no lo estaban
        if self.cid_click is None:
            self.cid_click = self.canvas.mpl_connect("button_press_event", self.on_click_map)
        if self.cid_move is None:
            self.cid_move = self.canvas.mpl_connect("motion_notify_event", self.on_move_map)
    
        self.canvas.draw_idle()


    def update_small_cuts(self, x, y, WL_sel=None, data_sel=None):
        """Actualiza los subplots de cinética y espectro para la posición (x, y)."""
        if x is None or y is None or self.data is None:
            return
    
        # Si no se pasan, usar sliders por defecto
        if WL_sel is None or data_sel is None:
            wl_min_idx = self.slider_min.value()
            wl_max_idx = self.slider_max.value()
            wl_max_idx = min(wl_max_idx + 1, len(self.WL))
            WL_sel = self.WL[wl_min_idx:wl_max_idx]
            data_sel = self.data[wl_min_idx:wl_max_idx, :]  # (n_wl_visible, n_td)
    
        if WL_sel.size == 0:
            return
    
        idx_wl = np.argmin(np.abs(WL_sel - x))
        idx_td = np.argmin(np.abs(self.TD - y))
    
        # Cinética (fila idx_wl)
        y_time = data_sel[idx_wl, :]
        self.cut_time_small.set_data(self.TD, y_time)
        self.ax_time_small.relim()
        self.ax_time_small.autoscale_view()
        self.ax_time_small.set_title(f"Kinetics at {WL_sel[idx_wl]:.1f} nm")
    
        # Espectro (columna idx_td)
        y_spec = data_sel[:, idx_td]
        self.cut_spec_small.set_data(WL_sel, y_spec)
        self.ax_spec_small.relim()
        self.ax_spec_small.autoscale_view()
        self.ax_spec_small.set_title(f"Spectra at {self.TD[idx_td]:.2f} ps")
    
        self.canvas.draw_idle()


    # ------------------------------------------------------------------
    # EVENTO DE MOVIMIENTO DE RATÓN
    # ------------------------------------------------------------------
    def on_move_map(self, event):
        """Actualiza líneas y subplots al mover el cursor sobre el mapa."""
        if event.inaxes != self.ax_map or self.data is None:
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        # Líneas cruzadas
        if self.vline_map is None:
            self.vline_map = self.ax_map.axvline(x, color='k', ls='--', lw=1, zorder=6)
        else:
            self.vline_map.set_xdata([x, x])

        if self.hline_map is None:
            self.hline_map = self.ax_map.axhline(y, color='k', ls='--', lw=1, zorder=6)
        else:
            self.hline_map.set_ydata([y, y])

        # Marcador
        if self.marker_map is None:
            self.marker_map, = self.ax_map.plot([x], [y], 'wx', markersize=8, markeredgewidth=2, zorder=7)
        else:
            self.marker_map.set_data([x], [y])

        # Actualizar subplots
        self.update_small_cuts(x, y)
        
# -*- coding: utf-8 -*-
"""
GlobalFitPanel - PyQt5 dialog that reproduces the behavior of your Tk script
Embedded colormesh (Experimental/Fit/Residual tabs) + external DAS/residuals figures.
Author: generated for you (adapted to your project)
Requires: fit.py (with eval_global_model, convolved_exp, crop_spectrum, crop_kinetics, binning, load_npy)
"""


class GlobalFitPanel(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Global Fit — FLUPS")
        self.resize(1000, 700)

        # Data placeholders: prefer parent's data if present
        self.parent_app = parent
        self.data_c = None   # shape: (numWL, numTD)
        self.TD = None
        self.WL = None
        self.base_dir = None

        # fit-related
        self.numExp = 3
        self.t0_choice = 'No'
        self.ini = None
        self.limi = None
        self.lims = None
        self.tech = 'FLUPS'

        # UI elements
        layout = QVBoxLayout()

        top_row = QHBoxLayout()
        self.label_status = QLabel("No data loaded")
        top_row.addWidget(self.label_status)

        self.btn_load = QPushButton("Load treated .npy")
        self.btn_load.clicked.connect(self.load_data)
        top_row.addWidget(self.btn_load)

        self.btn_use_parent = QPushButton("Use data from main app")
        self.btn_use_parent.clicked.connect(self.use_parent_data)
        top_row.addWidget(self.btn_use_parent)

        self.btn_run = QPushButton("Run Fit")
        self.btn_run.clicked.connect(self.run_fit_pipeline)
        self.btn_run.setEnabled(False)
        top_row.addWidget(self.btn_run)

        self.btn_show_das = QPushButton("Show DAS / Residuals")
        self.btn_show_das.clicked.connect(self.plot_das_and_more)
        self.btn_show_das.setEnabled(False)
        top_row.addWidget(self.btn_show_das)

        layout.addLayout(top_row)

        # Tabs for experimental / fit / residual
        self.tabs = QTabWidget()
        # Each tab will host a FigureCanvas (colormesh)
        self.tab_exp = QWidget()
        self.tab_fit = QWidget()
        self.tab_resid = QWidget()
        self.tabs.addTab(self.tab_exp, "Experimental")
        self.tabs.addTab(self.tab_fit, "Fit")
        self.tabs.addTab(self.tab_resid, "Residual")
        layout.addWidget(self.tabs, stretch=1)

    # 🔧 Forzar estilo visible (fondo claro, texto oscuro)
        self.tabs.setStyleSheet("""
            QTabWidget::pane { 
                border: 1px solid #aaa; 
                background: white;
            }
            QTabBar::tab {
                background: #f0f0f0;
                color: black;
                padding: 6px 12px;
                border: 1px solid #aaa;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                font-weight: bold;
            }
        """)
        # For each tab create a canvas
        self.canvas_exp, self.ax_exp = self._create_canvas_for_tab(self.tab_exp)
        self.canvas_fit, self.ax_fit = self._create_canvas_for_tab(self.tab_fit)
        self.canvas_resid, self.ax_resid = self._create_canvas_for_tab(self.tab_resid)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)

        # Internal plotting state
        self.pcm_exp = None
        self.cbar_exp = None
        self.pcm_fit = None
        self.cbar_fit = None
        self.pcm_resid = None
        self.cbar_resid = None

        # results storage
        self.fit_result = None
        self.fit_x = None
        self.fit_resid = None
        self.fit_fitres = None
        self.ci = None
        self.As = None
        self.errAs = None
        self.t0s = None
        self.errt0s = None

    def _create_canvas_for_tab(self, tab_widget):
        """Create a Matplotlib FigureCanvas inside the given tab QWidget."""
        fig = plt.Figure(figsize=(6,4))
        ax = fig.add_subplot(111)
        canvas = FigureCanvas(fig)
        tab_layout = QVBoxLayout()
        tab_layout.addWidget(canvas)
        tab_widget.setLayout(tab_layout)
        return canvas, ax

    def use_parent_data(self):
        """Cargar datos del parent y actualizar canvas"""
        self.update_from_parent()
        self.btn_run.setEnabled(True)
        self.btn_show_das.setEnabled(False)
    
    def update_from_parent(self):
        """Actualizar data_c con los datos actuales de TAS o FLUPS"""
        p = self.parent_app
        if p is None:
            return
    
        if getattr(p, "is_TAS_mode", False):
            if hasattr(p, "data") and p.data is not None:
                self.data_c = np.array(p.data, copy=True)
            else:
                return
        else:  # FLUPS
            if hasattr(p, "data_corrected") and p.data_corrected is not None:
                self.data_c = np.array(p.data_corrected, copy=True)
            elif hasattr(p, "data") and p.data is not None:
                self.data_c = np.array(p.data, copy=True)
            else:
                return
    
        self.WL = getattr(p, "WL", None)
        self.TD = getattr(p, "TD", None)
        self._update_exp_canvas()



    def load_data(self):
        """Call fit.load_npy to open a file dialog and load processed .npy"""
        try:
            data_c, TD, WL, base_dir = fit.load_npy(self)
            self.data_c = data_c.copy()
            self.TD = TD
            self.WL = WL
            self.base_dir = base_dir
            self.label_status.setText(f"Loaded: {len(self.WL)} WL, {len(self.TD)} TD")
            self.btn_run.setEnabled(True)
            self._update_exp_canvas()
            self.btn_show_das.setEnabled(False)
        except Exception as e:
            QMessageBox.critical(self, "Error loading file", str(e))

    def _clear_colorbar_if_exists(self, cbar):
        try:
            if cbar is not None:
                cbar.remove()
        except Exception:
            pass

    def _update_exp_canvas(self):
        """Draw experimental colormesh on the Experimental tab without normalizing."""
        if self.data_c is None:
            return
    
        self.ax_exp.clear()
        self._clear_colorbar_if_exists(self.cbar_exp)
    
        # Escala real de los datos con margen
        vmin = np.nanmin(self.data_c)
        vmax = np.nanmax(self.data_c)
        margin = 0.05 * (vmax - vmin)
        vmin -= margin
        vmax += margin
    
        # pcolormesh
        self.pcm_exp = self.ax_exp.pcolormesh(
            self.WL, self.TD, self.data_c.T,
            shading="auto",
            cmap="jet",
            vmin=vmin,
            vmax=vmax
        )
    
        self.ax_exp.set_xlabel("Wavelength (nm)")
        self.ax_exp.set_ylabel("Delay (ps)")
        self.ax_exp.set_yscale("symlog")
        self.ax_exp.set_title("Experimental ΔA")
    
        divider = make_axes_locatable(self.ax_exp)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        self.cbar_exp = self.canvas_exp.figure.colorbar(self.pcm_exp, cax=cax, label='ΔA')
    
        self.canvas_exp.draw_idle()



    def _update_fit_canvas(self):
        """Draw fitted colormesh (if available)"""
        if self.fit_fitres is None:
            self.ax_fit.clear()
            self.canvas_fit.draw_idle()
            return
        self.ax_fit.clear()
        self._clear_colorbar_if_exists(self.cbar_fit)
        self.pcm_fit = self.ax_fit.pcolormesh(self.WL, self.TD, self.fit_fitres.T, shading='auto', cmap='jet')
        self.ax_fit.set_xlabel("Wavelength (nm)")
        self.ax_fit.set_ylabel("Delay (ps)")
        self.ax_fit.set_yscale("symlog")
        self.ax_fit.set_title("Fit reconstructed")
        divider = make_axes_locatable(self.ax_fit)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        self.cbar_fit = self.canvas_fit.figure.colorbar(self.pcm_fit, cax=cax, label='ΔA')
        self.canvas_fit.draw_idle()

    def _update_resid_canvas(self):
        """Draw residual colormesh (if available)"""
        if self.fit_resid is None:
            self.ax_resid.clear()
            self.canvas_resid.draw_idle()
            return
        self.ax_resid.clear()
        self._clear_colorbar_if_exists(self.cbar_resid)
        self.pcm_resid = self.ax_resid.pcolormesh(self.WL, self.TD, self.fit_resid.T, shading='auto', cmap='jet')
        self.ax_resid.set_xlabel("Wavelength (nm)")
        self.ax_resid.set_ylabel("Delay (ps)")
        self.ax_resid.set_title("Residuals")
        divider = make_axes_locatable(self.ax_resid)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        self.cbar_resid = self.canvas_resid.figure.colorbar(self.pcm_resid, cax=cax, label='ΔA')
        self.canvas_resid.draw_idle()

    def run_fit_pipeline(self):
        """Full interactive pipeline (crop spectrum, crop kinetics, binning, params, run fit)."""
        try:
            if self.data_c is None:
                QMessageBox.warning(self, "No data", "Load data first (from main app or .npy).")
                return

            # 1) Crop spectrum?
            crop = QMessageBox.question(self, "Crop Spectrum", "Do you want to crop the spectrum?", QMessageBox.Yes | QMessageBox.No)
            if crop == QMessageBox.Yes:
                WLmin, ok1 = QInputDialog.getDouble(self, "Lower WL", "Enter lower wavelength:", decimals=6)
                WLmax, ok2 = QInputDialog.getDouble(self, "Upper WL", "Enter upper wavelength:", decimals=6)
                if ok1 and ok2:
                    self.data_c, self.WL = fit.crop_spectrum(self.data_c, self.WL, min(WLmin, WLmax), max(WLmin, WLmax))
                    self._update_exp_canvas()

            # 2) Crop kinetics?
            crop_t = QMessageBox.question(self, "Crop Kinetics", "Do you want to crop the kinetics?", QMessageBox.Yes | QMessageBox.No)
            if crop_t == QMessageBox.Yes:
                TDmin, ok1 = QInputDialog.getDouble(self, "Lower TD", "Enter lower time:", decimals=6)
                TDmax, ok2 = QInputDialog.getDouble(self, "Upper TD", "Enter upper time:", decimals=6)
                if ok1 and ok2:
                    self.data_c, self.TD = fit.crop_kinetics(self.data_c, self.TD, min(TDmin, TDmax), max(TDmin, TDmax))
                    self._update_exp_canvas()

            # 3) Binning?
            bin_ans = QMessageBox.question(self, "Binning", "Do you want to bin wavelengths?", QMessageBox.Yes | QMessageBox.No)
            if bin_ans == QMessageBox.Yes:
                bin_size, ok = QInputDialog.getInt(self, "Binning", "Points to average:", 5, 1)
                if ok and bin_size > 1:
                    self.data_c, self.WL = fit.binning(self.data_c, self.WL, bin_size)
                    self._update_exp_canvas()

            # 4) Number of exponentials & t0 choice
            numExp, ok = QInputDialog.getInt(self, "Exponents", "Number of exponentials:", self.numExp, 1)
            if ok:
                self.numExp = int(numExp)
            t0_choice_bool = QMessageBox.question(self, "t0 Choice", "Fit independent t0s?", QMessageBox.Yes | QMessageBox.No)
            self.t0_choice = 'Yes' if t0_choice_bool == QMessageBox.Yes else 'No'

            # 5) Technique selection (affects initial guesses/ranges same as script)
            tech_choice, ok = QInputDialog.getItem(self, "Technique", "Select technique:", ["TAS", "FLUPS", "TCSPC"], 1, False)
            if ok:
                self.tech = tech_choice

            # 6) Initialize ini, limi, lims (using same shapes as script)
            numWL = len(self.WL)
            if self.t0_choice == 'Yes':
                L = 1 + self.numExp + numWL*(self.numExp+1)
            else:
                L = 2 + self.numExp + numWL*self.numExp
            self.ini = np.zeros(L)
            self.limi = -np.inf * np.ones(L)
            self.lims = np.inf * np.ones(L)

            # default taus similar to your script
            taus_defaults = [0.5,14,200,1000,4000,6000]
            # fill defaults following your script logic
            if self.t0_choice == 'Yes':
                # w
                if self.tech == 'TAS':
                    self.ini[0] = 0.15
                elif self.tech == 'FLUPS':
                    self.ini[0] = 0.35
                elif self.tech == 'TCSPC':
                    self.ini[0] = 0.1
                self.limi[0] = 0.05
                self.lims[0] = 2
                # taus
                for n in range(self.numExp):
                    idx = 1 + n
                    self.ini[idx] = taus_defaults[n] if n < len(taus_defaults) else 1.0
                    self.limi[idx] = 0.05
                    self.lims[idx] = 1e18
                # per-wl t0s and As
                for k in range(numWL):
                    t0_idx = 1 + self.numExp + k*(self.numExp+1)
                    self.ini[t0_idx] = 0.0
                    self.limi[t0_idx] = -2.0
                    self.lims[t0_idx] = 2.0
                    for n in range(self.numExp):
                        Aidx = t0_idx + 1 + n
                        if self.tech == 'TAS':
                            self.ini[Aidx] = 0.005
                            self.limi[Aidx] = -1
                            self.lims[Aidx] = 1
                        elif self.tech == 'FLUPS':
                            self.ini[Aidx] = 5
                            self.limi[Aidx] = -500
                            self.lims[Aidx] = 500
                        else:  # TCSPC
                            self.ini[Aidx] = 1000
                            self.limi[Aidx] = -50000
                            self.lims[Aidx] = 50000
            else:
                # common t0
                if self.tech == 'TAS':
                    self.ini[0] = 0.15
                elif self.tech == 'FLUPS':
                    self.ini[0] = 0.35
                elif self.tech == 'TCSPC':
                    self.ini[0] = 0.1
                self.limi[0] = 0.05
                self.lims[0] = 2
                # t0 common
                self.ini[1] = 0.0
                self.limi[1] = -2.0
                self.lims[1] = 2.0
                # taus
                for n in range(self.numExp):
                    idx = 2 + n
                    self.ini[idx] = taus_defaults[n] if n < len(taus_defaults) else 1.0
                    self.limi[idx] = 0.05
                    self.lims[idx] = 1e18
                # amplitudes
                for k in range(numWL):
                    for n in range(self.numExp):
                        Aidx = 2 + self.numExp + n + k*self.numExp
                        if self.tech == 'TAS':
                            self.ini[Aidx] = 0.005
                            self.limi[Aidx] = -1
                            self.lims[Aidx] = 1
                        elif self.tech == 'FLUPS':
                            self.ini[Aidx] = 5
                            self.limi[Aidx] = -500
                            self.lims[Aidx] = 500
                        else:
                            self.ini[Aidx] = 1000
                            self.limi[Aidx] = -50000
                            self.lims[Aidx] = 50000

            # 7) Ask user whether to edit initial guesses
            edit_ans = QMessageBox.question(self, "Initial guess", "Would you like to edit initial guesses and limits manually?", QMessageBox.Yes | QMessageBox.No)
            if edit_ans == QMessageBox.Yes:
                self._open_guess_editor_and_update()

            # 8) Run least_squares with progress callback
            self._run_least_squares_with_progress()

            # After fit: compute outputs, update canvases and enable DAS button
            self._postprocess_fit_and_save()

        except Exception as e:
            QMessageBox.critical(self, "Error running fit", str(e))

    def _open_guess_editor_and_update(self):
        """Simple QTableWidget to edit ini/limi/lims (variable names optional)."""
        L = len(self.ini)
        dlg = QDialog(self)
        dlg.setWindowTitle("Edit initial guesses and limits")
        dlg.resize(800, 400)
        v = QVBoxLayout()
        table = QTableWidget(L, 4)
        table.setHorizontalHeaderLabels(["Index","Guess","Lower","Upper"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for i in range(L):
            table.setItem(i, 0, QTableWidgetItem(str(i)))
            table.setItem(i, 1, QTableWidgetItem(str(self.ini[i])))
            table.setItem(i, 2, QTableWidgetItem(str(self.limi[i])))
            table.setItem(i, 3, QTableWidgetItem(str(self.lims[i])))
        v.addWidget(table)
        btn_row = QHBoxLayout()
        btn_ok = QPushButton("OK")
        btn_cancel = QPushButton("Cancel")
        btn_row.addWidget(btn_ok)
        btn_row.addWidget(btn_cancel)
        v.addLayout(btn_row)
        dlg.setLayout(v)

        def on_ok():
            # read back values safely
            try:
                for i in range(L):
                    self.ini[i] = float(table.item(i,1).text())
                    self.limi[i] = float(table.item(i,2).text())
                    self.lims[i] = float(table.item(i,3).text())
            except Exception as e:
                QMessageBox.warning(self, "Invalid input", f"Could not parse numbers: {e}")
                return
            dlg.accept()

        btn_ok.clicked.connect(on_ok)
        btn_cancel.clicked.connect(dlg.reject)

        dlg.exec_()

    def _run_least_squares_with_progress(self):
        """Run least_squares directly using fit.eval_global_model as model,
        so we can pass a callback to update the progress bar."""
        # build residual function (keeps same flatten ordering as script)
        numWL = len(self.WL)
        TD = self.TD
        data_flat = self.data_c.T.flatten()

        def residuals(x):
            F = fit.eval_global_model(x, TD, self.numExp, numWL, self.t0_choice)
            return F.flatten() - data_flat

        # progress bookkeeping
        self.progress_bar.setValue(0)
        iterations = {'count':0}

        def callback(xk, *args, **kwargs):
            iterations['count'] += 1
            # update value cyclically if no better estimate of total
            val = (iterations['count'] % 100)
            self.progress_bar.setValue(val)
            QTimer.singleShot(1, lambda: None)  # keep UI responsive

        try:
            # call least_squares (verbose left out; we show our own progress)
            res = least_squares(residuals, self.ini, bounds=(self.limi, self.lims), jac='2-point', callback=callback)
        except Exception as e:
            QMessageBox.critical(self, "Optimization error", str(e))
            raise

        # finalize progress bar
        self.progress_bar.setValue(100)
        self.fit_result = res
        self.fit_x = res.x

    def _postprocess_fit_and_save(self):
        """Compute fitres, resid, covariance, As, t0s, taus and save outputs."""
        if self.fit_result is None:
            QMessageBox.warning(self, "No result", "No fit result available.")
            return
    
        x = self.fit_x
        numWL = len(self.WL)
        TD = self.TD
    
        # compute fitted matrix and residuals
        F_mat = fit.eval_global_model(x, TD, self.numExp, numWL, self.t0_choice)
        fitres = F_mat.T
        resid = self.data_c - fitres
    
        self.fit_fitres = fitres
        self.fit_resid = resid
    
        # covariance estimation
        J = self.fit_result.jac
        try:
            cov = np.linalg.inv(J.T @ J)
            s_sq = np.sum(resid.T.flatten()**2) / (len(resid.T.flatten()) - len(x))
            ci = cov * s_sq
        except np.linalg.LinAlgError:
            QMessageBox.warning(self, "Warning", "Jacobian singular; covariance can't be estimated.")
            ci = np.zeros((len(x), len(x)))
    
        self.ci = ci
    
        # extract taus, As, t0s and their errors
        As = np.zeros((self.numExp, numWL))
        errAs = np.zeros_like(As)
        errtaus = np.zeros(self.numExp)
    
        if self.t0_choice == 'Yes':
            t0s = np.zeros(numWL)
            errt0s = np.zeros(numWL)
            for n in range(self.numExp):
                tau_idx = 1 + n
                errtaus[n] = np.sqrt(np.abs(ci[tau_idx, tau_idx])) if ci.size else 0.0
            for n in range(numWL):
                t0_idx = 1 + self.numExp + n*(self.numExp+1)
                t0s[n] = x[t0_idx]
                errt0s[n] = np.sqrt(np.abs(ci[t0_idx, t0_idx])) if ci.size else 0.0
                for k in range(self.numExp):
                    Aidx = t0_idx + 1 + k
                    As[k, n] = x[Aidx]
                    errAs[k, n] = np.sqrt(np.abs(ci[Aidx, Aidx])) if ci.size else 0.0
        else:
            t0s = x[1]
            errt0s = np.sqrt(np.abs(ci[1,1])) if ci.size else 0.0
            for n in range(self.numExp):
                tau_idx = 2 + n
                errtaus[n] = np.sqrt(np.abs(ci[tau_idx, tau_idx])) if ci.size else 0.0
            for n in range(numWL):
                for k in range(self.numExp):
                    Aidx = 2 + self.numExp + k + n*self.numExp
                    As[k, n] = x[Aidx]
                    errAs[k, n] = np.sqrt(np.abs(ci[Aidx, Aidx])) if ci.size else 0.0
    
        self.As = As
        self.errAs = errAs
        self.t0s = t0s
        self.errt0s = errt0s
        self.errtaus = errtaus
    
        # update embedded canvases
        self._update_fit_canvas()
        self._update_resid_canvas()
        self.btn_show_das.setEnabled(True)
    
        # --- Guardar outputs en la carpeta del CSV cargado ---
        base_dir = getattr(self, "base_dir", None)
        
        # Si no existe o es None, intenta recuperar desde el parent_app
        if not base_dir:
            if hasattr(self, "parent_app") and self.parent_app is not None:
                p = self.parent_app
                if hasattr(p, "base_dir") and p.base_dir:
                    base_dir = p.base_dir
                elif hasattr(p, "csv_path") and p.csv_path:
                    base_dir = os.path.dirname(p.csv_path)
        
        # Fallback final: el directorio actual (donde está el .py)
        if not base_dir:
            base_dir = os.getcwd()
        
        print("Guardando resultados en:", base_dir)
        
        outdir = os.path.join(base_dir, "fit")
        os.makedirs(outdir, exist_ok=True)
    
        np.save(os.path.join(outdir, "GFitResults.npy"), {
            "numExp": self.numExp, "numWL": numWL, "x": x, "data_c": self.data_c,
            "WL_a": self.WL, "TD_a": self.TD, "fitres": fitres, "ci": ci,
            "As": As, "t0s": t0s, "errAs": errAs, "errt0s": errt0s,
            "jacobian": self.fit_result.jac
        })
    
        # text dumps
        with open(os.path.join(outdir, "Amplitudes.txt"), 'w') as fid:
            fid.write('\t'.join([f'A{i+1}' for i in range(self.numExp)]) + '\n')
            for i in range(numWL):
                fid.write('\t'.join([f'{As[j, i]:.6e}' for j in range(self.numExp)]) + '\n')
        with open(os.path.join(outdir, "WL.txt"), 'w') as fid:
            fid.write('WL\n')
            for i in range(numWL):
                fid.write(f'{self.WL[i]:.6f}\n')
        with open(os.path.join(outdir, "TD.txt"), 'w') as fid:
            fid.write('TD\n')
            for i in range(len(self.TD)):
                fid.write(f'{self.TD[i]:.6f}\n')
        with open(os.path.join(outdir, "GFit.txt"), 'w') as fid:
            fid.write('\t'.join([f'kin_{round(self.WL[i]):.0f}_nm' for i in range(numWL)]) + '\n')
            for i in range(len(self.TD)):
                fid.write('\t'.join([f'{fitres[j, i]:.6e}' for j in range(numWL)]) + '\n')
        with open(os.path.join(outdir, "GFit_resid.txt"), 'w') as fid:
            fid.write('\t'.join([f'kin_{round(self.WL[i]):.0f}_nm' for i in range(numWL)]) + '\n')
            for i in range(len(self.TD)):
                fid.write('\t'.join([f'{resid[j, i]:.6e}' for j in range(numWL)]) + '\n')
    
        QMessageBox.information(self, "Fit finished",
                                f"Fit completed.\nRMSD: {np.sqrt(np.sum(resid**2)/resid.size):.6e}\nResults saved in:\n{outdir}")

    def plot_das_and_more(self):
        """Open external Matplotlib figures for DAS, t0s and interactive per-wavelength fits."""
        if self.As is None:
            QMessageBox.warning(self, "No fit", "Run a fit first.")
            return
    
        # --- Carpeta donde se guardarán los gráficos ---
        base_dir = getattr(self, "base_dir", None)
        
        # Si no está definida, intenta recuperarla desde el parent_app
        if not base_dir:
            if hasattr(self, "parent_app") and self.parent_app is not None:
                p = self.parent_app
                if hasattr(p, "base_dir") and p.base_dir:
                    base_dir = p.base_dir
                elif hasattr(p, "csv_path") and p.csv_path:
                    base_dir = os.path.dirname(p.csv_path)
        
        # Fallback final: si todo falla, usa el directorio actual
        if not base_dir:
            base_dir = os.getcwd()
        
        print("Guardando gráficos en:", base_dir)

        outdir = os.path.join(base_dir, "Plots")
        os.makedirs(outdir, exist_ok=True)

    
        # define palette fija
        palette = ['blue', 'red', 'green', 'orange', 'yellow']
        
        # limitar la cantidad de colores a numExp
        colors = palette[:self.numExp]
        
        plt.figure(figsize=(8,6))
        plt.gcf().set_facecolor('white')
        
        if self.t0_choice == 'Yes':
            for n in range(self.numExp):
                label = f"τ{n+1} = {self.fit_x[n+1]:.3f} ± {self.errtaus[n]:.3f}"
                plt.plot(self.WL, self.As[n,:], '-', lw=2, label=label, color=colors[n])
                plt.fill_between(self.WL, self.As[n,:]-self.errAs[n,:], self.As[n,:]+self.errAs[n,:], alpha=0.2)
        else:
            for n in range(self.numExp):
                label = f"τ{n+1} = {self.fit_x[n+2]:.3f} ± {self.errtaus[n]:.3f}"
                plt.plot(self.WL, self.As[n,:], '-', lw=2, label=label, color=colors[n])
                plt.fill_between(self.WL, self.As[n,:]-self.errAs[n,:], self.As[n,:]+self.errAs[n,:], alpha=0.2)
        
        plt.axhline(0, color='k', lw=1, ls='--')
        plt.title("Decay-Associated Spectra (DAS)")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "DAS.png"), dpi=300, bbox_inches='tight')
        plt.show(block=False)
    
        # t0s plot if independent
        if self.t0_choice == 'Yes':
            plt.figure()
            plt.plot(self.WL, self.t0s)
            plt.axhline(0,color='k')
            plt.title("Time zero")
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("t0 (ps)")
            plt.savefig(os.path.join(outdir, "T0s.png"), dpi=200)
            plt.show(block=False)
    
        # Residuals plot
        fig, ax = plt.subplots(figsize=(9, 6))
        pcm = ax.pcolormesh(self.WL, self.TD, self.fit_resid.T, shading='auto', cmap='jet')
        ax.set_title("Residual from fit")
        ax.set_ylabel("Delay (ps)")
        ax.set_xlabel("Wavelength (nm)")
        fig.colorbar(pcm, ax=ax, label='ΔA')
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "Residual.png"), dpi=200)
        plt.close(fig)
    
        # Interactive per-wavelength plots
        cont = True
        while cont:
            wl_str, ok = QInputDialog.getText(
                self,
                "Check fit",
                "Enter wavelength to check the fit (or Cancel to finish):",
                text=str(self.WL[len(self.WL)//2])
            )
            if not ok:
                break
            try:
                Wlobs = float(wl_str)
            except Exception:
                QMessageBox.warning(self, "Bad input", "Enter a valid number.")
                continue
            pos3 = int(np.argmin(np.abs(self.WL - Wlobs)))
            fig, (ax1, ax2) = plt.subplots(1,2, sharey=True, figsize=(9,4))
            ax1.plot(self.TD, self.data_c[pos3,:], 'b.', label='Data')
            ax1.plot(self.TD, self.fit_fitres[pos3,:], 'r', label='Fit')
            ax1.legend()
            ax1.set_xlim(self.TD[0], np.percentile(self.TD, 80))
            ax2.plot(self.TD, self.data_c[pos3,:], 'b.')
            ax2.plot(self.TD, self.fit_fitres[pos3,:], 'r')
            ax2.set_xscale('log')
            ax1.set_title(f"Fit at {self.WL[pos3]:.1f} nm")
            ax1.set_xlabel("Time / ps")
            ax1.set_ylabel("ΔA")
            plt.tight_layout()
            fname = os.path.join(outdir, f"Fit_{int(round(self.WL[pos3]))}nm.png")
            plt.savefig(fname, dpi=200)
            plt.show(block=True)
            # ask to continue
            resp = QMessageBox.question(self, "Continue", "Choose another wavelength?", QMessageBox.Yes | QMessageBox.No)
            if resp != QMessageBox.Yes:
                cont = False
    
        QMessageBox.information(self, "Plots saved", f"Plots saved to {outdir}")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainApp()
    
    window.show()

    sys.exit(app.exec_())