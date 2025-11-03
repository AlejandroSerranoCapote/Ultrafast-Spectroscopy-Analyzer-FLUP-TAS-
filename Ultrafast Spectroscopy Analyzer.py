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
from matplotlib import cm
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
    QHeaderView, QComboBox, QDoubleSpinBox, QFrame,QSpinBox,QDial,QSpacerItem, QSizePolicy
    ,QGroupBox, QHBoxLayout, QRadioButton
)
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QLineEdit, QLabel, QHBoxLayout
import fit
from core_analysis import fit_t0, load_data
from PyQt5.QtWidgets import QLineEdit, QLabel, QHBoxLayout
import time
from matplotlib.colors import BoundaryNorm

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
        btn_tas = QPushButton("TAS Analyzer")
        
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
        
         # Desactivar el TAS/FLUPS Analyzer (tachado y gris)
        # font_tas = QFont("Segoe UI", 12)
        # font_tas.setStrikeOut(True)
        # btn_flups.setFont(font_tas)
        # btn_flups.setEnabled(False)   
        # btn_flups.setStyleSheet("""
        #     QPushButton {
        #         background-color: #3a3a3a;
        #         color: gray;
        #         border-radius: 10px;
        #         border: 2px dashed #555;
        #         padding: 8px 16px;
        #     }
        # """)

        # --- Conexiones ---
        btn_flups.clicked.connect(self.launch_flups)
        btn_tas.clicked.connect(self.launch_tas)

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
        self.use_discrete_levels = True  # Cambia a False si prefieres mapa continuo

        # widgets
        self.btn_load = QPushButton("Load CSV")
        self.btn_load.clicked.connect(self.load_file)

        self.btn_plot = QPushButton("Show Map")
        self.btn_plot.clicked.connect(self.plot_map)
        self.btn_plot.setEnabled(False)
        self.btn_remove_fringe = QPushButton("Remove Pump Fringe")
        self.btn_remove_fringe.clicked.connect(self.remove_pump_fringe)
        self.btn_remove_fringe.setEnabled(True)
        # self.n_levels =   #  número inicial de niveles para el mapa discreto
        
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
        

        # define sliders sólo una vez (en __init__, elimina la otra ocurrencia)
        self.slider_min = QSlider(Qt.Horizontal)
        self.slider_max = QSlider(Qt.Horizontal)
        self.slider_min.setMinimum(0)
        self.slider_max.setMinimum(0)
        self.slider_min.valueChanged.connect(self.update_wl_range)
        self.slider_max.valueChanged.connect(self.update_wl_range)

        # controla el throttling del movimiento del ratón
        self._last_move_time = 0.0
        self._move_min_interval = 1.0 / 25.0  # como máximo ~25 FPS de actualización por movimiento
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

        
        
        layout = QVBoxLayout()
        layout.addLayout(top_layout)
        layout.addWidget(self.canvas)
        # layout.addLayout(slider_layout)
    
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


        # --- Layout horizontal principal ---
        main_layout = QHBoxLayout()
        
        # --- Layout vertical para sliders de Delay ---
        delay_layout = QVBoxLayout()
        delay_layout.setSpacing(5)
        
        # Delay min
        delay_layout.addWidget(QLabel("Delay min (ps):"))
        self.xmin_edit = QLineEdit("-1")
        self.xmin_edit.setFixedWidth(50)
        delay_layout.addWidget(self.xmin_edit)
        
        # Delay max
        delay_layout.addWidget(QLabel("Delay max (ps):"))
        self.xmax_edit = QLineEdit("3")
        self.xmax_edit.setFixedWidth(50)
        delay_layout.addWidget(self.xmax_edit)
        
        # Botón Apply X limits
        self.btn_apply_xlim = QPushButton("Apply X limits")
        self.btn_apply_xlim.setFixedWidth(120)
        self.btn_apply_xlim.clicked.connect(self.apply_x_limits)
        delay_layout.addWidget(self.btn_apply_xlim)
        
        # --- Layout vertical para sliders de λ ---
        wl_layout = QVBoxLayout()
        wl_layout.setSpacing(5)
        
        # --- λ min ---
        wl_min_layout = QHBoxLayout()
        wl_min_label = QLabel("λ min:")
        self.lbl_min_value = QLabel(str(400))  # valor inicial mostrado
        self.slider_min = QSlider(Qt.Horizontal)
        self.slider_min.setMinimum(400)
        self.slider_min.setMaximum(800)
        self.slider_min.setValue(500)
        self.slider_min.valueChanged.connect(self.update_wl_range)
        wl_min_layout.addWidget(wl_min_label)
        wl_min_layout.addWidget(self.slider_min)
        wl_min_layout.addWidget(self.lbl_min_value)
        wl_layout.addLayout(wl_min_layout)
        
        # --- λ max ---
        wl_max_layout = QHBoxLayout()
        wl_max_label = QLabel("λ max:")
        self.lbl_max_value = QLabel(str(800))  # valor inicial mostrado
        self.slider_max = QSlider(Qt.Horizontal)
        self.slider_max.setMinimum(400)
        self.slider_max.setMaximum(800)
        self.slider_max.setValue(700)
        self.slider_max.valueChanged.connect(self.update_wl_range)
        wl_max_layout.addWidget(wl_max_label)
        wl_max_layout.addWidget(self.slider_max)
        wl_max_layout.addWidget(self.lbl_max_value)
        wl_layout.addLayout(wl_max_layout)

        
        # --- Layout vertical para la rueda ---
        dial_layout = QVBoxLayout()
        self.n_levels = 5
        self.dial_levels = QDial()
        self.dial_levels.setRange(2, 100)
        self.dial_levels.setValue(self.n_levels)
        self.dial_levels.setNotchesVisible(True)
        self.dial_levels.setWrapping(False)
        self.dial_levels.setFixedSize(80, 80)
        self.dial_levels.valueChanged.connect(self.update_n_levels)
        self.lbl_dial = QLabel(f"{self.n_levels}")
        self.lbl_dial.setAlignment(Qt.AlignCenter)
        dial_layout.addWidget(self.dial_levels, alignment=Qt.AlignCenter)
        dial_layout.addWidget(self.lbl_dial, alignment=Qt.AlignCenter)
        
        # --- Combinar en layout horizontal principal ---
        main_layout.addLayout(delay_layout)
        main_layout.addSpacing(10)          # un pequeño espacio (en lugar de un Spacer grande)
        main_layout.addLayout(wl_layout)
        main_layout.addSpacing(10)
        main_layout.addLayout(dial_layout)
        
        # ajustar márgenes generales del layout principal
        main_layout.setContentsMargins(5, 0, 5, 0)
        main_layout.setSpacing(15)  # espacio entre columnas principales
                
        # Contenedor final
        range_container = QWidget()
        range_container.setLayout(main_layout)
        range_container.setMaximumWidth(800)  # opcional, controla el ancho total
        layout.addWidget(range_container)

        fit_group = QGroupBox("Modelo de ajuste t₀")
        fit_layout = QHBoxLayout()
        
        self.radio_poly = QRadioButton("Polinómico")
        self.radio_nonlinear = QRadioButton("No lineal")
        self.radio_nonlinear.setChecked(True)  # valor por defecto
        
        fit_layout.addWidget(self.radio_poly)
        fit_layout.addWidget(self.radio_nonlinear)
        fit_group.setLayout(fit_layout)
        
        # Añadir este grupo al layout principal (si usas un QVBoxLayout central)
        main_layout.addWidget(fit_group)
                
        # --- Botón Switch dinámico (cambia entre FLUPS y TAS) ---
        self.btn_switch = QPushButton("Switch to TAS")
        self.btn_switch.setFixedWidth(160)
        self.btn_switch.setCursor(Qt.PointingHandCursor)
        self.btn_switch.setStyleSheet("""
            QPushButton {
                background-color: #8E24AA;
                color: white;
                font-weight: bold;
                padding: 6px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #AB47BC;
            }
        """)
        self.btn_switch.clicked.connect(self.switch_analyzer)

        # Layout inferior alineado a la derecha (una sola vez)
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.btn_switch)
        layout.addLayout(bottom_layout)
 
        # Añadir el botón al layout inferior (alineado a la derecha)
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.btn_switch)
        layout.addLayout(bottom_layout)
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
    def switch_analyzer(self):
        """Cambia entre FLUPSAnalyzer y TASAnalyzer sin cerrar la nueva ventana."""
        try:
            target_cls_name = "FLUPSAnalyzer" if isinstance(self, TASAnalyzer) else "TASAnalyzer"
            
            if target_cls_name in globals() and callable(globals()[target_cls_name]):
                TargetCls = globals()[target_cls_name]
            else:
                raise NameError(f"{target_cls_name} not found")

            # Guarda la referencia en self (no variable local)
            self.new_window = TargetCls()
            self.new_window.show()

            # Cierra la ventana actual
            self.close()

        except Exception as e:
            QMessageBox.critical(self, "Switch error", f"Cannot switch analyzer:\n{e}")




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

            #  Guarda también la ruta y el directorio base del CSV
            self.csv_path = file_path
            self.base_dir = os.path.dirname(file_path)
            
            self.label_status.setText(f"Loaded : {os.path.basename(file_path)}")
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
            self.update_wl_range()
        except Exception as e:
            QMessageBox.critical(self, "Error loading file", str(e))
    def apply_wl_range(self):
        min_val = self.slider_min.value()
        max_val = self.slider_max.value()
        print(f"Aplicando λ min={min_val}, λ max={max_val}")
        # Aquí actualiza tu mapa o cálculos

    def _plot_discrete_map(self, ax, WL, TD, data, n_levels=5, cmap='jet', shading='auto', vmin=None, vmax=None):
        """Dibuja mapa tipo contourf con pcolormesh discreto."""

        # Forzar límites si se pasan, sino usar datos
        if vmin is None:
            vmin = np.nanmin(data)
        if vmax is None:
            vmax = np.nanmax(data)
    
        levels = np.linspace(vmin, vmax, n_levels)
        norm = BoundaryNorm(levels, ncolors=plt.get_cmap(cmap).N, clip=True)
    
        pcm = ax.pcolormesh(WL, TD, data.T, shading=shading, cmap=cmap, norm=norm)
        return pcm
    def update_n_levels(self, value):
        """Actualiza el número de niveles del mapa discreto y redibuja el mapa, respetando el rango visible."""
        self.n_levels = value
        self.lbl_dial.setText(f"{value} levels")  #  actualiza texto
    
        if self.data is None:
            return
    
        #  Determinar qué datos y WL usar (respetando el rango visible actual)
        if hasattr(self, "WL_visible") and self.WL_visible is not None:
            WL_used = self.WL_visible
            if getattr(self, "showing_corrected", False):
                # si estamos mostrando el mapa corregido
                wl_min = self.WL_visible[0]
                wl_max = self.WL_visible[-1]
                wl_min_idx = np.argmin(np.abs(self.WL - wl_min))
                wl_max_idx = np.argmin(np.abs(self.WL - wl_max)) + 1
                data_used = self.data_corrected[wl_min_idx:wl_max_idx, :]
            else:
                data_used = self.data_visible
        else:
            WL_used = self.WL
            data_used = self.data_corrected if getattr(self, "showing_corrected", False) else self.data
    
        #  Redibujar mapa directamente (sin resetear)
        self.ax_map.clear()
        if self.cbar:
            try: self.cbar.remove()
            except: pass
            self.cbar = None
    
        self.pcm = self._plot_discrete_map(
            self.ax_map,
            WL_used,
            self.TD,
            data_used,
            n_levels=self.n_levels,
            shading="auto",
            vmin=-1,
            vmax=1
        )
    
        self.ax_map.set_xlabel("Wavelength (nm)")
        self.ax_map.set_ylabel("Delay (ps)")
        self.ax_map.set_title("ΔA Map")
        self.ax_map.set_yscale("symlog")
    
        #  Colorbar
        divider = make_axes_locatable(self.ax_map)
        cax = divider.append_axes("right", size="3%", pad=0.02)
        self.cbar = self.figure.colorbar(self.pcm, cax=cax, label="ΔA")
    
        #  Estilo visual coherente
        self.ax_map.set_facecolor("white")
        for spine in self.ax_map.spines.values():
            spine.set_color("black")
        self.ax_map.tick_params(colors="black")
        self.ax_map.xaxis.label.set_color("black")
        self.ax_map.yaxis.label.set_color("black")
        self.ax_map.title.set_color("black")
    
        self.canvas.draw_idle()


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
        if self.use_discrete_levels:
            self.pcm = self._plot_discrete_map(self.ax_map, self.WL, self.TD, self.data,n_levels=self.n_levels)
        else:
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
        """Actualizar el mapa y subplots pequeños según el rango de λ de los sliders.
           Además actualiza las etiquetas que muestran las longitudes de onda (nm)."""
        # si no hay datos aún, solo actualiza etiquetas si es posible (evita crash)
        if getattr(self, "WL", None) is None or getattr(self, "data", None) is None:
            # Si las etiquetas existen pero no hay WL definidos, poner guiones
            if hasattr(self, "lbl_min_value"):
                self.lbl_min_value.setText("- nm")
            if hasattr(self, "lbl_max_value"):
                self.lbl_max_value.setText("- nm")
            return
    
        # sliders devuelven índices de WL (según cómo los configuras en load_file)
        wl_min_idx = int(self.slider_min.value())
        wl_max_idx = int(self.slider_max.value())
    
        # proteger orden y límites
        if wl_min_idx >= wl_max_idx:
            wl_max_idx = wl_min_idx + 1
    
        wl_min_idx = max(0, min(wl_min_idx, len(self.WL) - 1))
        # queremos usar wl_max_idx como índice inclusivo -> limitar a len(WL)-1
        wl_max_idx = max(0, min(wl_max_idx, len(self.WL) - 1))
    
        # actualizar etiquetas (valores reales en nm)
        try:
            if hasattr(self, "lbl_min_value"):
                self.lbl_min_value.setText(f"{self.WL[wl_min_idx]:.1f} nm")
            if hasattr(self, "lbl_max_value"):
                self.lbl_max_value.setText(f"{self.WL[wl_max_idx]:.1f} nm")
        except Exception:
            # fallback: mostrar índices si algo raro pasa con self.WL
            if hasattr(self, "lbl_min_value"):
                self.lbl_min_value.setText(f"{wl_min_idx}")
            if hasattr(self, "lbl_max_value"):
                self.lbl_max_value.setText(f"{wl_max_idx}")
    
        # ahora construimos la selección (nota: slice end no inclusivo por Python)
        wl_slice_end = wl_max_idx + 1
        WL_sel = self.WL[wl_min_idx:wl_slice_end]
        data_sel = self.data[wl_min_idx:wl_slice_end, :]  # (N_wl_selected, N_td)
        # Guardar los datos visibles actuales para otras funciones (como on_move_map)
        self.WL_visible = WL_sel
        self.data_visible = data_sel

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
        if self.use_discrete_levels:
            self.pcm = self._plot_discrete_map(
                self.ax_map,
                WL_sel,
                self.TD,
                data_sel,
                n_levels=self.n_levels,
                shading="auto"
            )
        else:
            self.pcm = self.ax_map.pcolormesh(
                WL_sel,
                self.TD,
                data_sel.T,
                shading="auto",
                cmap="jet"
            )
    
        self.ax_map.set_xlabel("Wavelength (nm)")
        self.ax_map.set_ylabel("Delay (ps)")
        self.ax_map.set_title("ΔA Map")
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
            x0 = float(np.median(WL_sel))
        else:
            x0 = float(self.WL[0])
        y0 = float(np.median(self.TD))
    
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
    
        #  Prioridad: datos explícitos → rango visible guardado → todos los datos
        if WL_sel is not None and data_sel is not None:
            WL_vis = WL_sel
            data_vis = data_sel
        elif hasattr(self, "WL_visible") and self.WL_visible is not None:
            WL_vis = self.WL_visible
            data_vis = self.data_visible
        else:
            WL_vis = self.WL
            data_vis = self.data
    
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
        self.update_small_cuts(
            x, y,
            WL_sel=getattr(self, "WL_visible", None),
            data_sel=getattr(self, "data_visible", None)
        )
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

        # Determinar qué modelo usar según los radio buttons
        if self.radio_poly.isChecked():
            mode = 'poly'
        elif self.radio_nonlinear.isChecked():
            mode = 'nonlinear'
        else:
            mode = 'auto'
        
        # Intentar el ajuste
        try:
            result = fit_t0(w_points, t0_points, self.WL, self.TD, self.data, mode=mode)
        except Exception as e:
            QMessageBox.critical(self, "Error de ajuste t₀", str(e))
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

        base_dir = os.path.dirname(self.file_path)
        base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        self.save_dir = os.path.join(base_dir, f"{base_name}_Results")  # 🔹 guardamos como atributo
        os.makedirs(self.save_dir, exist_ok=True)
        
        data_corr = result['corrected']
        WL = self.WL
        TD = self.TD

        np.save(os.path.join(self.save_dir, f"{base_name}_treated_data.npy"),
                {'data_c': data_corr, 'WL': WL, 'TD': TD})

        np.savetxt(os.path.join(self.save_dir, f"{base_name}_WL.txt"), WL,
                   fmt='%.6f', header='Wavelength (nm)', comments='')
        np.savetxt(os.path.join(self.save_dir, f"{base_name}_TD.txt"), TD,
                   fmt='%.6f', header='Delay (ps)', comments='')

        with open(os.path.join(self.save_dir, f"{base_name}_kin.txt"), 'w') as f:
            f.write("\t".join([f"{base_name}_kin_{round(wl,1)}nm" for wl in WL]) + "\n")
            np.savetxt(f, data_corr.T, fmt='%.6e', delimiter='\t')

        with open(os.path.join(self.save_dir, f"{base_name}_spec.txt"), 'w') as f:
            f.write("\t".join([f"{base_name}_spec_{td:.2f}ps" for td in TD]) + "\n")
            np.savetxt(f, data_corr, fmt='%.6e', delimiter='\t')

        t0_lambda = result['t0_lambda']
        popt = result['popt']
        method = result['method']

        t0_file = os.path.join(self.save_dir, f"{base_name}_t0_fit.txt")
        np.savetxt(t0_file, np.column_stack((WL, t0_lambda)),
                   fmt='%.6f', header='Wavelength (nm)\t t0 (ps)', comments='')

        params_file = os.path.join(self.save_dir, f"{base_name}_fit_params.txt")
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
                                f"Results saved in:\n{self.save_dir}")
        QMessageBox.information(self, "t₀ Fit Result",
                                f"Fit completed using {method} model.\nParameters: {np.round(popt,4)}")


    def toggle_corrected_map(self):
        """Alterna entre mapa original y corregido dentro de la misma ventana,
           mostrando el mapa limpio y manteniendo crucetas."""
    
        if self.data_corrected is None:
            QMessageBox.warning(self, "No corrected data", "Run 'Fit t₀' first to generate corrected data.")
            return
    
        # ==============================================================
        # DECIDIR QUÉ MAPA MOSTRAR (RESPETANDO RANGO DE SLIDERS)
        # ==============================================================
    
        if getattr(self, "showing_corrected", False):
            # Mostrar mapa original
            if hasattr(self, "WL_visible") and self.WL_visible is not None:
                wl_min = self.WL_visible[0]
                wl_max = self.WL_visible[-1]
                wl_min_idx = np.argmin(np.abs(self.WL - wl_min))
                wl_max_idx = np.argmin(np.abs(self.WL - wl_max)) + 1
                WL_used = self.WL_visible
                data_to_plot = self.data[wl_min_idx:wl_max_idx, :]
            else:
                WL_used = self.WL
                data_to_plot = self.data
    
            self.showing_corrected = False
            self.btn_show_corr.setText("Show Corrected Map")
    
        else:
            # Mostrar mapa corregido
            if hasattr(self, "WL_visible") and self.WL_visible is not None:
                wl_min = self.WL_visible[0]
                wl_max = self.WL_visible[-1]
                wl_min_idx = np.argmin(np.abs(self.WL - wl_min))
                wl_max_idx = np.argmin(np.abs(self.WL - wl_max)) + 1
                WL_used = self.WL_visible
                data_to_plot = np.copy(self.data_corrected[wl_min_idx:wl_max_idx, :])
            else:
                WL_used = self.WL
                data_to_plot = np.copy(self.data_corrected)
    
            self.showing_corrected = True
            self.btn_show_corr.setText("Show Original Map")
    
        #  Guardar los visibles actuales (para on_move_map, etc.)
        self.WL_visible = WL_used
        self.data_visible = data_to_plot
    
        # ==============================================================
        # LIMPIAR MAPA Y COLORBAR
        # ==============================================================
    
        self.ax_map.clear()
        if self.cbar:
            try:
                self.cbar.remove()
            except Exception:
                pass
            self.cbar = None
    
        # Borrar puntos seleccionados y línea de fit
        for p in getattr(self, "clicked_points", []):
            try:
                p['artist'].remove()
            except Exception:
                pass
        self.clicked_points = []
    
        if self.fit_line_artist is not None:
            try:
                self.fit_line_artist.remove()
            except Exception:
                pass
        self.fit_line_artist = None
    
        # ==============================================================
        # ESCALAS DE COLOR
        # ==============================================================
    
        if getattr(self, "is_TAS_mode", False):
            finite_vals = data_to_plot[np.isfinite(data_to_plot)]
            if finite_vals.size == 0:
                vmin, vmax = -1, 1
            else:
                vmin, vmax = np.percentile(finite_vals, [1, 99])
        else:
            vmin, vmax = -1, 1
    
        # ==============================================================
        # DIBUJAR NUEVO MAPA (USANDO WL_visible)
        # ==============================================================
    
        if self.use_discrete_levels:
            self.pcm = self._plot_discrete_map(
                self.ax_map,
                self.WL_visible,
                self.TD,
                self.data_visible,
                n_levels=self.n_levels,
                shading="auto",
                vmin=vmin,
                vmax=vmax
            )
        else:
            self.pcm = self.ax_map.pcolormesh(
                self.WL_visible,
                self.TD,
                self.data_visible.T,
                shading="auto",
                cmap="jet",
                vmin=vmin,
                vmax=vmax
            )
    
        self.ax_map.set_xlabel("Wavelength (nm)")
        self.ax_map.set_ylabel("Delay (ps)")
        self.ax_map.set_title("ΔA Map (FLUPS)")
        self.ax_map.set_yscale("symlog")
    
        # ==============================================================
        # COLORBAR Y ESTILO
        # ==============================================================
    
        divider = make_axes_locatable(self.ax_map)
        cax = divider.append_axes("right", size="3%", pad=0.02)
        self.cbar = self.figure.colorbar(self.pcm, cax=cax, label="ΔA")
        self.cbar.ax.yaxis.set_tick_params(color="black", labelcolor="black")
        self.cbar.ax.yaxis.label.set_color("black")
        for spine in self.cbar.ax.spines.values():
            spine.set_color("black")
    
        self.ax_map.set_facecolor("white")
        self.ax_map.tick_params(colors="black")
        self.ax_map.xaxis.label.set_color("black")
        self.ax_map.yaxis.label.set_color("black")
        self.ax_map.title.set_color("black")
        for spine in self.ax_map.spines.values():
            spine.set_color("black")
    
        # ==============================================================
        # CRUCETAS Y EVENTOS
        # ==============================================================
    
        if self.vline_map is not None:
            x0 = self.vline_map.get_xdata()[0]
        else:
            x0 = self.WL_visible[0]
    
        if self.hline_map is not None:
            y0 = self.hline_map.get_ydata()[0]
        else:
            y0 = self.TD[0]
    
        self.vline_map = self.ax_map.axvline(x0, color='k', ls='--', lw=1)
        self.hline_map = self.ax_map.axhline(y0, color='k', ls='--', lw=1)
        self.marker_map, = self.ax_map.plot([x0], [y0], 'wx', markersize=8, markeredgewidth=2)
    
        # limpiar subplots pequeños
        self.cut_time_small.set_data([], [])
        self.cut_spec_small.set_data([], [])
        self.ax_time_small.relim(); self.ax_time_small.autoscale_view()
        self.ax_spec_small.relim(); self.ax_spec_small.autoscale_view()
    
        # funciones internas de actualización
        def update_small_cuts(x, y):
            if x is None or y is None:
                return
            idx_wl = int(np.argmin(np.abs(self.WL_visible - x)))
            idx_td = int(np.argmin(np.abs(self.TD - y)))
            y_time = self.data_visible[idx_wl, :].ravel()
            self.cut_time_small.set_data(self.TD, y_time)
            self.ax_time_small.relim(); self.ax_time_small.autoscale_view()
            self.ax_time_small.set_title(f"Kinetics at {self.WL_visible[idx_wl]:.1f} nm")
            y_spec = self.data_visible[:, idx_td].ravel()
            self.cut_spec_small.set_data(self.WL_visible, y_spec)
            self.ax_spec_small.relim(); self.ax_spec_small.autoscale_view()
            self.ax_spec_small.set_title(f"Spectra at {self.TD[idx_td]:.2f} ps")
    
        def onclick(event):
            if event.inaxes != self.ax_map:
                return
            x, y = event.xdata, event.ydata
            if x is None or y is None:
                return
            self.vline_map.set_xdata([x, x])
            self.hline_map.set_ydata([y, y])
            self.marker_map.set_data([x], [y])
            update_small_cuts(x, y)
            self.canvas.draw_idle()
    
        def onmove(event):
            if event.inaxes != self.ax_map:
                return
            x, y = event.xdata, event.ydata
            if x is None or y is None:
                return
            self.vline_map.set_xdata([x, x])
            self.hline_map.set_ydata([y, y])
            self.marker_map.set_data([x], [y])
            update_small_cuts(x, y)
            self.canvas.draw_idle()
    
        # desconectar eventos previos
        if hasattr(self, "cid_corr_click") and self.cid_corr_click is not None:
            self.canvas.mpl_disconnect(self.cid_corr_click)
        if hasattr(self, "cid_corr_move") and self.cid_corr_move is not None:
            self.canvas.mpl_disconnect(self.cid_corr_move)
    
        # conectar nuevos
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
        self.use_discrete_levels = False  # Cambia a False si prefieres mapa continuo
        self.dial_levels.hide()
        self.lbl_dial.hide()

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
 
 
        # --- Modificar el botón switch (ya existe del padre) ---
        self.btn_switch.setText("Switch to FLUPS")
        # Añadir sliders adicionales al layout principal
        self.centralWidget().layout().addLayout(slider_layout_extra)
        # Desconectamos el slot anterior para evitar doble conexión
        try:
            self.btn_switch.clicked.disconnect()
        except TypeError:
            pass
        
        # Conectamos al nuevo método
        self.btn_switch.clicked.connect(self.switch_analyzer)

    def switch_analyzer(self):
        """Cambia entre FLUPSAnalyzer y TASAnalyzer sin cerrar la nueva ventana."""
        try:
            target_cls_name = "FLUPSAnalyzer" if isinstance(self, TASAnalyzer) else "TASAnalyzer"
            
            if target_cls_name in globals() and callable(globals()[target_cls_name]):
                TargetCls = globals()[target_cls_name]
            else:
                raise NameError(f"{target_cls_name} not found")

            # Guarda la referencia en self (no variable local)
            self.new_window = TargetCls()
            self.new_window.show()

            # Cierra la ventana actual
            self.close()

        except Exception as e:
            QMessageBox.critical(self, "Switch error", f"Cannot switch analyzer:\n{e}")


    def get_base_dir(self):
        """
        Devuelve la carpeta donde se encuentra el CSV de medida.
        Crea subcarpetas 'fit' y 'plots' si no existen.
        """

        if hasattr(self, 'file_path') and self.file_path:
            base_dir = os.path.dirname(self.file_path)
        else:
            base_dir = os.getcwd()
    
        fit_dir = os.path.join(base_dir, "fit")
        plots_dir = os.path.join(base_dir, "plots")
        os.makedirs(fit_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
    
        return base_dir, fit_dir, plots_dir
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
        
        #  Guardar ruta del primer CSV leído
        self.file_path = file_path_medida
        
        #  Crear carpeta específica para esta medición
        base_dir = os.path.dirname(self.file_path)
        base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        self.results_dir = os.path.join(base_dir, f"{base_name}_results")
        os.makedirs(self.results_dir, exist_ok=True)
        
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
            os.path.dirname(self.file_path),  # ✅ Abre el diálogo en la misma carpeta
            "CSV Files (*.csv);;Data Files (*.txt *.dat)"
        )
        if not file_path_solvente or not os.path.exists(file_path_solvente):
            self.label_status.setText(" No solvent file selected.")
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
        self.label_status.setText(" TAS data loaded")
        self.update_am_sf()
        self.plot_map()
        
        # --- Definir ruta base para compatibilidad con FLUPSAnalyzer ---
        self.file_path = file_path_medida
        
        # --- Asegurar que los botones t₀ funcionan ---
        if hasattr(self, "btn_plot"):
            self.btn_plot.setEnabled(True)
        if hasattr(self, "btn_select"):
            self.btn_select.setEnabled(True)
        if hasattr(self, "btn_fit"):
            self.btn_fit.setEnabled(True)
        
        #  Mostrar solo el nombre del archivo cargado
        file_name = os.path.basename(file_path_medida)
        self.label_status.setText(f"TAS data loaded from: {file_name}")


    def fit_t0_points(self):
        if not getattr(self, "clicked_points", None) or len(self.clicked_points) < 2:
            QMessageBox.warning(self, "Not enough points", "Select at least 2 points on the map.")
            return
    
        w_points = np.array([p['x'] for p in self.clicked_points])
        t0_points = np.array([p['y'] for p in self.clicked_points])
    
        try:
            # recalcular base antes del fit
            self.update_am_sf()
            result = fit_t0(w_points, t0_points, self.WL, self.TD, self.data)
        except Exception as e:
            QMessageBox.critical(self, "Fit error", str(e))
            return
    
        # --- Guardar datos corregidos globalmente ---
        self.result_fit = result
        self.data_corrected = result['corrected']
        self.data = np.copy(self.data_corrected)
        self.plot_map(show_fit=True)
        self.btn_show_corr.setEnabled(True)
    
        # --- Crear carpeta de resultados junto al CSV ---
        base_dir = os.path.dirname(self.file_path)
        base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        save_dir = os.path.join(base_dir, f"{base_name}_results")
        os.makedirs(save_dir, exist_ok=True)
    
        # --- Guardar matrices y resultados ---
        data_corr = np.copy(self.data_corrected)
        WL = self.WL
        TD = self.TD
        t0_lambda = result['t0_lambda']
        popt = result['popt']
        method = result['method']
    
        #  Guardar archivos principales
        np.save(os.path.join(save_dir, f"{base_name}_treated_data.npy"),
                {'data_c': data_corr, 'WL': WL, 'TD': TD})
    
        np.savetxt(os.path.join(save_dir, f"{base_name}_WL.txt"), WL,
                   fmt='%.6f', header='Wavelength (nm)', comments='')
        np.savetxt(os.path.join(save_dir, f"{base_name}_TD.txt"), TD,
                   fmt='%.6f', header='Delay (ps)', comments='')
    
        np.savetxt(os.path.join(save_dir, f"{base_name}_kin.txt"),
                   data_corr.T, fmt='%.6e', delimiter='\t')
        np.savetxt(os.path.join(save_dir, f"{base_name}_spec.txt"),
                   data_corr, fmt='%.6e', delimiter='\t')
    
        np.savetxt(os.path.join(save_dir, f"{base_name}_t0_fit.txt"),
                   np.column_stack((WL, t0_lambda)),
                   fmt='%.6f', header='Wavelength (nm)\t t0 (ps)', comments='')
    
        with open(os.path.join(save_dir, f"{base_name}_fit_params.txt"), 'w') as f:
            f.write(f"Fit method: {method}\n")
            f.write("Fit parameters:\n")
            if method.startswith('poly'):
                names = ['c4', 'c3', 'c2', 'c1', 'c0']
            else:
                names = ['a', 'b', 'c', 'd']
            for name, val in zip(names, popt):
                f.write(f"  {name} = {val:.6g}\n")
    
        QMessageBox.information(self, "Files saved",
                                f" Results saved in:\n{save_dir}")
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
    
        # Base: medida - solvente
        base_data = self.medida - solvente_interp
    
        # Aplicar máscara si existe
        if self.pump_mask is not None:
            base_data[self.pump_mask] = 1e-10
    
        # Si existe data corregida (fit t₀), usarla
        if hasattr(self, "data_corrected") and self.data_corrected is not None:
            self.data = np.copy(self.data_corrected)
            # Reaplicar solvente y máscara encima de la corrección
            self.data -= (self.medida - base_data)  # sustrae solo la parte del solvente
            if self.pump_mask is not None:
                self.data[self.pump_mask] = 1e-10
        else:
            self.data = base_data
    
        self.update_wl_range()
    
        if hasattr(self, "global_fit_panel") and self.global_fit_panel is not None:
            self.global_fit_panel.update_from_parent()
    
        self._updating_am_sf = False

    # ------------------------------------------------------------------
    # DIBUJAR MAPA ΔA
    # ------------------------------------------------------------------
    def plot_map(self, show_fit=False):
        """Dibuja el mapa principal TAS con crucetas y fit persistentes."""
        if self.data is None:
            return
    
        # --- Guardar estado previo (crucetas y fit) ---
        x_cross, y_cross = None, None
        if hasattr(self, "vline_map") and self.vline_map is not None:
            x_cross = self.vline_map.get_xdata()[0]
        if hasattr(self, "hline_map") and self.hline_map is not None:
            y_cross = self.hline_map.get_ydata()[0]
    
        # --- Limpiar eje principal ---
        self.ax_map.clear()
        if self.cbar:
            try: self.cbar.remove()
            except Exception: pass
            self.cbar = None
    
        # Escala real con margen
        vmin = np.nanmin(self.data)
        vmax = np.nanmax(self.data)
        margin = 0.05 * (vmax - vmin)
        vmin -= margin
        vmax += margin
    
        # Dibujar mapa
        self.pcm = self.ax_map.pcolormesh(
            self.WL, self.TD, self.data.T,
            shading="auto", cmap="jet",
            vmin=vmin, vmax=vmax
        )
    
        self.ax_map.set_xlabel("Wavelength (nm)")
        self.ax_map.set_ylabel("Delay (ps)")
        self.ax_map.set_title("ΔA Map (TAS)")
        self.ax_map.set_yscale("symlog")
        self.ax_map.set_ylim(-1, 10)
        # Colorbar
        divider = make_axes_locatable(self.ax_map)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        self.cbar = self.figure.colorbar(self.pcm, cax=cax, label="ΔA")
    
        # --- Restaurar crucetas ---
        if x_cross is None:
            x_cross = np.median(self.WL)
        if y_cross is None:
            y_cross = np.median(self.TD)
    
        self.vline_map = self.ax_map.axvline(x_cross, color="k", ls="--", lw=1, zorder=6)
        self.hline_map = self.ax_map.axhline(y_cross, color="k", ls="--", lw=1, zorder=6)
        self.marker_map, = self.ax_map.plot([x_cross], [y_cross], "wx",
                                            markersize=8, markeredgewidth=2, zorder=7)
    
        if hasattr(self, "result_fit") and self.result_fit is not None:
            fit_x = self.result_fit.get("fit_x", None)
            fit_y = self.result_fit.get("fit_y", None)
            if fit_x is not None and fit_y is not None:
                # --- Dibujar el fit temporalmente en escala lineal ---
                prev_scale = self.ax_map.get_yscale()
                self.ax_map.set_yscale("linear")
        
                self.fit_line_artist, = self.ax_map.plot(
                    fit_x, fit_y, "r-", lw=2, label="t₀ fit", zorder=10
                )
        
                # --- Restaurar la escala original (symlog) ---
                self.ax_map.set_yscale(prev_scale)
                self.ax_map.legend()

        # Actualizar subplots pequeños
        self.update_small_cuts(x_cross, y_cross)
    
        # Reconectar eventos
        if self.cid_click is None:
            self.cid_click = self.canvas.mpl_connect("button_press_event", self.on_click_map)
        if self.cid_move is None:
            self.cid_move = self.canvas.mpl_connect("motion_notify_event", self.on_move_map)
    
        self.canvas.draw_idle()

    def update_small_cuts(self, x, y, WL_sel=None, data_sel=None):
        """Actualiza los subplots de cinética y espectro para la posición (x, y)."""
        if x is None or y is None or self.data is None:
            return
    
        # --- Subconjunto según sliders ---
        if WL_sel is None or data_sel is None:
            wl_min_idx = self.slider_min.value()
            wl_max_idx = self.slider_max.value()
            wl_max_idx = min(wl_max_idx + 1, len(self.WL))
            WL_sel = self.WL[wl_min_idx:wl_max_idx]
            data_sel = self.data[wl_min_idx:wl_max_idx, :]
    
        if WL_sel.size == 0:
            return
    
        idx_wl = np.argmin(np.abs(WL_sel - x))
        idx_td = np.argmin(np.abs(self.TD - y))
    
        # --- Cinética ---
        y_time = data_sel[idx_wl, :]
        self.cut_time_small.set_data(self.TD, y_time)
        self.ax_time_small.relim()
        self.ax_time_small.autoscale_view()
        self.ax_time_small.set_title(f"Kinetics at {WL_sel[idx_wl]:.1f} nm")
    
        #  Dibujar o actualizar la línea de tiempo
        if not hasattr(self, "vline_time_small") or self.vline_time_small is None:
            self.vline_time_small = self.ax_time_small.axvline(
                x=y, color='k', ls='--', lw=1.2, zorder=5)
        else:
            self.vline_time_small.set_xdata([y, y])
            self.vline_time_small.set_visible(True)
    
        # --- Espectro ---
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
        


class GlobalFitPanel(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Global Fit")
        self.resize(1000, 700)

        # Data placeholders: prefer parent's data if present
        self.parent_app = parent
        self.data_c = None   # shape: (numWL, numTD)
        self.TD = None
        self.WL = None
        self.base_dir = None
        # --- Determinar carpeta base de guardado ---
        if hasattr(parent, "save_dir") and parent.save_dir:
            #  Usa la carpeta de resultados del programa principal
            self.base_dir = parent.save_dir
        elif hasattr(parent, "file_path") and parent.file_path:
            #  Si no hay save_dir pero sí un archivo cargado
            base_name = os.path.splitext(os.path.basename(parent.file_path))[0]
            self.base_dir = os.path.join(os.path.dirname(parent.file_path), f"{base_name}_Results")
            os.makedirs(self.base_dir, exist_ok=True)
        else:
            # ⚠️ Fallback: si no hay nada, usa el directorio actual
            self.base_dir = os.getcwd()
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
    
        # --- Detectar técnica ---
        tech = "TAS" if getattr(self.parent_app, "is_TAS_mode", False) else "FLUPS"
    
        # --- Ajustar escala según técnica ---
        if tech == "FLUPS":
            vmin, vmax = -1, 1
        else:  # TAS u otra
            vmin = np.nanmin(self.data_c)
            vmax = np.nanmax(self.data_c)
            # margen pequeño para que no se corte
            margin = 0.05 * (vmax - vmin)
            vmin -= margin
            vmax += margin
    
        # --- Dibujar mapa ---
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
        self.ax_exp.set_title(f"Experimental ΔA ({tech})")
    
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
        with optional callback for progress update (compatible with older SciPy)."""
        import inspect
        numWL = len(self.WL)
        TD = self.TD
        data_flat = self.data_c.T.flatten()
    
        def residuals(x):
            F = fit.eval_global_model(x, TD, self.numExp, numWL, self.t0_choice)
            return F.flatten() - data_flat
    
        # --- progress bookkeeping ---
        self.progress_bar.setValue(0)
        iterations = {'count': 0}
    
        def callback(xk, *args, **kwargs):
            iterations['count'] += 1
            val = (iterations['count'] % 100)
            self.progress_bar.setValue(val)
            QTimer.singleShot(1, lambda: None)  # keeps UI responsive
    
        try:
            # --- detect if least_squares supports the 'callback' argument ---
            lsq_signature = inspect.signature(least_squares)
            kwargs = dict(
                fun=residuals,
                x0=self.ini,
                bounds=(self.limi, self.lims),
                jac='2-point',
            )
    
            if "callback" in lsq_signature.parameters:
                kwargs["callback"] = callback  # ✅ use it if available
    
            res = least_squares(**kwargs)
    
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
    
        # --- Usar la carpeta base determinada al inicio ---
        base_dir = self.base_dir
        print(f" Guardando resultados en: {base_dir}")
        
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
    
        # ---  Usar directamente la carpeta _Results creada por FLUPSAnalyzer ---
        base_dir = self.base_dir
        print(f" Guardando resultados en: {base_dir}")

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
            
            # --- Guardar figura ---
            fname_img = os.path.join(outdir, f"Fit_{int(round(self.WL[pos3]))}nm.png")
            plt.savefig(fname_img, dpi=200)
            
            # --- Guardar datos numéricos (Delay, Exp, Fit) ---
            fname_txt = os.path.join(outdir, f"Fit_{int(round(self.WL[pos3]))}nm.txt")
            with open(fname_txt, "w") as ftxt:
                ftxt.write("# TD(ps)\tExp(A)\tFit(A)\n")
                for td, exp, fitv in zip(self.TD, self.data_c[pos3, :], self.fit_fitres[pos3, :]):
                    ftxt.write(f"{td:.6e}\t{exp:.6e}\t{fitv:.6e}\n")
            
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