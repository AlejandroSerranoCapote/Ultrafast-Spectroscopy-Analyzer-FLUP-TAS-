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
    ,QGroupBox, QHBoxLayout, QRadioButton,QCheckBox,QFormLayout
)
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QLineEdit, QLabel, QHBoxLayout
import fit
from core_analysis import fit_t0, load_data,eV_a_nm
from PyQt5.QtWidgets import QLineEdit, QLabel, QHBoxLayout
import time
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.colors import BoundaryNorm

class MainApp(QMainWindow):
    '''
    VENTANA PRINCIPAL (FLUPS/TAS)
    '''
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Analyzer Selector")
        self.setMinimumSize(500, 300)

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
        btn_global_fit = QPushButton("Global fit")
        
        for btn in [btn_flups, btn_tas,btn_global_fit]:
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
        
        # --- Conexiones ---
        btn_flups.clicked.connect(self.launch_flups)
        btn_tas.clicked.connect(self.launch_tas)
        btn_global_fit.clicked.connect(self.launch_global)
        
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

    #Lanzar el global fit
    def launch_global(self):
        self.analyzer = GlobalFitPanel()
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
        self.use_discrete_levels = True  # Cambia a False mapa continuo
        
        self.bg_cache = None
        self.cid_draw = None 
        self._is_drawing = False
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
        

        self.slider_min = QSlider(Qt.Horizontal)
        self.slider_max = QSlider(Qt.Horizontal)
        self.slider_min.setMinimum(0)
        self.slider_max.setMinimum(0)
        self.slider_min.valueChanged.connect(self.update_wl_range)
        self.slider_max.valueChanged.connect(self.update_wl_range)

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
        self.cid_draw = self.canvas.mpl_connect('draw_event', self.on_draw)
        
        self.clicked_points = []   
        self.cid_click = None     
        self.cid_move = None  
    
        self.cid_move = self.canvas.mpl_connect("motion_notify_event", self.on_move_map)
            
        # elementos interactivos
        self.pcm = None
        self.cbar = None
        self.marker_map = None
        self.vline_map = None
        self.hline_map = None
        self.fit_line_artist = None
    
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
        
        wl_layout = QVBoxLayout()
        wl_layout.setSpacing(5)
        

        wl_min_layout = QHBoxLayout()
        wl_min_label = QLabel("λ min:")
        self.lbl_min_value = QLabel(str(400)) 
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
        self.lbl_max_value = QLabel(str(800)) 
        self.slider_max = QSlider(Qt.Horizontal)
        self.slider_max.setMinimum(400)
        self.slider_max.setMaximum(800)
        self.slider_max.setValue(700)
        self.slider_max.valueChanged.connect(self.update_wl_range)
        wl_max_layout.addWidget(wl_max_label)
        wl_max_layout.addWidget(self.slider_max)
        wl_max_layout.addWidget(self.lbl_max_value)
        wl_layout.addLayout(wl_max_layout)


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
        

        main_layout.addLayout(delay_layout)
        main_layout.addSpacing(10)          
        main_layout.addLayout(wl_layout)
        main_layout.addSpacing(10)
        main_layout.addLayout(dial_layout)
        
       
        main_layout.setContentsMargins(5, 0, 5, 0)
        main_layout.setSpacing(15)  
                
        range_container = QWidget()
        range_container.setLayout(main_layout)
        range_container.setMaximumWidth(800)  #
        layout.addWidget(range_container)

        fit_group = QGroupBox("Modelo de ajuste t₀")
        fit_layout = QHBoxLayout()
        
        self.radio_poly = QRadioButton("Polinómico")
        self.radio_nonlinear = QRadioButton("No lineal")
        self.radio_nonlinear.setChecked(True) 
        
        fit_layout.addWidget(self.radio_poly)
        fit_layout.addWidget(self.radio_nonlinear)
        fit_group.setLayout(fit_layout)
        
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
        self.ax_map.tick_params(colors="black")
        self.ax_map.xaxis.label.set_color("black")
        self.ax_map.yaxis.label.set_color("black")
        self.ax_map.title.set_color("black")
        for spine in self.ax_map.spines.values():
            spine.set_color("black")
        
        if self.cbar is not None:
            self.cbar.ax.yaxis.set_tick_params(color="black", labelcolor="black")
            self.cbar.ax.yaxis.label.set_color("black")
            for spine in self.cbar.ax.spines.values():
                spine.set_color("black")
                
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

            self.new_window = TargetCls()
            self.new_window.show()


            self.close()

        except Exception as e:
            QMessageBox.critical(self, "Switch error", f"Cannot switch analyzer:\n{e}")

    def on_draw(self, event):
            """Captura el fondo para Blitting con protección anti-recursión."""
           
            if event is not None and event.canvas != self.canvas:
                return
            
            if self._is_drawing:
                return

            self._is_drawing = True 
            try:
                
                self.bg_cache = self.canvas.copy_from_bbox(self.figure.bbox)
                
               
                self.draw_animated_artists()
            finally:

                self._is_drawing = False
        
    def draw_animated_artists(self):
        """Dibuja solo los elementos móviles."""
        # Mapa
        if self.vline_map: self.ax_map.draw_artist(self.vline_map)
        if self.hline_map: self.ax_map.draw_artist(self.hline_map)
        if self.marker_map: self.ax_map.draw_artist(self.marker_map)
        
        # Plots pequeños (si tienen datos)
        if self.cut_time_small: self.ax_time_small.draw_artist(self.cut_time_small)
        if self.vline_time_small: self.ax_time_small.draw_artist(self.vline_time_small)
        if self.cut_spec_small: self.ax_spec_small.draw_artist(self.cut_spec_small)


    def open_global_fit(self):
        dlg = GlobalFitPanel(self)
        dlg.exec_()
    def _init_small_plots(self):

        self.ax_time_small.set_xlabel("Delay (ps)")
        self.ax_time_small.set_ylabel("ΔA")
        self.ax_time_small.set_title("Kinetics (cursor)")
        self.ax_time_small.set_xlim(-1, 3)
        self.cut_time_small, = self.ax_time_small.plot([], [], '-', lw=1.5)
    

        self.vline_time_small = self.ax_time_small.axvline(
            x=0, color='k', ls='--', lw=1, visible=False, zorder=5
        )
    

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
            

            self.ax_time_small.set_xlim(x_min, x_max)
            self.canvas.draw_idle()
    
        except ValueError:
            QMessageBox.warning(self, "Error", "Introduce valores numéricos válidos para los límites de Delay.")

    def remove_pump_fringe(self):
        """Quita la franja de bombeo directamente sobre los datos actuales."""
        if self.data is None:
            QMessageBox.warning(self, "No data", "Load data first.")
            return
    
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
            
            # =============================================================================
            #             NORMALIZACIÓN DATOS EN FLUPS
            # =============================================================================

            max_val = np.nanmax(np.abs(data))
            if max_val != 0:
                data = data / max_val
    
    
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
            self.slider_min.blockSignals(True) 
            self.slider_max.blockSignals(True)
            
            self.slider_min.setMinimum(0)
            self.slider_min.setMaximum(nwl - 1)
            self.slider_max.setMinimum(0)
            self.slider_max.setMaximum(nwl - 1)
            
            self.slider_min.setValue(0)
            self.slider_max.setValue(nwl - 1)
            
            self.slider_min.blockSignals(False)
            self.slider_max.blockSignals(False)
            self.update_wl_range()
        except Exception as e:
            QMessageBox.critical(self, "Error loading file", str(e))
    def apply_wl_range(self):
        min_val = self.slider_min.value()
        max_val = self.slider_max.value()
        print(f"Aplicando λ min={min_val}, λ max={max_val}")

    def _plot_discrete_map(self, ax, WL, TD, data, n_levels=5, cmap='jet', shading='auto', vmin=None, vmax=None):
        """Dibuja mapa tipo contourf con pcolormesh discreto."""

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
            """Dibuja el mapa principal configurado para Blitting (alta velocidad)."""
            if self.data is None: return
    
            # Limpieza estándar
            self.ax_map.clear()
            if self.cbar:
                try: self.cbar.remove()
                except: pass
                self.cbar = None
    
            # --- Determinar datos a pintar (respetando filtros) ---
            WL_plot = self.WL_visible if hasattr(self, "WL_visible") and self.WL_visible is not None else self.WL
            data_plot = self.data_visible if hasattr(self, "data_visible") and self.data_visible is not None else self.data
    
            # 1. Dibujar Mapa (Estático)
            if self.use_discrete_levels:
                self.pcm = self._plot_discrete_map(self.ax_map, WL_plot, self.TD, data_plot, n_levels=self.n_levels)
            else:
                self.pcm = self.ax_map.pcolormesh(WL_plot, self.TD, data_plot.T, shading="auto", cmap="jet")
    
            self.ax_map.set_yscale("symlog")
            self.ax_map.set_title("ΔA Map")
            self.ax_map.set_xlabel("Wavelength (nm)")
            self.ax_map.set_ylabel("Delay (ps)")
    
            divider = make_axes_locatable(self.ax_map)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            self.cbar = self.figure.colorbar(self.pcm, cax=cax, label="ΔA")
    
            # 2. Inicializar Elementos Dinámicos (animated=True)
            x0, y0 = WL_plot[0], self.TD[0]
            
            self.vline_map = self.ax_map.axvline(x0, color='k', ls='--', lw=1, animated=True, zorder=6)
            self.hline_map = self.ax_map.axhline(y0, color='k', ls='--', lw=1, animated=True, zorder=6)
            self.marker_map, = self.ax_map.plot([x0], [y0], 'wx', markersize=8, markeredgewidth=2, animated=True, zorder=7)
    
            # 3. Preparar subplots pequeños (IMPORTANTE: Fijar límites aquí)
            self.ax_time_small.clear()
            self.ax_spec_small.clear()
            
            # Inicializamos líneas animadas vacías o con el primer valor
            self.cut_time_small, = self.ax_time_small.plot(self.TD, data_plot[0, :], 'b-', lw=1.5, animated=True)
            self.vline_time_small = self.ax_time_small.axvline(y0, color='k', ls='--', lw=1, animated=True)
            
            self.cut_spec_small, = self.ax_spec_small.plot(WL_plot, data_plot[:, 0], 'r-', lw=1.5, animated=True)
    
            # --- FIJAR LÍMITES ESTÁTICOS ---
            vmin_g, vmax_g = np.nanmin(data_plot), np.nanmax(data_plot)
            margin = (vmax_g - vmin_g) * 0.05
            
            self.ax_time_small.set_xlim(self.TD.min(), self.TD.max())
            self.ax_time_small.set_ylim(vmin_g - margin, vmax_g + margin)
            self.ax_time_small.set_xlabel("Delay (ps)")
            self.ax_time_small.set_title("Kinetics (Preview)") # Título estático
    
            self.ax_spec_small.set_xlim(WL_plot.min(), WL_plot.max())
            self.ax_spec_small.set_ylim(vmin_g - margin, vmax_g + margin)
            self.ax_spec_small.set_xlabel("Wavelength (nm)")
            self.ax_spec_small.set_title("Spectra (Preview)") # Título estático
    
            # Conectar eventos
            if self.cid_click is None:
                self.cid_click = self.canvas.mpl_connect("button_press_event", self.on_click_map)
    
            # 4. Disparar el primer dibujado completo (Genera el bg_cache)
            self.canvas.draw()

    
    def update_wl_range(self):
            """
            Actualiza las variables de datos visibles según los sliders 
            y llama a plot_map para dibujar todo correctamente.
            """
            if getattr(self, "WL", None) is None or getattr(self, "data", None) is None:
                 # Actualizar textos a guiones si no hay datos
                if hasattr(self, "lbl_min_value"): self.lbl_min_value.setText("- nm")
                if hasattr(self, "lbl_max_value"): self.lbl_max_value.setText("- nm")
                return
    
            # 1. Obtener índices de los sliders
            wl_min_idx = int(self.slider_min.value())
            wl_max_idx = int(self.slider_max.value())
    
            # 2. Corregir cruces de índices
            if wl_min_idx >= wl_max_idx: 
                wl_max_idx = wl_min_idx + 1
            
            # Asegurar límites del array
            wl_min_idx = max(0, min(wl_min_idx, len(self.WL) - 1))
            wl_max_idx = max(0, min(wl_max_idx, len(self.WL) - 1))
    
            # 3. Actualizar Etiquetas de Texto (nm)
            try:
                self.lbl_min_value.setText(f"{self.WL[wl_min_idx]:.1f} nm")
                self.lbl_max_value.setText(f"{self.WL[wl_max_idx]:.1f} nm")
            except Exception:
                pass
    
            # 4. DEFINIR LOS DATOS VISIBLES (Estado Global de Visualización)
            source_data = self.data_corrected if getattr(self, "showing_corrected", False) else self.data
            
            # Cortamos los datos
            self.WL_visible = self.WL[wl_min_idx : wl_max_idx + 1]
            self.data_visible = source_data[wl_min_idx : wl_max_idx + 1, :]
    
            # 5. LLAMADA CENTRALIZADA
            self.plot_map()
            
    def enable_point_selection(self):
        self.clicked_points = []
        if self.cid_click is None:
            self.cid_click = self.canvas.mpl_connect("button_press_event", self.on_click_map)
        QMessageBox.information(self, "Mode: Select points",
                                "Click izquierdo: añadir punto\nClick derecho: borrar último punto.\nLuego pulsa 'Fit t₀'.")
    def update_small_cuts(self, x, y, WL_sel=None, data_sel=None):
            """Actualización completa tras un clic."""
            # Reutilizamos la lógica del movimiento simulando un evento
            # Esto asegura coherencia visual
            class MockEvent:
                pass
            evt = MockEvent()
            evt.xdata = x
            evt.ydata = y
            evt.inaxes = self.ax_map
            
            # Llamamos a on_move_map para pintar rápido
            self.on_move_map(evt)
            
            # Si fue un clic, aseguramos que se quede fijo (opcional)
            # self.canvas.draw_idle()
    
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
            """Versión optimizada: usa restore_region y blit en vez de redibujar todo."""
            # Si no hay caché o no estamos en el eje, salir
            if self.bg_cache is None or self.data is None: 
                return
            if event.inaxes != self.ax_map: 
                return
    
            # 1. Restaurar fondo limpio (borra cursores anteriores instantáneamente)
            self.canvas.restore_region(self.bg_cache)
    
            # 2. Actualizar posiciones matemáticas (sin dibujar aún)
            x, y = event.xdata, event.ydata
            
            # Líneas del mapa
            self.vline_map.set_xdata([x, x])
            self.hline_map.set_ydata([y, y])
            self.marker_map.set_data([x], [y])
            
            # Calcular índices para los subplots
            # Usamos WL_visible si existe, si no WL completo
            cur_WL = self.WL_visible if hasattr(self, 'WL_visible') and self.WL_visible is not None else self.WL
            cur_data = self.data_visible if hasattr(self, 'data_visible') and self.data_visible is not None else self.data
            
            if cur_WL is not None and len(cur_WL) > 0:
                idx_wl = int(np.abs(cur_WL - x).argmin())
                idx_td = int(np.abs(self.TD - y).argmin())
    
                # Actualizar curvas pequeñas
                self.cut_time_small.set_data(self.TD, cur_data[idx_wl, :])
                self.vline_time_small.set_xdata([y, y])
                self.cut_spec_small.set_data(cur_WL, cur_data[:, idx_td])
    
                # Info en barra de estado
                val = cur_data[idx_wl, idx_td]
                self.label_status.setText(f"Cursor: {x:.1f} nm, {y:.2f} ps | Val: {val:.4e}")
    
            # 3. Dibujar SOLO lo animado y volcar a pantalla
            self.draw_animated_artists()
            self.canvas.blit(self.figure.bbox)

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
            """Alterna entre mapa original y corregido usando el renderizado optimizado."""
            
            # 1. Validación de seguridad
            if self.data_corrected is None:
                QMessageBox.warning(self, "No corrected data", "Run 'Fit t₀' first.")
                return
    
            # 2. Alternar estado (flag booleano)
            self.showing_corrected = not getattr(self, "showing_corrected", False)
    
            # 3. Decidir la fuente de datos
            # Si showing_corrected es True, usamos los datos corregidos.
            # Si es False, usamos self.data (que es el base/original).
            source_data = self.data_corrected if self.showing_corrected else self.data
    
            # 4. Actualizar Textos
            if self.showing_corrected:
                self.btn_show_corr.setText("Show Base/Original Map")
                suffix = "(t₀ Corrected)"
            else:
                self.btn_show_corr.setText("Show Corrected Map")
                suffix = "(Base/Original)"
    
            # 5. Recalcular el slice visible (Respetando los Sliders)
            # Esto es crucial para que al cambiar no se resetee el zoom de longitud de onda
            if hasattr(self, 'slider_min') and hasattr(self, 'slider_max'):
                wl_min_idx = int(self.slider_min.value())
                wl_max_idx = int(self.slider_max.value())
                
                # Protecciones de índice
                if wl_min_idx >= wl_max_idx: wl_max_idx = wl_min_idx + 1
                wl_min_idx = max(0, min(wl_min_idx, len(self.WL) - 1))
                wl_max_idx = max(0, min(wl_max_idx, len(self.WL) - 1))
                
                # Actualizamos las variables que usa plot_map
                self.WL_visible = self.WL[wl_min_idx:wl_max_idx+1]
                self.data_visible = source_data[wl_min_idx:wl_max_idx+1, :]
            else:
                # Fallback por si no hay sliders
                self.WL_visible = self.WL
                self.data_visible = source_data
    
            # 6. LLAMADA MÁGICA: Usamos el plot_map optimizado
            # Esto se encargará del Blitting, SymLog, Limites y Eventos automáticamente.
            self.plot_map()
            
            # Actualizamos el título explícitamente para reflejar el estado
            tech_name = "TAS" if getattr(self, "is_TAS_mode", False) else "FLUPS"
            self.ax_map.set_title(f"ΔA Map ({tech_name}) {suffix}")
            
            # Un redraw final para asegurar que el título se actualice
            self.canvas.draw()
    
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
        # --- NUEVO: Inicializar variables para Blitting (optimización) ---
        self.bg_cache = None
        self.cid_draw = None
        self.cid_click = None
        self.cid_move = None
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
    # === Checkbox para conversión automática de .dat → .csv ===
        self.chk_convert_dat = QCheckBox("Convert .dat → .csv (IMDEA DATA)")
        self.chk_convert_dat.setChecked(True)  # activado por defecto

        # Insertarlo en el layout arriba o donde prefieras
        self.centralWidget().layout().addWidget(self.chk_convert_dat)
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

    def convert_dat_to_csv(self, file_path):
        """Convierte un archivo .dat en un .csv con el formato TAS."""
        try:
            data = np.loadtxt(file_path)
    
            # wl = primera columna
            wl = data[:, 0]
    
            # t = primera fila (ps)
            t = data[0] * 1e-3

            # Reemplazar en la matriz
            data[:, 0] = wl
            data[0, :] = t
    
            # Crear ruta .csv
            csv_path = os.path.splitext(file_path)[0] + ".csv"
    
            # Guardar
            np.savetxt(csv_path, data, delimiter=",")
            return csv_path
    
        except Exception as e:
            QMessageBox.critical(self, "Conversion error", f"Cannot convert .dat:\n{e}")
            return None
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
        # --- Conversión automática .dat → .csv si la opción está activada ---
        if self.chk_convert_dat.isChecked() and file_path_medida.lower().endswith(".dat"):
            new_path = self.convert_dat_to_csv(file_path_medida)
            if new_path:
                file_path_medida = new_path
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
        self.TD = raw[0, 1:]       # delay (ps)
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
        if self.chk_convert_dat.isChecked() and file_path_solvente.lower().endswith(".dat"):
            new_path = self.convert_dat_to_csv(file_path_solvente)
            if new_path:
                file_path_solvente = new_path        
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
        
        self.idx_min = 0
        self.idx_max = nwl - 1
        
        # --- CONEXIONES (ESTO ES LO CRUCIAL) ---
        try: self.slider_min.valueChanged.disconnect()
        except: pass
        try: self.slider_max.valueChanged.disconnect()
        except: pass
        self.slider_min.valueChanged.connect(self.update_wl_range)
        self.slider_max.valueChanged.connect(self.update_wl_range)
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


    # En TASAnalyzer (reemplazar la versión actual)
    def fit_t0_points(self):
        if not getattr(self, "clicked_points", None) or len(self.clicked_points) < 2:
            QMessageBox.warning(self, "Not enough points", "Select at least 2 points on the map.")
            return
    
        w_points = np.array([p['x'] for p in self.clicked_points])
        t0_points = np.array([p['y'] for p in self.clicked_points])
    
        try:
            # Re-calcular la base (self.data) con el solvente/shift más reciente
            self.update_am_sf() 
            
            # Usar self.data (Base Data: solvente-corregida) para el fit
            result = fit_t0(w_points, t0_points, self.WL, self.TD, self.data)
        except Exception as e:
            QMessageBox.critical(self, "Fit error", str(e))
            return
    
        
        # --- Guardar datos corregidos globalmente ---
        self.result_fit = result
        self.data_corrected = result['corrected']
        # ⚠️ LÍNEA ELIMINADA: La línea 'self.data = np.copy(self.data_corrected)' se elimina.
        # Ahora self.data_corrected mantiene los datos finales y self.data los base.
        
        self.plot_map(show_fit=True)
        self.btn_show_corr.setEnabled(True)
    
        # --- Crear carpeta de resultados junto al CSV y guardar ---
        base_dir = os.path.dirname(self.file_path)
        base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        save_dir = os.path.join(base_dir, f"{base_name}_results")
        os.makedirs(save_dir, exist_ok=True)
    
        data_corr = np.copy(self.data_corrected)
        WL = self.WL
        TD = self.TD
        t0_lambda = result['t0_lambda']
        popt = result['popt']
        method = result['method']
    
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
        

    def update_wl_range(self):
            """Actualiza los índices de recorte y refresca el mapa."""
            if self.medida is None:
                return
    
            # 1. Leer valores de los sliders
            # Aseguramos que son enteros (índices del array)
            s_min = int(self.slider_min.value())
            s_max = int(self.slider_max.value())
    
            # 2. Validar cruce (Min no puede ser >= Max)
            if s_min >= s_max:
                s_min = s_max - 1
                if s_min < 0: s_min = 0
                self.slider_min.blockSignals(True) # Evitar bucle infinito
                self.slider_min.setValue(s_min)
                self.slider_min.blockSignals(False)
    
            # 3. Guardar en las variables de clase
            self.idx_min = s_min
            self.idx_max = s_max
    
            # 4. Actualizar etiquetas de texto (Opcional, si tienes labels)
            # self.lbl_min_val.setText(f"{self.WL[s_min]:.1f} nm")
            # self.lbl_max_val.setText(f"{self.WL[s_max]:.1f} nm")
    
            # 5. Redibujar
            self.plot_map()
    # ------------------------------------------------------------------
    # ACTUALIZACIÓN DE MAPA TRAS SLIDERS
        # ------------------------------------------------------------------
    # En TASAnalyzer (reemplazar la versión actual)
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
        
        # Se elimina todo el bloque 'if hasattr(self, "data_corrected") ...' que causaba el doble cálculo.
        self.data = base_data 
    
        self.update_wl_range() 
    
        if hasattr(self, "global_fit_panel") and self.global_fit_panel is not None:
            self.global_fit_panel.update_from_parent()
    
        self._updating_am_sf = False
    # ------------------------------------------------------------------
    # DIBUJAR MAPA ΔA
    # ------------------------------------------------------------------
    def plot_map(self, show_fit=False):
            """Dibuja el mapa (SymLog en Y) con soporte para modo Corregido/Original."""
            
            # 1. Determinar qué datos usar (Base vs Corregidos)
            # Verificamos el flag que activa el botón toggle
            showing_corrected = getattr(self, "showing_corrected", False)
            
            if showing_corrected and hasattr(self, "data_corrected") and self.data_corrected is not None:
                source_data = self.data_corrected
                mode_suffix = "(t₀ Corrected)"
            else:
                # Si no hay flag o es False, usamos self.data (que ya tiene la resta de solvente)
                source_data = self.data
                mode_suffix = "(Base Data)"
    
            if source_data is None:
                return
    
            # --- Limpieza ---
            self.ax_map.clear()
            self.ax_time_small.clear()
            self.ax_spec_small.clear()
            
            # Reset de variables para evitar errores
            self.vline_map = None
            self.hline_map = None
            self.marker_map = None
            self.cut_time_small = None
            self.cut_spec_small = None
            
            if self.cbar:
                try: self.cbar.remove()
                except: pass
                self.cbar = None
    
            # --- 2. Recortes (Slicing) ---
            if not hasattr(self, 'idx_min'): self.idx_min = 0
            if not hasattr(self, 'idx_max'): self.idx_max = len(self.WL) - 1
    
            idx_start = self.idx_min
            idx_end = self.idx_max + 1
            
            wl_plot = self.WL[idx_start:idx_end]
            
            # AQUI ESTA EL CAMBIO IMPORTANTE: Usamos source_data en vez de self.data
            data_plot = source_data[idx_start:idx_end, :]
            
            if len(wl_plot) < 2: return
    
            # --- 3. Calcular Límites Globales ---
            g_min = np.nanmin(data_plot)
            g_max = np.nanmax(data_plot)
                
            data_range = g_max - g_min
            if data_range == 0: data_range = 1.0
            y_lim_min = g_min - (0.1 * data_range)
            y_lim_max = g_max + (0.1 * data_range)
    
            # --- 4. Dibujar Mapa Principal ---
            self.pcm = self.ax_map.pcolormesh(
                wl_plot, self.TD, data_plot.T,
                shading="auto", cmap="jet",
            )
            
            self.ax_map.set_yscale('symlog', linthresh=1.0)
            self.ax_map.set_xlabel("Wavelength (nm)")
            self.ax_map.set_ylabel("Delay (ps) - SymLog")
            
            # Actualizamos el título dinámicamente según el modo
            self.ax_map.set_title(f"ΔA Map (TAS) {mode_suffix}")
            
            self.ax_map.set_xlim(wl_plot.min(), wl_plot.max())
            
            # Colorbar
            divider = make_axes_locatable(self.ax_map)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            self.cbar = self.figure.colorbar(self.pcm, cax=cax, label="ΔA")
            self.ax_map.set_yscale('symlog', linthresh=1.0)
    
            # --- 5. Elementos Dinámicos ---
            mid_x = np.median(wl_plot)
            mid_y = np.median(self.TD)
            
            self.vline_map = self.ax_map.axvline(mid_x, color="k", ls="--", lw=1, animated=True)
            self.hline_map = self.ax_map.axhline(mid_y, color="k", ls="--", lw=1, animated=True)
            self.marker_map, = self.ax_map.plot([mid_x], [mid_y], "wx", markersize=8, markeredgewidth=2, animated=True)
    
            # --- 6. Configurar Subplots Pequeños ---
            
            # A) CINÉTICA (Abajo-Izquierda)
            y_cut_time = data_plot[np.abs(wl_plot - mid_x).argmin(), :]
            self.cut_time_small, = self.ax_time_small.plot(self.TD, y_cut_time, 'b-', animated=True)
            self.vline_time_small = self.ax_time_small.axvline(mid_y, color='k', ls='--', lw=1.2, animated=True)
            
            self.ax_time_small.set_xscale('linear') 
            self.ax_time_small.set_xlim(self.TD.min(), self.TD.max())
            self.ax_time_small.set_ylim(y_lim_min, y_lim_max)
            self.ax_time_small.set_title("Kinetics")
            self.ax_time_small.set_xlabel("Delay (ps)")
    
            # B) ESPECTRO (Abajo-Derecha)
            y_cut_spec = data_plot[:, np.abs(self.TD - mid_y).argmin()]
            self.cut_spec_small, = self.ax_spec_small.plot(wl_plot, y_cut_spec, 'r-', animated=True)
            
            self.ax_spec_small.set_xlim(wl_plot.min(), wl_plot.max())
            self.ax_spec_small.set_ylim(y_lim_min, y_lim_max)
            self.ax_spec_small.set_title("Spectrum")
            self.ax_spec_small.set_xlabel("Wavelength (nm)")
    
            # --- 7. Eventos ---
            self.bg_cache = None
            if self.cid_draw is not None: self.canvas.mpl_disconnect(self.cid_draw)
            self.cid_draw = self.canvas.mpl_connect('draw_event', self.on_draw)
            
            if self.cid_click is None:
                self.cid_click = self.canvas.mpl_connect("button_press_event", self.on_click_map)
            if self.cid_move is None:
                self.cid_move = self.canvas.mpl_connect("motion_notify_event", self.on_move_map)
    
            self.canvas.draw()
    def on_draw(self, event):
            """Captura el fondo para blitting cuando se redibuja la figura completa."""
            if event is not None and event.canvas != self.canvas:
                return
            # Copiamos la región del canvas (sin las líneas animadas)
            self.bg_cache = self.canvas.copy_from_bbox(self.figure.bbox)
            
            # Aprovechamos para redibujar las líneas animadas una vez
            self.draw_animated_artists()
            
    def draw_animated_artists(self):
            """Ayuda para dibujar solo los elementos dinámicos."""
            # 1. Verificación de seguridad:
            # Si vline_map no existe o es None, no hacemos nada.
            # Esto evita el crash cuando la ventana se abre antes de cargar datos.
            vline = getattr(self, 'vline_map', None)
            if vline is None:
                return
    
            # 2. Dibujar elementos del Mapa
            # (Como ya comprobamos vline, asumimos que el resto se crearon junto a él)
            try:
                self.ax_map.draw_artist(self.vline_map)
                self.ax_map.draw_artist(self.hline_map)
                self.ax_map.draw_artist(self.marker_map)
                
                # 3. Dibujar elementos de los subplots
                # Verificamos también estos por seguridad
                if getattr(self, 'cut_time_small', None) is not None:
                    self.ax_time_small.draw_artist(self.cut_time_small)
                    self.ax_time_small.draw_artist(self.vline_time_small)
                
                if getattr(self, 'cut_spec_small', None) is not None:
                    self.ax_spec_small.draw_artist(self.cut_spec_small)
    
            except AttributeError:
                # Si algo falla internamente en matplotlib (ej. ventana cerrada), ignoramos
                pass
    def update_small_cuts(self, x, y, WL_sel=None, data_sel=None):
            """Actualización completa (lenta) para clicks o cambios de slider."""
            # Podemos reutilizar la lógica de movimiento o forzar un draw completo
            # Para mantener coherencia visual tras un click:
            self.on_move_map(type('Event', (object,), {'xdata': x, 'ydata': y, 'inaxes': self.ax_map})())
            self.canvas.draw_idle() # Asegura que todo quede fijo

    # ------------------------------------------------------------------
    # EVENTO DE MOVIMIENTO DE RATÓN
    # ------------------------------------------------------------------
    def on_move_map(self, event):
            """Actualización ultra-rápida usando Blitting."""
            # 1. Validaciones básicas de ejes y datos
            if self.data is None or event.inaxes != self.ax_map:
                return
    
            # 2. --- CORRECCIÓN DEL ERROR ---
            # Verificamos si las líneas existen. Si vline_map es None,
            # significa que el gráfico se está limpiando o no se ha creado aún.
            # Usamos getattr por seguridad extra.
            if getattr(self, 'vline_map', None) is None:
                return
    
            # 3. Obtener coordenadas
            x, y = event.xdata, event.ydata
            if x is None or y is None: return
    
            # 4. Restaurar fondo limpio (borra las líneas anteriores)
            if self.bg_cache is not None:
                self.canvas.restore_region(self.bg_cache)
    
            # 5. Actualizar datos de las líneas (sin redibujar ejes)
            # --- Mapa ---
            self.vline_map.set_xdata([x, x])
            self.hline_map.set_ydata([y, y])
            self.marker_map.set_data([x], [y])
            
            # --- Datos para cortes ---
            # Búsqueda rápida de índices (usando WL y TD recortados si fuera necesario, 
            # pero para índices globales usamos self.WL/self.TD originales con cuidado)
            
            # Nota: Si usas recorte en plot_map, aquí debes tener cuidado. 
            # Para simplificar y evitar errores de índice, buscaremos en los arrays globales
            idx_wl = np.abs(self.WL - x).argmin()
            idx_td = np.abs(self.TD - y).argmin()
            
            # Validar índices (por si el ratón está fuera del rango de datos válidos)
            if idx_wl >= self.data.shape[0] or idx_td >= self.data.shape[1]:
                return
    
            # Actualizar curva Cinética
            y_time = self.data[idx_wl, :]
            self.cut_time_small.set_data(self.TD, y_time)
            self.vline_time_small.set_xdata([y, y])
            
            # Actualizar curva Espectro
            y_spec = self.data[:, idx_td]
            self.cut_spec_small.set_data(self.WL, y_spec)
    
            # 6. Dibujar SOLO los elementos animados
            self.draw_animated_artists()
    
            # 7. Volcar a pantalla (Blit)
            self.canvas.blit(self.figure.bbox)
            
            # Barra de estado
            val = self.data[idx_wl, idx_td]
            self.label_status.setText(f"Cursor: {x:.1f} nm, {y:.2f} ps | ΔA: {val:.4e}")


class Surface3DWindow(QDialog):
    """Ventana independiente para visualizar el 3D sin bloquear el panel principal."""
    def __init__(self, xs, ys, zs, scale='linear', parent=None):
        super().__init__(parent)
        self.setWindowTitle("3D Surface Preview")
        self.resize(800, 600)
        
        # Quitar el bloqueo modal para que deje tocar el panel de atrás
        self.setWindowModality(Qt.NonModal)

        layout = QVBoxLayout()
        
        # Crear el Canvas de Matplotlib
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # Dibujar el gráfico
        self.plot_data(xs, ys, zs, scale)

    def plot_data(self, xs, ys, zs, scale):
        ax = self.fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(xs, ys)
        Z = zs.T
        
        surf = ax.plot_surface(X, Y, Z, cmap='jet',edgecolor='none', antialiased=True)
        n_ticks = 6
        xticks = np.linspace(xs.min(), xs.max(), n_ticks)

        # ax.set_zticks(zticks)
        # ax.set_xlabel("Wavelength (nm)")
        ax.set_xlabel("Energy (eV)")
        
        ax.set_ylabel("Delay (ps)")
        ax.set_zlabel("Transient absorption / -")
        ax.grid(False)
        ax.xaxis._axinfo["grid"]["linewidth"] = 0
        ax.yaxis._axinfo["grid"]["linewidth"] = 0
        ax.zaxis._axinfo["grid"]["linewidth"] = 0
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.view_init(elev=25, azim=75)
        
        if scale == 'symlog':
            ax.set_yscale('symlog', linthresh=1.0)
            
        self.fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        self.canvas.draw()
        
class GlobalFitPanel(QDialog):
    def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Global Fit Analysis")
            self.resize(1150, 750) 
    
            # --- 1. Variables de Datos ---
            
            self.parent_app = parent
            self.data_c = None   
            self.data_raw = None 
            self.TD = None       
            self.WL = None       
            self.base_dir = None
            
            # Determine base dir
            if hasattr(parent, "save_dir") and parent.save_dir:
                self.base_dir = parent.save_dir
            elif hasattr(parent, "file_path") and parent.file_path:
                base_name = os.path.splitext(os.path.basename(parent.file_path))[0]
                self.base_dir = os.path.join(os.path.dirname(parent.file_path), f"{base_name}_Results")
                os.makedirs(self.base_dir, exist_ok=True)
            else:
                self.base_dir = os.getcwd()
    
            # --- 2. Variables del Ajuste ---
            self.numExp = 2
            self.model_type = 'Parallel' 
            self.t0_choice = 'No'
            self.tech = 'TAS'
            self.yscale = 'linear'
            
            # Placeholders para resultados
            self.fit_result = None
            self.fit_x = None
            self.As = None
            # ... resto de variables fit
            self.fit_resid = None
            self.fit_fitres = None
            self.ci = None
            self.errAs = None
            self.t0s = None
            self.errt0s = None
            self.errtaus = None
            self.ini = None
            self.limi = None
            self.lims = None
    
            # --- 3. DISEÑO PRINCIPAL (LAYOUT) ---
            main_layout = QHBoxLayout() 
            
            # --- A. Panel Izquierdo (Sidebar) ---
            self.sidebar = QWidget()
            self.sidebar.setFixedWidth(340) 
            self.sidebar_layout = QVBoxLayout(self.sidebar)
            self.sidebar_layout.setContentsMargins(5, 5, 5, 5)
            
            self._init_sidebar_ui() 
            
            main_layout.addWidget(self.sidebar)
    
            # --- B. Panel Derecho (Gráficos) ---
            self.right_area = QWidget()
            self.right_layout = QVBoxLayout(self.right_area)
            
            self._init_plots_ui() 
            
            main_layout.addWidget(self.right_area)
    
            self.setLayout(main_layout)
    
    # --- IMPORTANTE: INICIALIZAR VARIABLES DE PLOTTING ---
            # Si no pones esto, te dará el error de la Imagen 1
            self.pcm_exp = None
            self.cbar_exp = None
            self.pcm_fit = None
            self.cbar_fit = None
            self.pcm_resid = None
            self.cbar_resid = None
        

    def _init_sidebar_ui(self):
        """Construye todos los botones y cajas del panel izquierdo."""
        l = self.sidebar_layout
        
        # --- Grupo 1: Carga de Datos ---
        gb_load = QGroupBox("1. Data Source")
        v_load = QVBoxLayout()
        
        self.label_status = QLabel("No data loaded")
        self.label_status.setStyleSheet("color: gray; font-style: italic; font-weight: bold;")
        v_load.addWidget(self.label_status)
        
        h_btns = QHBoxLayout()
        self.btn_load = QPushButton("Load .npy")
        self.btn_load.clicked.connect(self.load_data) # Descomentar cuando tengas la funcion
        h_btns.addWidget(self.btn_load)
        
        self.btn_parent = QPushButton("Use Parent Data")
        self.btn_parent.clicked.connect(self.use_parent_data) # Descomentar luego
        h_btns.addWidget(self.btn_parent)
        
        v_load.addLayout(h_btns)
        gb_load.setLayout(v_load)
        l.addWidget(gb_load)

        # --- Grupo 2: Pre-procesado ---
        gb_prep = QGroupBox("2. Pre-processing")
        form_prep = QFormLayout()

        # Baseline
        self.spin_bl = QSpinBox()
        self.spin_bl.setRange(0, 500)
        self.spin_bl.setValue(5)
        self.spin_bl.valueChanged.connect(self.apply_baseline_correction) # Descomentar luego
        form_prep.addRow("Baseline Pts:", self.spin_bl)

        # Rangos WL
        self.spin_wl_min = QDoubleSpinBox(); self.spin_wl_min.setRange(0, 10000); 
        self.spin_wl_max = QDoubleSpinBox(); self.spin_wl_max.setRange(0, 10000); 
        self.spin_wl_max.setDecimals(6)    
        self.spin_wl_max.setSingleStep(0.5)
        self.spin_wl_min.setDecimals(6)
        self.spin_wl_min.setSingleStep(0.1)
        
        form_prep.addRow("Min WL (nm):", self.spin_wl_min)
        form_prep.addRow("Max WL (nm):", self.spin_wl_max)

        # Rangos Tiempo
        self.spin_t_min = QDoubleSpinBox(); self.spin_t_min.setRange(-100, 1e6); self.spin_t_min.setDecimals(3)
        self.spin_t_max = QDoubleSpinBox(); self.spin_t_max.setRange(-100, 1e6); self.spin_t_max.setDecimals(3)
        form_prep.addRow("Min Time (ps):", self.spin_t_min)
        form_prep.addRow("Max Time (ps):", self.spin_t_max)
        

        # Binning
        self.spin_bin = QSpinBox()
        self.spin_bin.setRange(1, 50)
        self.spin_bin.setValue(1)
        form_prep.addRow("Binning:", self.spin_bin)
        
        # Botón Preview
        self.btn_preview = QPushButton("Apply & Preview")
        self.btn_preview.clicked.connect(self._preview_data_processing) # Descomentar luego
        form_prep.addRow(self.btn_preview)

        gb_prep.setLayout(form_prep)
        l.addWidget(gb_prep)

        # --- Grupo 3: Modelo ---
        gb_model = QGroupBox("3. Model Settings")
        form_model = QFormLayout()
        
        self.btn_svd = QPushButton("Run SVD Analysis")
        self.btn_svd.clicked.connect(self.run_svd)
        form_model.addRow(self.btn_svd)
        
        # --- D. Visualización (NUEVO) ---
        gb_vis = QGroupBox("4. Visualization")
        form_vis = QFormLayout()
        
        self.btn_plot_3d = QPushButton("Ver Mapa en 3D")
        self.btn_plot_3d.clicked.connect(self.plot_3d_surface) # Conectamos a la nueva función
        form_vis.addRow(self.btn_plot_3d)
        
        self.combo_scale = QComboBox()
        self.combo_scale.addItems(["Linear", "SymLog"])
        self.combo_scale.currentTextChanged.connect(self._on_scale_changed) # Conectamos función
        form_vis.addRow("Time Axis Scale:", self.combo_scale)
        
        gb_vis.setLayout(form_vis)
        l.addWidget(gb_vis)
        # Num Exponenciales
        self.spin_numExp = QSpinBox()
        self.spin_numExp.setRange(1, 6)
        self.spin_numExp.setValue(2)
        form_model.addRow("Exponentials:", self.spin_numExp)

        # Tipo de Modelo
        self.combo_model = QComboBox()
        self.combo_model.addItems(["Parallel (DAS)", "Sequential (SAS)"])
        form_model.addRow("Model Type:", self.combo_model)

        # Técnica
        self.combo_tech = QComboBox()
        self.combo_tech.addItems(["FLUPS", "TAS", "TCSPC"])
        form_model.addRow("Technique:", self.combo_tech)

        # Chirp
        self.chk_chirp = QCheckBox("Fit Independent t0 (Chirp)")
        form_model.addRow(self.chk_chirp)
        
        # Initial Guesses
        self.btn_edit_guess = QPushButton("Edit Initial Guesses")
        self.btn_edit_guess.clicked.connect(self._open_guess_editor_and_update)
        form_model.addRow(self.btn_edit_guess)
        gb_model.setLayout(form_model)
        l.addWidget(gb_model)

        # --- Botones Finales ---
        self.btn_run = QPushButton("RUN FIT")
        self.btn_run.setFixedHeight(40)  
        self.btn_run.setEnabled(False) # Se activa al cargar datos
        self.btn_run.clicked.connect(self.run_fit_pipeline) # Descomentar luego
        l.addWidget(self.btn_run)
        
        self.btn_show_das = QPushButton("Show Plots / Results")
        self.btn_show_das.setEnabled(False)
        self.btn_show_das.clicked.connect(self.plot_das_and_more) # Descomentar luego
        l.addWidget(self.btn_show_das)

        l.addStretch() # Empujar todo arriba
        
    def run_svd(self):
        if self.data_c is None:
            QMessageBox.warning(self, "Error", "Carga y procesa datos primero (Apply & Preview).")
            return
    
        # 1. Ejecutar SVD matemático
        # data_c suele ser [WL x TD]
        try:
            # Usamos economy SVD (compute_uv=True por defecto)
            U, s, Vh = np.linalg.svd(self.data_c, full_matrices=False)
            
            self.svd_U = U    # Vectores espectrales (Especies)
            self.svd_s = s    # Importancia de cada uno
            self.svd_V = Vh.T # Vectores temporales (Cinéticas)
    
            self._plot_svd_results()
            self.tabs.setCurrentWidget(self.tab_svd) # Cambiar a la pestaña SVD automáticamente
            
        except Exception as e:
            print(f"SVD Error: {e}")
    def _create_svd_canvas(self, tab_widget):
        fig = plt.Figure(figsize=(5, 8))
        # ax1: Scree Plot, ax2: Primeros Componentes Espectrales
        ax1 = fig.add_subplot(211) 
        ax2 = fig.add_subplot(212)
        canvas = FigureCanvas(fig)
        
        layout = QVBoxLayout()
        layout.addWidget(canvas)
        tab_widget.setLayout(layout)
        return canvas, (ax1, ax2)

    def _plot_svd_results(self):
        ax1, ax2 = self.ax_svd
        ax1.clear()
        ax2.clear()
    
        # --- Plot 1: Scree Plot (Log scale) ---
        n_comp = min(len(self.svd_s), 10) # Ver top 10
        ax1.semilogy(range(1, n_comp + 1), self.svd_s[:n_comp], 'o-', color='red')
        ax1.set_title("Singular Values (Scree Plot)")
        ax1.set_ylabel("Eigenvalue (log)")
        ax1.set_xlabel("Component Number")
        ax1.grid(True, which="both", ls="-", alpha=0.2)
    
        # 2. Componentes Espectrales 
        wl = getattr(self, '_wl_proc', self.WL)
        # Leemos el valor del SpinBox de la interfaz
        n_mostrar = self.spin_numExp.value() 
        
        for i in range(min(n_mostrar, len(self.svd_s))):
            ax2.plot(wl, self.svd_U[:, i], label=f"Comp {i+1}")
        
        ax2.set_title(f"First {n_mostrar} Spectral Components")
        ax2.set_xlabel("Energy / Wavelength")
        ax2.axhline(0, color='black', lw=1, alpha=0.5)
        ax2.legend()
        
        self.canvas_svd.draw()        
        def _generate_defaults(self):
            """Genera los valores iniciales (Guesses) basados en la configuración actual."""
            # 1. Leer configuración actual
            numExp = self.spin_numExp.value()
            t0_choice = 'Yes' if self.chk_chirp.isChecked() else 'No'
            tech = self.combo_tech.currentText()
            
            if self.data_c is not None:
                numWL = self.data_c.shape[0]
            elif self.WL is not None:
                numWL = len(self.WL)
            else:
                QMessageBox.warning(self, "Warning", "Load data first to generate guesses.")
                return False
    
            # 2. Calcular tamaño vector L
            if t0_choice == 'Yes':
                L = 1 + numExp + numWL*(numExp+1)
            else:
                L = 2 + numExp + numWL*numExp
                
            self.ini = np.zeros(L)
            self.limi = -np.inf * np.ones(L)
            self.lims = np.inf * np.ones(L)
    
            # 3. Rellenar valores (Tu lógica estándar)
            taus_defaults = [0.5, 5.0, 50.0, 500.0, 2000.0, 5000.0]
            w_guess = 0.15 if tech == 'TAS' else (0.3 if tech == 'FLUPS' else 0.1)
            
            if t0_choice == 'No':
                # [w, t0, tau1..n, A...]
                self.ini[0] = w_guess; self.limi[0] = 0.05; self.lims[0] = 2.0
                self.ini[1] = 0.0;     self.limi[1] = -5.0; self.lims[1] = 5.0
                
                base_tau = 2
                for n in range(numExp):
                    idx = base_tau + n
                    val_t = taus_defaults[n] if n < len(taus_defaults) else 1000.0*(n+1)
                    self.ini[idx] = val_t; self.limi[idx] = 0.001; self.lims[idx] = 1e8
                
                start_A = base_tau + numExp
                val_A = 1000.0 if tech == 'TCSPC' else (5.0 if tech == 'FLUPS' else 0.01)
                self.ini[start_A:] = val_A
                
            else:
                # Chirp logic placeholder
                self.ini[0] = w_guess; self.limi[0] = 0.05; self.lims[0] = 2.0
                for n in range(numExp):
                    self.ini[1+n] = taus_defaults[n] if n < len(taus_defaults) else 100.0
                    self.limi[1+n] = 0.001; self.lims[1+n] = 1e8
                
                base_idx = 1 + numExp
                params_per_wl = 1 + numExp
                val_A = 1000.0 if tech == 'TCSPC' else 0.1
                self.ini[base_idx:] = val_A
                self.ini[base_idx::params_per_wl] = 0.0
                self.limi[base_idx::params_per_wl] = -5.0
                self.lims[base_idx::params_per_wl] = 5.0
            
        return True
    def _on_scale_changed(self, text):
            """Actualiza la variable de escala y repinta los gráficos."""
            self.yscale = text.lower() # 'linear' o 'symlog'
            
            # Repintar todo lo que esté activo
            self._update_exp_canvas()
            self._update_fit_canvas()
            self._update_resid_canvas()

    def _init_plots_ui(self):
            """Construye los Tabs y gráficos del panel derecho."""
            l = self.right_layout
            
            # Tabs
            self.tabs = QTabWidget()
            
            # --- ESTILO CORREGIDO PARA TEXTO NEGRO ---
            self.tabs.setStyleSheet("""
                        QTabWidget::pane { 
                            border: 1px solid #999; 
                            background: white; 
                        }
                        QTabBar::tab { 
                            background: #e0e0e0; 
                            color: black;
                            padding: 8px 20px; 
                            border: 1px solid #bbb; 
                            border-bottom: none; 
                            border-top-left-radius: 4px; 
                            border-top-right-radius: 4px; 
                            margin-right: 2px;
                        }
                        QTabBar::tab:selected { 
                            background: #ffffff; 
                            /* font-weight: bold;  <--- LINEA BORRADA */
                            border-bottom: 1px solid #ffffff; 
                        }
                        QTabBar::tab:hover {
                            background: #d0d0d0;
                        }
                    """)
            
            self.tab_exp = QWidget()
            self.tab_fit = QWidget()
            self.tab_resid = QWidget()
            self.tab_svd = QWidget() 
            
            self.tabs.addTab(self.tab_exp, "Experimental")
            self.tabs.addTab(self.tab_fit, "Fit Reconstructed")
            self.tabs.addTab(self.tab_resid, "Residuals")
            self.tabs.addTab(self.tab_svd, "SVD Diagnosis")
            
            # Crear Canvas (usando helper)
            self.canvas_exp, self.ax_exp = self._create_canvas_for_tab(self.tab_exp)
            self.canvas_fit, self.ax_fit = self._create_canvas_for_tab(self.tab_fit)
            self.canvas_resid, self.ax_resid = self._create_canvas_for_tab(self.tab_resid)
            self.canvas_svd, self.ax_svd = self._create_svd_canvas(self.tab_svd)
            
            l.addWidget(self.tabs)
            
            # Barra de progreso
            self.progress_bar = QProgressBar()
            self.progress_bar.setValue(0)
            self.progress_bar.setTextVisible(True)
            l.addWidget(self.progress_bar)
    def plot_3d_surface(self):
        """Lanza la ventana 3D independiente (No Modal)."""
        if self.data_c is None:
            QMessageBox.warning(self, "Sin datos", "Aplica 'Preview' antes de ver el 3D.")
            return
    
        # Obtener datos actuales
        xs = getattr(self, '_wl_proc', self.WL)
        ys = getattr(self, '_td_proc', self.TD)
        zs = self.data_c
        scale = getattr(self, 'yscale', 'linear')
    
        # Crear y mostrar la ventana (guardando referencia en self)
        self.pop_3d = Surface3DWindow(xs, ys, zs, scale, parent=self)
        self.pop_3d.show() # .show() no bloquea la ejecución    
    def _generate_defaults(self):
            """Genera los valores iniciales (Guesses) basados en la configuración actual."""
            # 1. Leer configuración actual
            numExp = self.spin_numExp.value()
            t0_choice = 'Yes' if self.chk_chirp.isChecked() else 'No'
            tech = self.combo_tech.currentText()
            
            # Necesitamos saber numWL para calcular el tamaño
            # Si data_c existe usamos su tamaño, si no, usamos WL raw, si no, error
            if self.data_c is not None:
                numWL = self.data_c.shape[0]
            elif self.WL is not None:
                numWL = len(self.WL)
            else:
                QMessageBox.warning(self, "Warning", "Load data first to generate guesses.")
                return False
    
            # 2. Calcular tamaño vector L
            if t0_choice == 'Yes':
                L = 1 + numExp + numWL*(numExp+1)
            else:
                L = 2 + numExp + numWL*numExp
                
            self.ini = np.zeros(L)
            self.limi = -np.inf * np.ones(L)
            self.lims = np.inf * np.ones(L)
    
            # 3. Rellenar valores (Tu lógica estándar)
            taus_defaults = [0.5, 5.0, 50.0, 500.0, 2000.0, 5000.0]
            w_guess = 0.15 if tech == 'TAS' else (0.3 if tech == 'FLUPS' else 0.1)
            
            if t0_choice == 'No':
                # [w, t0, tau1..n, A...]
                self.ini[0] = w_guess; self.limi[0] = 0.05; self.lims[0] = 2.0
                self.ini[1] = 0.0;     self.limi[1] = -5.0; self.lims[1] = 5.0
                
                base_tau = 2
                for n in range(numExp):
                    idx = base_tau + n
                    val_t = taus_defaults[n] if n < len(taus_defaults) else 1000.0*(n+1)
                    self.ini[idx] = val_t; self.limi[idx] = 0.001; self.lims[idx] = 1e8
                
                start_A = base_tau + numExp
                val_A = 1000.0 if tech == 'TCSPC' else (5.0 if tech == 'FLUPS' else 0.01)
                self.ini[start_A:] = val_A
                
            else:
                # Chirp logic placeholder
                self.ini[0] = w_guess; self.limi[0] = 0.05; self.lims[0] = 2.0
                for n in range(numExp):
                    self.ini[1+n] = taus_defaults[n] if n < len(taus_defaults) else 100.0
                    self.limi[1+n] = 0.001; self.lims[1+n] = 1e8
                
                base_idx = 1 + numExp
                params_per_wl = 1 + numExp
                val_A = 1000.0 if tech == 'TCSPC' else 0.1
                self.ini[base_idx:] = val_A
                self.ini[base_idx::params_per_wl] = 0.0
                self.limi[base_idx::params_per_wl] = -5.0
                self.lims[base_idx::params_per_wl] = 5.0
                
            return True
    def _create_canvas_for_tab(self, tab_widget):
        """Helper para inicializar matplotlib dentro de un tab."""
        fig = plt.Figure(figsize=(5,4))
        ax = fig.add_subplot(111)
        canvas = FigureCanvas(fig)
        
        layout = QVBoxLayout()
        layout.addWidget(canvas)
        tab_widget.setLayout(layout)
        
        return canvas, ax

    # --- Métodos auxiliares de limpieza de UI ---
    def _clear_colorbar_if_exists(self, cbar):
        try:
            if cbar is not None: cbar.remove()
        except: pass

    def use_parent_data(self):
        """Cargar datos del parent y actualizar canvas"""
        self.update_from_parent()
        self.btn_run.setEnabled(True)
        self.btn_show_das.setEnabled(False)
        
    def update_from_parent(self):
         p = self.parent_app
         if p is None: return
             
         if getattr(p, "is_TAS_mode", False):
              if hasattr(p, "data_corrected") and p.data_corrected is not None:
                  incoming_data = np.array(p.data_corrected, copy=True)
                  # ...
              # ... (resto de ifs)
         
         # --- AQUÍ EL CAMBIO ---
         # Guardamos en RAW
         self.data_raw = incoming_data 
         self.WL = getattr(p, "WL", None)
         self.TD = getattr(p, "TD", None)
         
         # Aplicamos la corrección y pintamos
         self.apply_baseline_correction()
    def apply_baseline_correction(self):
            """Recalcula data_c desde data_raw usando el valor actual del SpinBox y actualiza el plot."""
            if self.data_raw is None:
                return
    
            n_pts = self.spin_bl.value()
            
            # Siempre trabajamos desde la copia original 'raw'
            # para no acumular restas.
            temp_data = self.data_raw.copy()
            
            if n_pts > 0:
                if temp_data.shape[1] >= n_pts:
                    # Calcular baseline (media de las primeras n columnas de tiempo)
                    # Asumiendo shape [NumWL, NumTD] o viceversa. 
                    # En tu código anterior usabas axis=1, lo que implica [WL, Time]
                    baseline = np.mean(temp_data[:, :n_pts], axis=1, keepdims=True)
                    temp_data = temp_data - baseline
                else:
                    print("Warning: Not enough points for baseline.")
    
            # Actualizamos la variable oficial que usa el ajuste
            self.data_c = temp_data
            
            # Repintamos el canvas experimental inmediatamente
            self._update_exp_canvas()
# --- LÓGICA DE CARGA DE DATOS ---

    def _update_ui_limits_from_data(self):
        """Actualiza los rangos de las cajas numéricas (SpinBoxes) según los datos cargados."""
        if self.WL is not None and len(self.WL) > 0:
            self.spin_wl_min.setValue(np.min(self.WL))
            self.spin_wl_max.setValue(np.max(self.WL))
        
        if self.TD is not None and len(self.TD) > 0:
            self.spin_t_min.setValue(np.min(self.TD))
            self.spin_t_max.setValue(np.max(self.TD))
        
        # Al cargar, reseteamos data_c a raw y pintamos
        self.data_c = self.data_raw.copy()
        
        # Pintamos inmediatamente la data cruda
        self._update_exp_canvas(use_processed=False)

    def use_parent_data(self):
        """Cargar datos desde la ventana principal (si existe)."""
        if self.parent_app is None: return
        
        # Reutilizamos tu lógica original de detección
        if hasattr(self.parent_app, "data_corrected") and self.parent_app.data_corrected is not None:
            self.data_raw = np.array(self.parent_app.data_corrected, copy=True)
            self.WL = getattr(self.parent_app, "WL", None)
            self.TD = getattr(self.parent_app, "TD", None)
            
            # Detectar técnica
            if getattr(self.parent_app, "is_TAS_mode", False):
                self.combo_tech.setCurrentText("TAS")
            else:
                self.combo_tech.setCurrentText("FLUPS")
                
            self._update_ui_limits_from_data()
            self.btn_run.setEnabled(True)
            self.label_status.setText(f"Loaded from Parent: {len(self.WL)} WL, {len(self.TD)} TD")

    def load_data(self):
        """Cargar desde .npy usando tu módulo 'fit'."""
        try:
            raw_data, TD, WL, base_dir = fit.load_npy(self)
            
            self.data_raw = raw_data.copy()
            self.TD = TD
            self.WL = WL
            self.base_dir = base_dir
            
            self._update_ui_limits_from_data()
            self.btn_run.setEnabled(True)
            self.label_status.setText(f"Loaded File: {len(self.WL)} WL, {len(self.TD)} TD")
            
        except Exception as e:
            QMessageBox.critical(self, "Error loading", str(e))

    def _clear_colorbar_if_exists(self, cbar):
        try:
            if cbar is not None:
                cbar.remove()
        except Exception:
            pass
    
# --- LÓGICA DE PROCESADO Y VISUALIZACIÓN ---

    def _preview_data_processing(self):
        """
        Toma data_raw, aplica Baseline -> Crop WL -> Crop Time -> Binning 
        y guarda el resultado en self.data_c para usarlo en el ajuste.
        """
        if self.data_raw is None: return
        
        # 1. Empezar siempre desde copia de RAW
        temp_data = self.data_raw.copy()
        temp_WL = self.WL.copy()
        temp_TD = self.TD.copy()

        # 2. Baseline Correction
        n_pts = self.spin_bl.value()
        if n_pts > 0 and temp_data.shape[1] >= n_pts:
            # Asumiendo forma (WL, TD) -> axis 1 es tiempo
            baseline = np.mean(temp_data[:, :n_pts], axis=1, keepdims=True)
            temp_data = temp_data - baseline

        # 3. Crop Wavelength
        w_min = self.spin_wl_min.value()
        w_max = self.spin_wl_max.value()
        mask_w = (temp_WL >= min(w_min, w_max)) & (temp_WL <= max(w_min, w_max))
        
        if np.any(mask_w):
            temp_data = temp_data[mask_w, :]
            temp_WL = temp_WL[mask_w]

        # 4. Crop Time
        t_min = self.spin_t_min.value()
        t_max = self.spin_t_max.value()
        mask_t = (temp_TD >= min(t_min, t_max)) & (temp_TD <= max(t_min, t_max))
        
        if np.any(mask_t):
            temp_data = temp_data[:, mask_t]
            temp_TD = temp_TD[mask_t]

        # 5. Binning (Simple averaging)
        b_size = self.spin_bin.value()
        if b_size > 1:
            # Binning en eje espectral (WL)
            n_wl = temp_data.shape[0]
            new_len = n_wl // b_size
            if new_len > 0:
                # Recortamos el sobrante y hacemos reshape+mean
                temp_data = temp_data[:new_len*b_size, :]
                temp_data = temp_data.reshape(new_len, b_size, temp_data.shape[1]).mean(axis=1)
                temp_WL = temp_WL[:new_len*b_size]
                temp_WL = temp_WL.reshape(new_len, b_size).mean(axis=1)

        # GUARDAR RESULTADO PROCESADO
        self.data_c = temp_data
        
        # Guardamos versiones temporales de WL/TD para pintar correctamente
        self._wl_proc = temp_WL
        self._td_proc = temp_TD
        
        # Pintar
        self._update_exp_canvas(use_processed=True)
        self.label_status.setText(f"Data Ready: {len(temp_WL)} WL, {len(temp_TD)} TD")

    def _update_exp_canvas(self, use_processed=False):
            """Pinta el mapa experimental con escala dinámica y soporte Linear/SymLog."""
            if self.data_c is None: return
            
            self.ax_exp.clear()
            self._clear_colorbar_if_exists(self.cbar_exp)
            
            # Elegir qué ejes usar
            if use_processed and hasattr(self, '_wl_proc'):
                Xs = self._wl_proc
                Ys = self._td_proc
                Title = "Experimental (Processed)"
            else:
                Xs = self.WL
                Ys = self.TD
                Title = "Experimental (Raw)"
                
            # Protección ejes
            if Xs.shape[0] != self.data_c.shape[0] or Ys.shape[0] != self.data_c.shape[1]:
                Xs = np.arange(self.data_c.shape[0])
                Ys = np.arange(self.data_c.shape[1])
    
            try:
                vals = self.data_c.flatten()
                vmin = np.percentile(vals, 1) 
                vmax = np.percentile(vals, 99)
                
                self.pcm_exp = self.ax_exp.pcolormesh(Xs, Ys, self.data_c.T, 
                                                      shading="auto", cmap='jet', 
                                                      vmin=vmin, vmax=vmax)
                
                self.ax_exp.set_title(Title)
                self.ax_exp.set_xlabel("Energy (eV)")
                self.ax_exp.set_ylabel("Delay (ps)")
                
                # --- APLICAR ESCALA Y (CONDICIONAL) ---
                if hasattr(self, 'yscale') and self.yscale == 'symlog':
                    self.ax_exp.set_yscale('symlog', linthresh=1.0)
                else:
                    self.ax_exp.set_yscale('linear')
                
                divider = make_axes_locatable(self.ax_exp)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                self.cbar_exp = self.canvas_exp.figure.colorbar(self.pcm_exp, cax=cax, label='Transient absorption / -')
                
                self.canvas_exp.draw_idle()
                
            except Exception as e:
                print(f"Plotting error: {e}")
    def _update_fit_canvas(self):
            """Pinta la reconstrucción con escala dinámica y soporte para Log/Linear."""
            if self.fit_fitres is None: return
    
            self.ax_fit.clear()
            self._clear_colorbar_if_exists(self.cbar_fit)
            
            Xs = getattr(self, '_wl_proc', self.WL)
            Ys = getattr(self, '_td_proc', self.TD)
            Z = self.fit_fitres.T 
    
            if Xs is None or Xs.shape[0] != Z.shape[1]: Xs = np.arange(Z.shape[1])
            if Ys is None or Ys.shape[0] != Z.shape[0]: Ys = np.arange(Z.shape[0])
    
            try:
                if Z.shape[0] < 2 or Z.shape[1] < 2: return
    
                vals = Z.flatten()
                vmin = np.percentile(vals, 1)  
                vmax = np.percentile(vals, 99) 
    
                self.pcm_fit = self.ax_fit.pcolormesh(Xs, Ys, Z, shading='auto', cmap='jet', 
                                                      vmin=vmin, vmax=vmax)
                
                self.ax_fit.set_title("Fit Reconstructed")
                self.ax_fit.set_xlabel("Energy (eV)")
                self.ax_fit.set_ylabel("Delay (ps)")
                
                # --- APLICAR ESCALA Y (CONDICIONAL) ---
                if hasattr(self, 'yscale') and self.yscale == 'symlog':
                    self.ax_fit.set_yscale('symlog', linthresh=1.0)
                else:
                    self.ax_fit.set_yscale('linear')
                
                divider = make_axes_locatable(self.ax_fit)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                self.cbar_fit = self.canvas_fit.figure.colorbar(self.pcm_fit, cax=cax, label='Transient absorption / -')
                self.canvas_fit.draw()
            except Exception as e:
                print(f"Error painting Fit: {e}")
        
    def _update_resid_canvas(self):
            """Pinta residuos con escala dinámica, JET y soporte para Log/Linear."""
            if self.fit_resid is None: return
    
            self.ax_resid.clear()
            self._clear_colorbar_if_exists(self.cbar_resid)
            
            Xs = getattr(self, '_wl_proc', self.WL)
            Ys = getattr(self, '_td_proc', self.TD)
            Z = self.fit_resid.T
    
            if Xs is None or Xs.shape[0] != Z.shape[1]: Xs = np.arange(Z.shape[1])
            if Ys is None or Ys.shape[0] != Z.shape[0]: Ys = np.arange(Z.shape[0])
    
            try:
                if Z.shape[0] < 2 or Z.shape[1] < 2: return
    
                vals = Z.flatten()
                vmin = np.percentile(vals, 1)
                vmax = np.percentile(vals, 99)
    
                self.pcm_resid = self.ax_resid.pcolormesh(Xs, Ys, Z, shading='auto', cmap='jet',
                                                          vmin=vmin, vmax=vmax)
                
                self.ax_resid.set_title("Residuals")
                self.ax_resid.set_xlabel("Energy (eV)")
                self.ax_resid.set_ylabel("Delay (ps)")
                
                # --- APLICAR ESCALA Y (CONDICIONAL) ---
                if hasattr(self, 'yscale') and self.yscale == 'symlog':
                    self.ax_resid.set_yscale('symlog', linthresh=1.0)
                else:
                    self.ax_resid.set_yscale('linear')
                
                divider = make_axes_locatable(self.ax_resid)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                self.cbar_resid = self.canvas_resid.figure.colorbar(self.pcm_resid, cax=cax, label='Residual')
                self.canvas_resid.draw()
            except Exception as e:
                print(f"Error painting Resid: {e}")
# --- LOGICA DEL AJUSTE (PIPELINE) ---
 
    def run_fit_pipeline(self):
            try:
                if self.data_raw is None:
                    QMessageBox.warning(self, "No data", "Load data first.")
                    return
    
                # 1. Preview y Procesado
                self._preview_data_processing()
                if self.data_c is None or self.data_c.size == 0: return
    
                # 2. Configuración
                self.numExp = self.spin_numExp.value()
                self.tech = self.combo_tech.currentText()
                self.t0_choice = 'Yes' if self.chk_chirp.isChecked() else 'No'
                model_str = self.combo_model.currentText()
                self.model_type = "Sequential" if "Sequential" in model_str else "Parallel"
    
                # 3. GESTIÓN DE GUESSES (NUEVA LÓGICA)
                # Comprobamos si el vector actual self.ini es válido para la configuración actual
                # (Calculamos el tamaño esperado L)
                numWL = self.data_c.shape[0]
                if self.t0_choice == 'Yes': L_needed = 1 + self.numExp + numWL*(self.numExp+1)
                else:                       L_needed = 2 + self.numExp + numWL*self.numExp
                
                # Si no hay guesses o el tamaño no coincide (porque cambiaste numExp), regenerar
                if self.ini is None or len(self.ini) != L_needed:
                    print("Generating new default guesses...")
                    self._generate_defaults()
                else:
                    print("Using existing (possibly edited) guesses.")
    
                # 4. Ejecutar Ajuste
                self._temp_fit_TD = getattr(self, '_td_proc', self.TD)
                self._temp_fit_WL = getattr(self, '_wl_proc', self.WL)
                
                self._run_least_squares_with_progress()
                self._postprocess_fit_and_save()
    
            except Exception as e:
                QMessageBox.critical(self, "Fit Error", str(e))
                import traceback
                traceback.print_exc()

    def _open_guess_editor_and_update(self):
            """Abre la tabla de edición con ETIQUETAS DESCRIPTIVAS."""
            
            # 1. Regenerar si hace falta (Lógica anterior)
            numExp = self.spin_numExp.value()
            is_chirp = self.chk_chirp.isChecked()
            
            # Calcular longitud esperada
            if self.data_c is not None: numWL = self.data_c.shape[0]
            elif self.WL is not None: numWL = len(self.WL)
            else: numWL = 1
                
            if is_chirp: L_needed = 1 + numExp + numWL*(numExp+1)
            else:        L_needed = 2 + numExp + numWL*numExp
                
            if self.ini is None or len(self.ini) != L_needed:
                self._generate_defaults()
    
            # 2. Configurar Tabla
            L = len(self.ini)
            dlg = QDialog(self)
            dlg.setWindowTitle(f"Edit Initial Guesses ({L} parameters)")
            dlg.resize(700, 500)
            v = QVBoxLayout()
            

            
            table = QTableWidget(L, 5) # Cambiamos a 5 columnas
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            table.setHorizontalHeaderLabels(["Parameter", "Value", "Lower Bound", "Upper Bound", "Fix?"])
            
            # Si no existe la variable de fijos, la creamos (False por defecto)
            if not hasattr(self, 'is_fixed'):
                self.is_fixed = np.zeros(L, dtype=bool)
            elif len(self.is_fixed) != L: # Si cambió el modelo, resetear
                self.is_fixed = np.zeros(L, dtype=bool)
                
            # --- BUCLE DE LLENADO CON ETIQUETAS ---
            for i in range(L):
                
                # A) Generar descripción inteligente
                label = str(i)
                # Columna 4: Checkbox para fijar
                chk_item = QTableWidgetItem()
                chk_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                chk_item.setCheckState(Qt.Checked if self.is_fixed[i] else Qt.Unchecked)
                table.setItem(i, 4, chk_item)
                
                if not is_chirp:
                    # Modelo Estándar
                    if i == 0:
                        label += " (w - IRF Width)"
                    elif i == 1:
                        label += " (t0 - Time Zero)"
                    elif i < 2 + numExp:
                        # Los taus empiezan en el índice 2
                        # i=2 -> tau1, i=3 -> tau2...
                        label += f" (τ{i-1} - Lifetime)"
                    else:
                        label += " (Amplitude A_ik)"
                else:
                    # Modelo Chirp
                    if i == 0:
                        label += " (w - IRF Width)"
                    elif i < 1 + numExp:
                        # En chirp, no hay t0 global, así que los taus empiezan en i=1
                        label += f" (τ{i} - Lifetime)"
                    else:
                        label += " (Local Param: t0 or A)"
    
                # B) Poner items en la tabla
                # Columna 0: Etiqueta generada
                item_lbl = QTableWidgetItem(label)
                item_lbl.setFlags(item_lbl.flags() ^ Qt.ItemIsEditable) # Hacer solo lectura la etiqueta
                table.setItem(i, 0, item_lbl)
                
                # Columnas de valores
                table.setItem(i, 1, QTableWidgetItem(str(self.ini[i])))
                table.setItem(i, 2, QTableWidgetItem(str(self.limi[i])))
                table.setItem(i, 3, QTableWidgetItem(str(self.lims[i])))
                
            v.addWidget(table)
            
            # Botones (Igual que antes)
            btn_reset = QPushButton("Reset to Defaults")
            def reset_vals():
                self._generate_defaults()
                # Refrescar tabla
                for j in range(L):
                    table.setItem(j, 1, QTableWidgetItem(str(self.ini[j])))
                    table.setItem(j, 2, QTableWidgetItem(str(self.limi[j])))
                    table.setItem(j, 3, QTableWidgetItem(str(self.lims[j])))
            btn_reset.clicked.connect(reset_vals)
            v.addWidget(btn_reset)
    
            btn_ok = QPushButton("Save & Close")
            btn_ok.clicked.connect(dlg.accept)
            v.addWidget(btn_ok)
            
            dlg.setLayout(v)
            
            if dlg.exec_() == QDialog.Accepted:
                try:
                    for i in range(L):
                        self.ini[i] = float(table.item(i, 1).text())
                        self.limi[i] = float(table.item(i, 2).text())
                        self.lims[i] = float(table.item(i, 3).text())
                        # Guardar estado de "fijado"
                        self.is_fixed[i] = (table.item(i, 4).checkState() == Qt.Checked)
                except ValueError:
                    QMessageBox.warning(self, "Error", "Invalid number format.")
        

    def _run_least_squares_with_progress(self):
        
        TD = self._temp_fit_TD
        WL = self._temp_fit_WL
        numWL = len(WL)
        data_flat = self.data_c.T.flatten()
    
        if not hasattr(self, 'is_fixed') or len(self.is_fixed) != len(self.ini):
            self.is_fixed = np.zeros(len(self.ini), dtype=bool)
            
        free_indices = np.where(~self.is_fixed)[0]
        x0_free = self.ini[free_indices]
        low_free = self.limi[free_indices]
        upp_free = self.lims[free_indices]
    
        # --- LÓGICA DE PROGRESO ---
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Iterating: %v") # Muestra el número de iteración
        self.iter_count = 0 
    
        def residuals(p_free):
            # 1. Incrementar contador cada vez que se evalúa el modelo
            self.iter_count += 1
            
            # 2. Actualizar la barra cada N evaluaciones para no ralentizar demasiado
            if self.iter_count % 10 == 0:
                # Como no sabemos el total, hacemos que la barra "baile" (modo indefinido)
                # o simplemente mostramos el conteo de iteraciones
                val = (self.iter_count // 10) % 101
                self.progress_bar.setValue(val)
                
                # ¡ESTO ES LO MÁS IMPORTANTE! 
                # Fuerza a la interfaz a procesar los cambios de diseño y botones
                QApplication.processEvents()
    
            # Reconstrucción del modelo (tu lógica original)
            x_full = self.ini.copy()
            x_full[free_indices] = p_free
            
            if self.model_type == "Sequential":
                F = fit.eval_sequential_model(x_full, TD, self.numExp, numWL, self.t0_choice)
            else:
                F = fit.eval_global_model(x_full, TD, self.numExp, numWL, self.t0_choice)
            
            return F.flatten() - data_flat
    
        try:
            res = least_squares(
                fun=residuals,
                x0=x0_free,
                bounds=(low_free, upp_free),
                method='trf',
                verbose=0
            )
            
            self.fit_result = res
            self.fit_x = self.ini.copy()
            self.fit_x[free_indices] = res.x
            
            # Al terminar, ponemos la barra al 100%
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("Fit Completed")
            
        except Exception as e:
            self.progress_bar.setValue(0)
            raise e

    def _postprocess_fit_and_save(self):
        """Calcula estadísticas, extrae espectros con errores y guarda archivos en /fit/."""
        import fit
        import os
        import numpy as np
        from PyQt5.QtWidgets import QMessageBox
    
        if self.fit_result is None:
            return
    
        x = self.fit_x
        # Recuperar ejes procesados (los que realmente se usaron en el ajuste)
        TD = getattr(self, '_temp_fit_TD', self.TD)
        WL = getattr(self, '_temp_fit_WL', self.WL)
        
        if TD is None or WL is None:
            print("Error: No se encontraron los ejes (TD/WL) del ajuste.")
            return
    
        numWL = len(WL)
        numExp = self.numExp
    
        # --- 1. Reconstruir Matriz de Ajuste y Residuos ---
        if self.model_type == "Sequential":
            F_mat = fit.eval_sequential_model(x, TD, numExp, numWL, self.t0_choice)
        else:
            F_mat = fit.eval_global_model(x, TD, numExp, numWL, self.t0_choice)
            
        # F_mat suele salir como (numTD, numWL) o viceversa dependiendo de tu módulo fit.
        # Asumimos que queremos fitres como (numWL, numTD) para que coincida con data_c
        fitres = F_mat.T 
        resid = self.data_c - fitres
        
        self.fit_fitres = fitres
        self.fit_resid = resid
    
        # --- 2. Cálculo de Errores (CI) considerando parámetros FIJOS ---
        L_total = len(x)
        self.ci = np.zeros(L_total) # Por defecto, error 0 para todos (incluyendo fijos)
        
        try:
            # Identificar índices que NO están fijos
            if not hasattr(self, 'is_fixed'):
                self.is_fixed = np.zeros(L_total, dtype=bool)
            
            free_indices = np.where(~self.is_fixed)[0]
            J = self.fit_result.jac # El jacobiano de least_squares solo contiene columnas libres
            
            if J is not None and J.size > 0:
                # Matriz de covarianza reducida
                cov_free = np.linalg.inv(J.T @ J)
                # Grados de libertad = Total puntos datos - Total parámetros libres
                dof = resid.size - len(free_indices)
                mse = np.sum(resid**2) / dof
                err_free = np.sqrt(np.maximum(np.diagonal(cov_free * mse), 0))
                
                # Mapear errores calculados a sus posiciones en el vector global
                self.ci[free_indices] = err_free
                
        except Exception as e:
            print(f"Advertencia en Covarianza: {e}. Los errores podrían ser 0.")
    
        # --- 3. Extraer Taus y sus Errores ---
        idx_tau = 1 if self.t0_choice == 'Yes' else 2
        self.extracted_taus = x[idx_tau : idx_tau + numExp]
        self.extracted_errtaus = self.ci[idx_tau : idx_tau + numExp]
    
        # --- 4. Extraer Amplitudes y sus Errores (DAS/SAS) ---
        self.As = np.zeros((numExp, numWL))
        self.errAs = np.zeros((numExp, numWL))
        
        try:
            if self.t0_choice == 'No':
                base_A = 2 + numExp
                # Reshape: de vector plano a matriz (numWL, numExp) y transponemos a (numExp, numWL)
                self.As = x[base_A:].reshape(numWL, numExp).T
                self.errAs = self.ci[base_A:].reshape(numWL, numExp).T
                self.t0s = np.full(numWL, x[1])
            else:
                # Lógica para Chirp/Independent t0 si fuera necesario
                pass
        except Exception as e:
            print(f"Error extrayendo amplitudes: {e}")
    
        # --- 5. Lógica de Guardado ---
        base_dir = self.base_dir
        outdir = os.path.join(base_dir, "fit")
        os.makedirs(outdir, exist_ok=True)
    
        try:
            # A) Guardar binario para recarga
            np.save(os.path.join(outdir, "GFitResults.npy"), {
                "taus": self.extracted_taus,
                "err_taus": self.extracted_errtaus,
                "As": self.As,
                "errAs": self.errAs,
                "WL": WL,
                "TD": TD,
                "fitres": fitres,
                "resid": resid
            })
    
            # B) Guardar archivos de texto planos
            np.savetxt(os.path.join(outdir, "WL.txt"), WL, fmt='%.6f', header="Wavelength (nm)")
            np.savetxt(os.path.join(outdir, "TD.txt"), TD, fmt='%.6f', header="Time Delay (ps)")
            
            # C) Amplitudes con formato de tabla
            with open(os.path.join(outdir, "Amplitudes.txt"), 'w') as f:
                header_list = [f"A{i+1}\tErrA{i+1}" for i in range(numExp)]
                f.write("WL(nm)\t" + "\t".join(header_list) + "\n")
                for i in range(numWL):
                    line_data = [f"{WL[i]:.2f}"]
                    for j in range(numExp):
                        line_data.append(f"{self.As[j, i]:.6e}")
                        line_data.append(f"{self.errAs[j, i]:.6e}")
                    f.write("\t".join(line_data) + "\n")
     
            print(f"Resultados exportados exitosamente a: {outdir}")
    
        except Exception as e:
            print(f"Error crítico guardando archivos: {e}")
    
        # Actualizar Interfaz
        self._update_fit_canvas()
        self._update_resid_canvas()
        self.btn_show_das.setEnabled(True)
    
        rmsd = np.sqrt(np.mean(resid**2))
        QMessageBox.information(self, "Ajuste Finalizado", 
                                f"Optimización completada.\nRMSD: {rmsd:.2e}\nDatos guardados en /fit/")
    def plot_das_and_more(self):
            """
            Abre ventana externa con DAS/SAS (guarda exactamente lo que muestra)
            y permite chequear trazas individuales guardándolas en el formato solicitado.
            """
            if self.As is None: return
    
            # Definir directorio de salida para plots
            outdir = os.path.join(self.base_dir, "Plots")
            os.makedirs(outdir, exist_ok=True)
    
            wl = getattr(self, '_wl_proc', self.WL)
            td = getattr(self, '_td_proc', self.TD)
    

            fig_das = plt.figure(figsize=(8, 5))
            ax = fig_das.gca()
    
            colors = ['b', 'r', 'g', 'orange', 'm', 'c']
    
            for n in range(self.numExp):
                tau_val = self.extracted_taus[n]
                # Verificar si existe error y no es NaN
                if self.extracted_errtaus is not None and n < len(self.extracted_errtaus):
                     err_tau = self.extracted_errtaus[n]
                     if np.isnan(err_tau): err_tau = 0.0
                else:
                     err_tau = 0.0
    
                lbl = f"τ{n+1} = {tau_val:.2f} ± {err_tau:.2f} ps"
                color = colors[n % len(colors)]
    
                # 1. Línea principal
                ax.plot(wl, self.As[n], label=lbl, color=color, linewidth=2)
    
                # 2. Sombra de error (Si existe)
                if self.errAs is not None:
                    lower = np.nan_to_num(self.As[n] - self.errAs[n])
                    upper = np.nan_to_num(self.As[n] + self.errAs[n])
                    ax.fill_between(wl, lower, upper, color=color, alpha=0.2)
    
            ax.set_xlabel("Energy (eV)")
            if self.model_type == "Sequential":
                ax.set_ylabel("SAS (Concentration)")
                ax.set_title("Species Associated Spectra (SAS)")
                savename = "SAS.png"
            else:
                ax.set_ylabel("DAS (Amplitude)")
                ax.set_title("Decay Associated Spectra (DAS)")
                savename = "DAS.png"
    
            ax.legend()
            ax.axhline(0, color='k', linestyle='--', alpha=0.5)
            ax.grid(True, linestyle=':', alpha=0.4)
            fig_das.tight_layout()
    
            try:
                fig_das.savefig(os.path.join(outdir, savename), dpi=300)
                print(f"DAS/SAS plot saved to {outdir}")
            except Exception as e:
                print(f"Error saving DAS plot: {e}")
    
            fig_das.show()
    
            fig_res, ax_res = plt.subplots()
            # Nota: Usamos fit_resid.T para que coincida con dimensiones (WL, TD) del pcolormesh
            pcm = ax_res.pcolormesh(wl, td, self.fit_resid.T, cmap='jet', shading='auto')
            fig_res.colorbar(pcm, ax=ax_res, label='Residuals')
            ax_res.set_title("Residuals Map")
            ax_res.set_xlabel("Energy (eV)")
            ax_res.set_ylabel("Delay (ps)")
            # Aplicar escala log si estaba seleccionada en la GUI
            if hasattr(self, 'yscale') and self.yscale == 'symlog':
                 ax_res.set_yscale('symlog', linthresh=1.0)
            fig_res.tight_layout()
            fig_res.savefig(os.path.join(outdir, "Residuals_Map.png"), dpi=300)
            plt.close(fig_res)
    
            cont = True
            while cont:
                text_default = f"{wl[len(wl)//2]:.1f}"
                wl_str, ok = QInputDialog.getText(self, "Check Trace", 
                                                  f"Enter wavelength nm ({wl.min():.1f}-{wl.max():.1f}):", 
                                                  text=text_default)
                if not ok: break
    
                try:
                    target_wl = float(wl_str)
                    idx = np.argmin(np.abs(wl - target_wl))
                    real_wl = wl[idx]
    
                    y_exp = self.data_c[idx, :]
                    y_fit = self.fit_fitres[idx, :]
    
                    # Crear la figura con los dos paneles (Lineal y Log)
                    fig_trace, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
                    fig_trace.suptitle(f"Fit at {real_wl:.1f} nm", fontsize=14)
    
                    # Panel Izquierdo: Lineal
                    ax1.plot(td, y_exp, 'bo', markersize=4, alpha=0.6, label='Data')
                    ax1.plot(td, y_fit, 'r-', linewidth=2, label='Fit')
                    ax1.set_xlabel("Time / ps")
                    ax1.set_ylabel("ΔA")
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
    
                    # Panel Derecho: Logarítmico
                    mask_pos = td > 0
                    if np.any(mask_pos):
                        ax2.plot(td[mask_pos], y_exp[mask_pos], 'bo', markersize=4, alpha=0.6)
                        ax2.plot(td[mask_pos], y_fit[mask_pos], 'r-', linewidth=2)
                        ax2.set_xscale('log')
                        ax2.set_xlabel("Time / ps (log scale)")
                        ax2.grid(True, which="both", ls="-", alpha=0.3)
    
                    plt.tight_layout()
    
                    plt.show(block=True) 
    
                    # --- ESTO SE EJECUTA SOLO DESPUÉS DE CERRAR EL GRÁFICO ---
                    resp = QMessageBox.question(self, "Save Trace?",
                                                f"¿Deseas guardar los archivos de la traza a {real_wl:.1f} nm?",
                                                QMessageBox.Yes | QMessageBox.No)
    
                    if resp == QMessageBox.Yes:
                        # 1. Guardar Imagen
                        img_name = f"Trace_{real_wl:.1f}nm.png"
                        fig_trace.savefig(os.path.join(outdir, img_name), dpi=300)
    
                        # 2. Guardar TXT con el formato de tu imagen
                        txt_name = f"Fit_{real_wl:.1f}nm.txt"
                        txt_path = os.path.join(outdir, txt_name)
                        
                        data_stack = np.column_stack((td, y_exp, y_fit))
                        
                        # Definimos la cabecera con el # inicial como en tu Bloc de notas
                        header_txt = "TD(ps)\tExp(A)\tFit(A)"
                        
                        np.savetxt(txt_path, data_stack, fmt='%1.6e', delimiter='\t',
                                   header=header_txt, comments='# ') # Mantiene el estilo de la imagen
    
                    plt.close(fig_trace) # Limpiamos la memoria de la figura
    
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Error al procesar la traza: {e}")
    
                # Preguntar si ver otra longitud de onda
                if QMessageBox.question(self, "Continuar", "¿Ver otra traza?", 
                                        QMessageBox.Yes|QMessageBox.No) == QMessageBox.No:
                    cont = False
if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainApp()
    
    window.show()

    sys.exit(app.exec_())