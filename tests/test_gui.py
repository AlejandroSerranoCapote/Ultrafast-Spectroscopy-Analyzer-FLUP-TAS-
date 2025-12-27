# -*- coding: utf-8 -*-
"""
GitHub Tests
"""

from core_analysis import eV_a_nm
import pytest

def test_conversion_energia():
    # Si 1240 es la constante común, verificamos una conversión conocida
    resultado = eV_a_nm(1.0) 
    assert round(resultado, 1) == 1239.8 # O el valor que esperes
    
def test_app_starts(qtbot):
    from Ultrafast_Spectroscopy_Analyzer import MainApp
    widget = MainApp()
    qtbot.addWidget(widget)
    assert widget.windowTitle() == "Data Analyzer Selector"    