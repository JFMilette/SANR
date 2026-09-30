"""
Application-wide dark theme: Fusion style, a matching QPalette, a stylesheet
and the pyqtgraph defaults.  Call apply(app) once before building windows.
"""

import os

import pyqtgraph as pg
from PyQt6 import QtGui

BG = '#1e1f22'          # window
SURFACE = '#2b2d31'     # group boxes, headers
INPUT = '#1b1c1f'       # line edits, spin boxes, lists
PLOT_BG = '#1b1c1f'
BORDER = '#3f4147'
BORDER_STRONG = '#55585f'
HOVER = '#393b41'
TEXT = '#e3e5e8'
MUTED = '#9aa0a6'
DISABLED = '#62666d'
ACCENT = '#4c8dff'
ACCENT_SOFT = '#2a3b5c'
ERROR = '#f87171'

ICONS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'icons').replace(os.sep, '/')

STYLE = f"""
QToolTip {{
    background: {SURFACE}; color: {TEXT}; border: 1px solid {BORDER};
    padding: 4px 6px;
}}

QTabWidget::pane {{ border: none; }}
QTabBar::tab {{
    background: transparent; padding: 8px 18px; margin-right: 2px;
    border: none; border-bottom: 2px solid transparent; color: {MUTED};
}}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QTabBar::tab:hover:!selected {{ color: {TEXT}; }}

QGroupBox {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px;
    margin-top: 22px; padding: 10px 8px 8px 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 6px; top: 2px; padding: 0 4px;
    color: {MUTED}; font-weight: 600;
}}

QPushButton {{
    background: {HOVER}; color: {TEXT}; border: 1px solid {BORDER};
    border-radius: 6px; padding: 5px 12px;
}}
QPushButton:hover {{ background: #43464d; border-color: {BORDER_STRONG}; }}
QPushButton:pressed {{ background: {BORDER}; }}
QPushButton:default {{ background: {ACCENT}; color: white; border-color: {ACCENT}; }}
QPushButton:disabled {{ color: {DISABLED}; background: {SURFACE}; border-color: {BORDER}; }}

QLineEdit, QAbstractSpinBox, QComboBox, QPlainTextEdit, QTextEdit, QListWidget {{
    background: {INPUT}; color: {TEXT}; border: 1px solid {BORDER};
    border-radius: 5px; padding: 2px 4px;
    selection-background-color: {ACCENT}; selection-color: white;
}}
QLineEdit:focus, QAbstractSpinBox:focus, QComboBox:focus,
QPlainTextEdit:focus, QTextEdit:focus {{ border: 1px solid {ACCENT}; }}
QLineEdit:disabled, QAbstractSpinBox:disabled, QComboBox:disabled {{
    color: {DISABLED};
}}

QAbstractSpinBox {{ padding-right: 18px; }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
    subcontrol-origin: border; width: 16px; border: none;
    background: transparent;
}}
QAbstractSpinBox::up-button {{
    subcontrol-position: top right; border-top-right-radius: 5px;
}}
QAbstractSpinBox::down-button {{
    subcontrol-position: bottom right; border-bottom-right-radius: 5px;
}}
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{
    background: {HOVER};
}}
QAbstractSpinBox::up-arrow {{ image: url({ICONS}/up.svg); width: 9px; height: 9px; }}
QAbstractSpinBox::down-arrow {{ image: url({ICONS}/down.svg); width: 9px; height: 9px; }}
QAbstractSpinBox::up-arrow:disabled, QAbstractSpinBox::up-arrow:off {{
    image: url({ICONS}/up_off.svg);
}}
QAbstractSpinBox::down-arrow:disabled, QAbstractSpinBox::down-arrow:off {{
    image: url({ICONS}/down_off.svg);
}}

QComboBox {{ padding-right: 18px; }}
QComboBox::drop-down {{
    subcontrol-origin: padding; subcontrol-position: center right;
    width: 18px; border: none;
}}
QComboBox::down-arrow {{ image: url({ICONS}/down.svg); width: 9px; height: 9px; }}
QComboBox::down-arrow:disabled {{ image: url({ICONS}/down_off.svg); }}
QComboBox QAbstractItemView {{
    background: {SURFACE}; color: {TEXT}; border: 1px solid {BORDER};
    selection-background-color: {ACCENT_SOFT}; outline: none;
}}

QCheckBox {{ spacing: 6px; }}
QCheckBox::indicator {{
    width: 14px; height: 14px; border: 1px solid {BORDER_STRONG};
    border-radius: 4px; background: {INPUT};
}}
QCheckBox::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked {{
    background: {ACCENT}; border-color: {ACCENT};
    image: url({ICONS}/check.svg);
}}
QCheckBox::indicator:disabled {{ background: {SURFACE}; border-color: {BORDER}; }}

QTableWidget, QTreeWidget {{
    background: {INPUT}; color: {TEXT}; border: 1px solid {BORDER};
    border-radius: 6px; gridline-color: {BORDER};
    alternate-background-color: {SURFACE};
    selection-background-color: {ACCENT_SOFT}; selection-color: {TEXT};
}}
QListWidget::item:selected, QTreeWidget::item:selected {{
    background: {ACCENT_SOFT}; color: {TEXT};
}}
QHeaderView::section {{
    background: {SURFACE}; color: {MUTED}; border: none;
    border-bottom: 1px solid {BORDER}; padding: 4px 6px; font-weight: 600;
}}
QTableCornerButton::section {{ background: {SURFACE}; border: none; }}

QToolBox::tab {{
    background: {SURFACE}; border-radius: 5px; padding: 5px 8px; color: {MUTED};
}}
QToolBox::tab:selected {{ background: {ACCENT_SOFT}; color: {TEXT}; font-weight: 600; }}

QSplitter::handle {{ background: {BORDER}; }}
QSplitter::handle:horizontal {{ width: 3px; }}
QSplitter::handle:vertical {{ height: 3px; }}

QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {ACCENT}; width: 14px; margin: -5px 0; border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{ background: #6ea3ff; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle {{ background: {BORDER_STRONG}; border-radius: 4px; margin: 2px; }}
QScrollBar::handle:hover {{ background: #6b6f77; }}
QScrollBar::handle:vertical {{ min-height: 24px; }}
QScrollBar::handle:horizontal {{ min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QMenuBar {{ background: {BG}; color: {TEXT}; }}
QMenuBar::item:selected {{ background: {HOVER}; }}
QMenu {{ background: {SURFACE}; color: {TEXT}; border: 1px solid {BORDER}; padding: 4px; }}
QMenu::item {{ padding: 5px 20px; border-radius: 4px; }}
QMenu::item:selected {{ background: {ACCENT_SOFT}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 6px; }}
"""


def palette():
    """Dark palette for whatever the stylesheet does not cover (dialogs,
    item views, text colours)."""
    P = QtGui.QPalette
    R = P.ColorRole
    pal = P()
    for role, col in ((R.Window, BG), (R.WindowText, TEXT), (R.Base, INPUT),
                      (R.AlternateBase, SURFACE), (R.Text, TEXT),
                      (R.Button, SURFACE), (R.ButtonText, TEXT),
                      (R.Highlight, ACCENT), (R.HighlightedText, '#ffffff'),
                      (R.ToolTipBase, SURFACE), (R.ToolTipText, TEXT),
                      (R.PlaceholderText, MUTED), (R.BrightText, ERROR),
                      (R.Link, ACCENT), (R.Mid, BORDER), (R.Dark, INPUT),
                      (R.Light, BORDER_STRONG), (R.Shadow, '#000000')):
        pal.setColor(role, QtGui.QColor(col))
    for role in (R.WindowText, R.Text, R.ButtonText):
        pal.setColor(P.ColorGroup.Disabled, role, QtGui.QColor(DISABLED))
    return pal


def apply(app):
    pg.setConfigOptions(antialias=True, background=PLOT_BG, foreground=MUTED)
    app.setStyle('Fusion')
    app.setPalette(palette())
    app.setStyleSheet(STYLE)
