"""
Application-wide theme (dark or light): Fusion style, a matching QPalette, a
stylesheet and the pyqtgraph defaults.  Call apply(app) once before building
windows; apply(app, 'light') later switches the running app in place.

The colour names below (BG, TEXT, ...) always hold the current theme's
values: read them as style.TEXT when drawing, not with `from style import`,
so a later switch is seen.  Widgets that bake a colour in when they are built
connect to `notifier().changed` to redraw.
"""

import os

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt6 import QtCore, QtGui, sip

THEMES = {
    'dark': dict(
        BG='#1e1f22',           # window
        SURFACE='#2b2d31',      # group boxes, headers
        INPUT='#1b1c1f',        # line edits, spin boxes, lists
        PLOT_BG='#1b1c1f',
        BORDER='#3f4147',
        BORDER_STRONG='#55585f',
        HOVER='#393b41',
        BUTTON_HOVER='#43464d',
        SCROLL_HOVER='#6b6f77',
        TEXT='#e3e5e8',
        MUTED='#9aa0a6',
        DISABLED='#62666d',
        ACCENT='#4c8dff',
        ACCENT_HOVER='#6ea3ff',
        ACCENT_SOFT='#2a3b5c',
        ERROR='#f87171',
        ERROR_BG='#4a1f22',     # out-of-bounds spin boxes
        BAND='#ffffff0c',       # alternate layer shading in the profile
        SELECT='#b48cff',       # the selected layer in the profile: violet,
                                # none of the profile curves' colours
        START_LINE='#5a4a1c',   # data preview: first line read ...
        START_REST='#34301f',   # ... and the lines after it
        BEAM='#d0d4da',         # geometry tab: k_i, k_f
        DENSITY_TOP='#dce8ff',  # DREAM 2-D histograms: densest bins
        ICON='',                # arrow icon suffix
    ),
    'light': dict(
        BG='#f3f4f6',
        SURFACE='#ffffff',
        INPUT='#ffffff',
        PLOT_BG='#ffffff',
        BORDER='#d5d8dd',
        BORDER_STRONG='#b5b9c0',
        HOVER='#e9ebef',
        BUTTON_HOVER='#dfe2e7',
        SCROLL_HOVER='#8f949c',
        TEXT='#1f2328',
        MUTED='#5f6368',
        DISABLED='#a0a4ab',
        ACCENT='#2f6fe0',
        ACCENT_HOVER='#5089ec',
        ACCENT_SOFT='#d7e4fc',
        ERROR='#d93025',
        ERROR_BG='#fde2e1',
        BAND='#0000000a',
        SELECT='#7c3aed',
        START_LINE='#ffe08a',
        START_REST='#fff4cc',
        BEAM='#3c4043',
        DENSITY_TOP='#0b2a66',
        ICON='_light',
    ),
}
DEFAULT = 'dark'

theme = DEFAULT
globals().update(THEMES[DEFAULT])

ICONS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'icons').replace(os.sep, '/')


class _Notifier(QtCore.QObject):
    changed = QtCore.pyqtSignal(str)        # the new theme's name


_notifier = None


def notifier():
    """The object whose `changed` signal follows apply(); made again when
    Qt has deleted it (it goes with the QApplication it was made under)."""
    global _notifier
    if _notifier is None or sip.isdeleted(_notifier):
        _notifier = _Notifier()
    return _notifier


def stylesheet():
    return f"""
QToolTip {{
    background: {SURFACE}; color: {TEXT}; border: 1px solid {BORDER};
    padding: 4px 6px;
}}

QLabel[role="muted"] {{ color: {MUTED}; }}

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
QPushButton:hover {{ background: {BUTTON_HOVER}; border-color: {BORDER_STRONG}; }}
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
QAbstractSpinBox::up-arrow {{ image: url({ICONS}/up{ICON}.svg); width: 9px; height: 9px; }}
QAbstractSpinBox::down-arrow {{ image: url({ICONS}/down{ICON}.svg); width: 9px; height: 9px; }}
QAbstractSpinBox::up-arrow:disabled, QAbstractSpinBox::up-arrow:off {{
    image: url({ICONS}/up_off{ICON}.svg);
}}
QAbstractSpinBox::down-arrow:disabled, QAbstractSpinBox::down-arrow:off {{
    image: url({ICONS}/down_off{ICON}.svg);
}}

QComboBox {{ padding-right: 18px; }}
QComboBox::drop-down {{
    subcontrol-origin: padding; subcontrol-position: center right;
    width: 18px; border: none;
}}
QComboBox::down-arrow {{ image: url({ICONS}/down{ICON}.svg); width: 9px; height: 9px; }}
QComboBox::down-arrow:disabled {{ image: url({ICONS}/down_off{ICON}.svg); }}
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
QSlider::handle:horizontal:hover {{ background: {ACCENT_HOVER}; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle {{ background: {BORDER_STRONG}; border-radius: 4px; margin: 2px; }}
QScrollBar::handle:hover {{ background: {SCROLL_HOVER}; }}
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
    """Palette for whatever the stylesheet does not cover (dialogs, item
    views, text colours)."""
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


def out_of_bounds():
    """Stylesheet of a spin box holding a value outside its fit bounds."""
    return 'QDoubleSpinBox { background: %s; border: 1px solid %s }' % (
        ERROR_BG, ERROR)


def _restyle_plots(app):
    """Give the plots and 3-D views already built the current colours:
    background, axes and legend text (the rest is drawn by their owners,
    which listen to notifier().changed)."""
    for w in app.allWidgets():
        if isinstance(w, gl.GLViewWidget):
            w.setBackgroundColor(PLOT_BG)
        elif isinstance(w, pg.GraphicsView):
            w.setBackground(PLOT_BG)
            for item in w.scene().items():
                if isinstance(item, pg.AxisItem):
                    item.setPen()               # current foreground
                    item.setTextPen()
                elif isinstance(item, pg.LegendItem):
                    item.setLabelTextColor(MUTED)


def apply(app, name=None):
    """Apply theme `name` ('dark' or 'light'; default: the current one)."""
    global theme
    theme = name if name in THEMES else theme
    globals().update(THEMES[theme])
    pg.setConfigOptions(antialias=True, background=PLOT_BG, foreground=MUTED)
    app.setStyle('Fusion')
    app.setPalette(palette())
    app.setStyleSheet(stylesheet())
    _restyle_plots(app)
    notifier().changed.emit(theme)
