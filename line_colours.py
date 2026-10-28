"""
Line colours of the reflectivity plots: one per channel (R++, R+-, R-+, R--
and the R+, R- of polarized without analysis), one for the non-polarized
(R+ + R-)/2 and one for a single-curve asymmetry, shared
by the Simulation and Experimental tabs and the DE and DREAM windows.

Read a colour with get(key) when drawing; set_colours() changes them and
emits notifier().changed, to which the plots that hold a colour connect to
redraw.  The main window keeps them between launches (QSettings).  A
dataset of the Experimental tab can override them with its own colours
(tabs.experimental COLOURS).

button() is a "Colours…" push button for a plot's bar; it opens the one
LineColoursDialog, whose changes apply as they are made.
"""

from PyQt6 import QtCore, QtGui, QtWidgets, sip

# key, label in the dialog
KEYS = [('++', 'R++'), ('+-', 'R+−'), ('-+', 'R−+'), ('--', 'R−−'),
        ('+', 'R+ (without analysis)'), ('-', 'R− (without analysis)'),
        ('unpol', '(R+ + R−)/2 (non-polarized)'), ('asym', 'Asymmetry')]
DEFAULTS = {'++': '#1f77b4', '+-': '#d62728', '-+': '#2ca02c',
            '--': '#ff7f0e', '+': '#1f77b4', '-': '#ff7f0e',
            'unpol': '#17becf', 'asym': '#b392f0'}

_current = dict(DEFAULTS)


class _Notifier(QtCore.QObject):
    changed = QtCore.pyqtSignal()


_notifier = None
_dialog = None


def notifier():
    """The object whose `changed` signal follows set_colours(); made again
    when Qt has deleted it, as style.notifier."""
    global _notifier
    if _notifier is None or sip.isdeleted(_notifier):
        _notifier = _Notifier()
    return _notifier


def get(key, fallback='#9aa0a6'):
    """Colour of a channel ('++', '+', ...) or of 'asym'."""
    return _current.get(key, fallback)


def colours():
    return dict(_current)


def set_colours(d):
    """Take the valid colours of d {key: '#rrggbb'}; others keep theirs."""
    for k, v in d.items():
        if k in DEFAULTS and QtGui.QColor(v).isValid():
            _current[k] = QtGui.QColor(v).name()
    notifier().changed.emit()


class LineColoursDialog(QtWidgets.QDialog):
    """A swatch per colour (click: a colour picker) and Defaults."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Line colours')
        f = QtWidgets.QFormLayout(self)
        self.swatches = {}
        for key, label in KEYS:
            b = QtWidgets.QPushButton()
            b.setFixedWidth(110)
            b.clicked.connect(lambda _, k=key: self._pick(k))
            self.swatches[key] = b
            f.addRow(label, b)
        row = QtWidgets.QHBoxLayout()
        defaults = QtWidgets.QPushButton('Defaults')
        defaults.clicked.connect(lambda: set_colours(DEFAULTS))
        close = QtWidgets.QPushButton('Close')
        close.clicked.connect(self.close)
        row.addWidget(defaults)
        row.addStretch(1)
        row.addWidget(close)
        f.addRow(row)
        notifier().changed.connect(self._show)
        self._show()

    def _show(self):
        for key, b in self.swatches.items():
            c = QtGui.QColor(get(key))
            text = 'black' if c.lightness() > 140 else 'white'
            b.setText(c.name())
            b.setStyleSheet('QPushButton { background: %s; color: %s }'
                            % (c.name(), text))

    def _pick(self, key):
        label = dict(KEYS)[key]
        c = QtWidgets.QColorDialog.getColor(QtGui.QColor(get(key)), self,
                                            'Colour of %s' % label)
        if c.isValid():
            set_colours({key: c.name()})


def show_dialog(parent=None):
    """Open (or raise) the one Line colours dialog."""
    global _dialog
    if _dialog is None or sip.isdeleted(_dialog):
        _dialog = LineColoursDialog(parent)
    _dialog.show()
    _dialog.raise_()
    _dialog.activateWindow()


def button():
    b = QtWidgets.QPushButton('Colours…')
    b.setToolTip('Line colours of the channels in the reflectivity plots '
                 '(shared by every window)')
    b.clicked.connect(lambda: show_dialog(b.window()))
    return b
