"""The mouse wheel never changes a spin box or a combo box; it scrolls the
panel instead.

Run from the repo root: python -m pytest tests
"""
import os
import pathlib
import sys

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

QtWidgets = pytest.importorskip('PyQt6.QtWidgets')
from PyQt6 import QtCore, QtGui                                 # noqa: E402
from main import NoWheelOnInputs                                # noqa: E402


@pytest.fixture(scope='module')
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def wheel(widget, dy=-120):
    """Send a wheel turn (down by default) over the middle of `widget`;
    returns the event."""
    pos = QtCore.QPointF(widget.rect().center())
    ev = QtGui.QWheelEvent(
        pos, QtCore.QPointF(widget.mapToGlobal(pos.toPoint())),
        QtCore.QPoint(), QtCore.QPoint(0, dy),
        QtCore.Qt.MouseButton.NoButton, QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.Qt.ScrollPhase.NoScrollPhase, False)
    QtWidgets.QApplication.sendEvent(widget, ev)
    return ev


def test_wheel_leaves_spin_boxes_alone(app):
    """The value stays and the event comes back ignored: Qt then hands a
    real (spontaneous) wheel event to the parent, which scrolls.  A test's
    synthetic event is never handed on, so that part is not checked here."""
    boxes = [QtWidgets.QSpinBox(), QtWidgets.QDoubleSpinBox()]
    for b in boxes:
        b.setRange(0, 100)
        b.setValue(50)
        b.show()
    app.processEvents()
    f = NoWheelOnInputs(app)
    app.installEventFilter(f)
    try:
        for b in boxes:
            b.setFocus()
            assert not wheel(b).isAccepted()
            assert b.value() == 50
        area = QtWidgets.QScrollArea()          # other widgets still scroll
        page = QtWidgets.QWidget()
        page.setMinimumHeight(2000)
        area.setWidget(page)
        area.resize(200, 150)
        area.show()
        app.processEvents()
        wheel(area.viewport())
        assert area.verticalScrollBar().value() > 0
    finally:
        app.removeEventFilter(f)
    wheel(boxes[0])                             # without it, the box moves
    assert boxes[0].value() != 50


def test_wheel_leaves_combo_boxes_alone(app):
    """Same for combo boxes, editable or not."""
    boxes = []
    for editable in (False, True):
        b = QtWidgets.QComboBox()
        b.addItems(['a', 'b', 'c'])
        b.setEditable(editable)
        b.setCurrentIndex(1)
        b.show()
        boxes.append(b)
    app.processEvents()
    f = NoWheelOnInputs(app)
    app.installEventFilter(f)
    try:
        for b in boxes:
            b.setFocus()
            assert not wheel(b).isAccepted()
            assert b.currentIndex() == 1
    finally:
        app.removeEventFilter(f)
    wheel(boxes[0])                             # without it, the box moves
    assert boxes[0].currentIndex() != 1
