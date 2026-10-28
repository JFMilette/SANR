"""pyqtgraph's CSV export of the SLD profile plots (right-click -> Export):
the Simulation tab's, a fit tab's and DREAM's give their smooth curves,
each column named, and a last column naming the layer of each row
(plot_export); other plots export as pyqtgraph's own exporter does.

Run from the repo root: python -m pytest tests
"""
import os
import pathlib
import sys
import warnings

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

QtWidgets = pytest.importorskip('PyQt6.QtWidgets')
APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

import pyqtgraph.exporters as E                                  # noqa: E402
import plot_export                                               # noqa: E402
from tabs import SimulationTab                                   # noqa: E402
from fit.de.de_window import DEWindow                            # noqa: E402
from fit.dream.dream_window import ProfilePanel                  # noqa: E402
from fit.problem import FitProblem                               # noqa: E402
from test_thickness_spread import data_of, film                  # noqa: E402

CURVES = ['NSLD_x', 'NSLD_y', 'MSLD_rho_x', 'MSLD_rho_y', 'MSLD_phi_deg_x',
          'MSLD_phi_deg_y', 'MSLD_theta_deg_x', 'MSLD_theta_deg_y', 'layer']


def export(plot, tmp_path):
    """(header, number of rows, the rows) of the export the dialog's
    'CSV of original plot data' entry writes."""
    name = E.CSVExporter.Name
    cls = [c for c in E.listExporters() if c.Name == name]
    assert cls == [plot_export.LayerCSVExporter]       # in place of it
    out = tmp_path / 'profile.csv'
    cls[0](plot).export(str(out))
    lines = out.read_text().splitlines()
    rows = [l.split(',') for l in lines[1:]]
    return lines[0].split(','), len(rows), rows


def check_layers(rows, Z, names):
    """Each row's layer is the one its depth (first column) is in."""
    z = np.array([float(r[0]) for r in rows])
    want = [names[i] for i in np.searchsorted(Z, z)]
    assert [r[-1] for r in rows] == want
    assert set(want) == set(names)


def test_simulation_and_fit_tab_profiles(tmp_path):
    sim = SimulationTab()
    sim.recompute()
    APP.processEvents()
    head, n, rows = export(sim.prof_plot, tmp_path)
    assert head == CURVES and n > 100
    check_layers(rows, sim._profile[1], [l.name for l in sim.stack.layers])
    # another plot: pyqtgraph's own columns, no layer
    head, _, _ = export(sim.refl_plot, tmp_path)
    assert 'layer' not in head
    st = film()
    st.layers[1].fit['thickness'] = dict(vary=True, min=100, max=140)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p = FitProblem(st, data_of(film()))
    win = DEWindow()
    win.start(p, [dict(name='d', data={}, vectors={},
                       grids={None: np.linspace(0.01, 0.2, 50)})], 'd', 'χ²')
    view = win.views[0]
    head, n, rows = export(view.prof_plot, tmp_path)
    assert head == CURVES and n > 100          # it was empty
    assert n == len(view.prof_curves[0].getData()[0])
    check_layers(rows, view._profile[1], [l.name for l in view.stack.layers])
    win.close()


def test_dream_profile_bands(tmp_path):
    pp = ProfilePanel()
    z = np.linspace(-50, 300, 400)
    b = np.array([np.sin(z / 50) + k for k in (-0.2, -0.1, 0, 0.1, 0.2)])
    Z, names = [0.0, 120.0, 250.0], ['front', 'a', 'b', 'Si']
    pp.show_bands(z, b * 1e-6, b * 0.5e-6, layers=(Z, names))
    head, n, rows = export(pp.plot, tmp_path)
    assert n == len(z) and head[-1] == 'layer'
    check_layers(rows, Z, names)
    for name in ('NSLD 95% low_y', 'NSLD 68% high_y', 'NSLD_y',
                 'MSLD in plane 95% high_y'):
        assert name in head
    # the band edges have no legend entry
    assert [l.text for _, l in pp.plot.legend.items] == \
        ['NSLD', 'MSLD in plane (ρ cos θ)']
