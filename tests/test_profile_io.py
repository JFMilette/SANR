"""Import of a layer-profile table (model.profile_io, File -> Import
profile).

Run from the repo root: python -m pytest tests
"""
import os
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from model.profile_io import PROFILE_SUBLAYERS, load_profile    # noqa: E402

PROFILE = """\
#Depth Thickness Re_NSLD Im_NSLD MSLD_rho MSLD_phi MSLD_theta Roughness
0 29.3667 1.02881e-006 0 0 0 0 9.99939
29.3667 22.2234 5.4502e-006 0 0 0 0 9.66238
51.5901 49.6781 2.20792e-006 0 2.98725e-007 30 90 5.25621
220.055 0 5.86024e-006 0 0 0 0 21.2628
"""


def test_load_profile(tmp_path):
    f = tmp_path / 'profile.txt'
    f.write_text(PROFILE)
    layers = load_profile(str(f))
    assert [l.name for l in layers] == ['vacuum', 'Layer 1', 'Layer 2',
                                        'Layer 3', 'substrate']
    mag = layers[3]
    assert (mag.thickness, mag.NSLD_real, mag.MSLD_rho,
            mag.roughness_sigma) == (49.6781, 2.20792e-6, 2.98725e-7,
                                     5.25621)
    # the angles are not read
    assert (mag.MSLD_theta, mag.MSLD_phi) == (0.0, 0.0)
    assert layers[-1].thickness == 0.0
    assert layers[-1].roughness_sigma == 21.2628
    assert all(l.roughness_model == 'tanh' and
               l.roughness_sublayer == PROFILE_SUBLAYERS == 16
               for l in layers[1:])
    g = tmp_path / 'short.txt'
    g.write_text('#a b\n1 2\n')
    with pytest.raises(ValueError, match='8'):
        load_profile(str(g))


def test_import_profile_into_the_tab(tmp_path, monkeypatch):
    """The layers replaced; the roughness scheme, background and
    polarisation kept."""
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PyQt6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])  # noqa: F841
    import main
    f = tmp_path / 'profile.dat'
    f.write_text(PROFILE)
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileName',
                        lambda *a, **k: (str(f), ''))
    win = main.MainWindow()
    sim = win.simulation
    sim.background.setValue(1e-6)
    before = sim.session_state()
    win.import_profile()
    assert [l.name for l in sim.stack.layers][-2:] == ['Layer 3', 'substrate']
    after = sim.session_state()
    assert after['background'] == 1e-6
    assert after['roughness_scheme'] == before['roughness_scheme']
    assert after['resolution'] == before['resolution']
    assert after['simulation']['polarisation'] == \
        before['simulation']['polarisation']
    assert not sim.stack.is_licorne_exact()
    win.close()
