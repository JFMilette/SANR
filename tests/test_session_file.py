"""Sessions as a .zip with a copy of their data files (session_file).

Run from the repo root: python -m pytest tests
"""
import copy
import os
import pathlib
import shutil
import sys
import zipfile

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import session_file                                             # noqa: E402
from tabs.experimental import file_ref                          # noqa: E402


def test_round_trip_names_and_copies(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    a.mkdir()
    b.mkdir()
    (a / 'r.dat').write_bytes(b'1 2 3\n')
    (b / 'R.dat').write_bytes(b'4 5 6\n')       # same name, other folder
    (a / 'q.dat').write_bytes(b'0.01\n')
    d = {'sets': [{'channels': {
        '+': {'file': file_ref(a / 'r.dat', tmp_path),
              'q_file': file_ref(a / 'q.dat', tmp_path)},
        '-': {'file': file_ref(b / 'R.dat', tmp_path),
              'q_file': file_ref(a / 'q.dat', tmp_path)},
        'x': {'values': {'Q': [1.0]}, 'path': 'theory'}}}]}
    out = tmp_path / 's.zip'
    session_file.write_session(str(out), d)
    with zipfile.ZipFile(out) as z:
        assert sorted(z.namelist()) == ['data/R_2.dat', 'data/q.dat',
                                        'data/r.dat', 'session.json']
    shutil.rmtree(a)
    shutil.rmtree(b)
    assert session_file.is_zip_session(str(out))
    e, folder = session_file.read_session(str(out))
    ch = e['sets'][0]['channels']
    assert open(ch['+']['file']['path'], 'rb').read() == b'1 2 3\n'
    assert open(ch['-']['file']['path'], 'rb').read() == b'4 5 6\n'
    assert ch['+']['q_file']['path'] == ch['-']['q_file']['path']
    assert os.path.dirname(ch['+']['file']['path']) == folder
    assert ch['x'] == {'values': {'Q': [1.0]}, 'path': 'theory'}
    assert len(session_file.only_in_session(e)) == 4


def test_missing_file_leaves_old_session(tmp_path):
    out = tmp_path / 's.zip'
    out.write_bytes(b'old')
    d = {'f': file_ref(tmp_path / 'gone.dat', tmp_path)}
    with pytest.raises(OSError):
        session_file.write_session(str(out), d)
    assert out.read_bytes() == b'old'
    assert not (tmp_path / 's.zip.part').exists()


@pytest.mark.parametrize('name', ['data/../evil.dat', 'data/sub/x.dat',
                                  '../x.dat', 'data/'])
def test_bad_names_refused(tmp_path, name):
    out = tmp_path / 's.zip'
    with zipfile.ZipFile(out, 'w') as z:
        z.writestr('session.json', '{"f": {"path": "p", "relative": "r", '
                   '"embedded": "%s"}}' % name)
        z.writestr(name if not name.endswith('/') else 'data/x', b'')
    with pytest.raises(ValueError):
        session_file.read_session(str(out))


def test_main_window_round_trip(tmp_path, monkeypatch):
    """A Licorne import saved as .zip opens with its folder deleted."""
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PyQt6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    import main
    from model.licorne_io import load_licorne_session
    from test_licorne_session import edited_session
    d = edited_session(tmp_path, Q_mult='1.01', Rexp_mult='[2,1,1,1,1,1]')
    win = main.MainWindow()
    lic = load_licorne_session(d)
    assert win._restore(win._licorne_state(lic, str(d)), str(d)) == []
    before = {ch: c['R'].copy()
              for ch, c in win.experimental.sets[0]['channels'].items()}
    out = tmp_path / 'saved' / 'session'
    out.parent.mkdir()
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName',
                        staticmethod(lambda *a, **k: (str(out), main.SAVE_FILTER
                                                      .split(';;')[0])))
    win.save_session()
    assert (out.parent / 'session.zip').is_file()
    win.close()
    shutil.rmtree(d)

    win = main.MainWindow()
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileName',
                        staticmethod(lambda *a, **k: (
                            str(out.parent / 'session.zip'), '')))
    warned = []
    monkeypatch.setattr(QtWidgets.QMessageBox, 'warning',
                        staticmethod(lambda *a, **k: warned.append(a)))
    monkeypatch.setattr(QtWidgets.QMessageBox, 'critical',
                        staticmethod(lambda *a, **k: warned.append(a)))
    win.open_session()
    assert warned == []
    after = win.experimental.sets[0]['channels']
    assert set(after) == set(before)
    for ch in before:
        assert np.array_equal(after[ch]['R'], before[ch])
    win.close()


def test_dream_npe_options_round_trip():
    """The multistart, DREAM and NPE pages' settings are kept by a
    session; an older session without them leaves the boxes as they
    are."""
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PyQt6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    import main
    win = main.MainWindow()
    exp = win.experimental
    exp.dream_archive.setChecked(False)
    exp.dream_gen.setValue(12345)
    exp.dream_chains.setValue(16)
    exp.dream_seed.setValue(7)
    exp.dream_rhat.setValue(1.05)
    exp.dream_snooker.setValue(0.2)
    exp.dream_beta.setValue(1e-2)
    exp.dream_start.setCurrentIndex(exp.dream_start.findData('current'))
    for b in (exp.dream_sigma, exp.dream_floor, exp.dream_stop,
              exp.dream_temper):
        b.setChecked(not b.isChecked())
    exp.npe_rounds.setValue(3)
    exp.npe_sims.setValue(5000)
    exp.npe_samples.setValue(2000)
    exp.npe_chains.setValue(12)
    exp.npe_seed.setValue(11)
    exp.npe_coarse.setChecked(False)
    exp.npe_threshold.setValue(0.5)
    exp.npe_calibrate.setChecked(False)
    exp.ms_starts.setValue(40)
    exp.ms_method.setCurrentIndex(exp.ms_method.findData('lsq'))
    pages = ('multistart', 'dream', 'npe')
    want = {p: exp.page_options(p) for p in pages}
    assert want['dream']['chains'] == 16 and want['npe']['seeds'] == 12
    state = exp.session_state(os.getcwd())
    win.close()

    win = main.MainWindow()
    exp = win.experimental
    default = {p: exp.page_options(p) for p in pages}
    old = copy.deepcopy(state)
    for k in pages:
        del old['fit'][k]
    exp.restore_state(old, os.getcwd())
    assert {p: exp.page_options(p) for p in pages} == default
    exp.restore_state(state, os.getcwd())
    got = {p: exp.page_options(p) for p in pages}
    assert got['npe'] == want['npe']
    assert got['multistart'] == {'starts': 40, 'method': 'lsq'}
    for k, v in want['dream'].items():
        assert got['dream'][k] == pytest.approx(v), k
    assert exp.dream_chains.minimum() == 7          # archive unticked
    win.close()


def test_fit_tabs_get_a_file_each(tmp_path):
    """Each fit tab is a tabs/<n>_<algorithm>.json of the zip, each DREAM
    run a tabs/<n>_DREAM.npz (its raw bytes); read back as a .json session has
    them."""
    import base64
    import json
    npz = bytes(range(256)) * 50
    tabs = [dict(kind='fit', tab_name='L-BFGS-B 1 · d', problem={'x': [1.0] * 500},
                 entries=[['a', 1.0, [2.0]]]),
            dict(kind='dream', tab_name='DREAM 2 · d', header='h',
                 status='s', title='t',
                 npz=base64.b64encode(npz).decode('ascii'))]
    d = {'layers': [], 'fit_tabs': json.loads(json.dumps(tabs))}
    out = tmp_path / 's.zip'
    session_file.write_session(str(out), d)
    with zipfile.ZipFile(out) as z:
        assert sorted(z.namelist()) == ['session.json', 'tabs/1_L-BFGS-B.json',
                                        'tabs/2_DREAM.npz']
        assert z.read('tabs/2_DREAM.npz') == npz
        assert json.loads(z.read('tabs/1_L-BFGS-B.json')) == tabs[0]
        light = json.loads(z.read('session.json'))['fit_tabs']
    assert light[0] == {'kind': 'fit', 'file': 'tabs/1_L-BFGS-B.json',
                        'tab_name': 'L-BFGS-B 1 · d'}
    assert 'npz' not in light[1] and light[1]['header'] == 'h'
    e, _ = session_file.read_session(str(out))
    assert e['fit_tabs'] == tabs


def test_bad_tab_file_refused(tmp_path):
    out = tmp_path / 's.zip'
    with zipfile.ZipFile(out, 'w') as z:
        z.writestr('session.json', '{"fit_tabs": [{"kind": "fit", '
                   '"file": "tabs/../session.json"}]}')
    with pytest.raises(ValueError):
        session_file.read_session(str(out))


def test_tab_label():
    assert session_file.tab_label({'tab_name': 'SHGO 3 · film/5K'}) == 'SHGO'
    assert session_file.tab_label({'tab_name': 'Multistart 1 · a'}) ==         'Multistart'
    assert session_file.tab_label({'method': 'de'}) == 'de'
    assert session_file.tab_label({'kind': 'dream'}) == 'dream'
    assert session_file.tab_label({'tab_name': 'a/b:c 1 · x'}) == 'a_b_c'
