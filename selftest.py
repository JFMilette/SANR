"""
SANR --self-test <report file>: checks a built SANR (e.g. the Windows
SANR.exe of make_windows_installer.py) without its window: the compiled
transfer-matrix kernel is there, a reflectivity of the default sample is
finite, and a fit cost evaluated in a 2-process 'spawn' worker pool (the
pool DE and DREAM use, fit.workers) equals the same cost in this process.
A frozen program reaches its worker processes only through
multiprocessing.freeze_support() in main.py: without it each worker would
start another SANR instead.

The report goes to a file, as a windowed program has no console: 'OK' on
its last line, or the error.  The exit code is 0 or 1.
"""

import sys
import traceback


def checks():
    """[str] of what was checked; raises on the first failure."""
    import numpy as np

    import model.stack as S
    from fit import workers
    from fit.de.de_fit import _Cost
    from fit.problem import FitProblem
    from model.polarisation import default_vectors
    from tabs.simulation import make_stack

    out = ['python %s, frozen %s' % (sys.version.split()[0],
                                     bool(getattr(sys, 'frozen', False)))]
    if S._transfer is None:
        raise RuntimeError('the compiled kernel model._transfer is missing')
    out.append('compiled kernel: %s' % S._transfer.__file__)

    st = make_stack()
    st.build_sublayers()
    P0, P = (np.asarray(v, dtype=float) for v in default_vectors()['++'])
    Q = np.linspace(0.01, 0.2, 60)
    R = st.reflectivities(Q, [(P0, P)])[:, 0]
    if not np.all(np.isfinite(R)) or not 0 < R.max() <= 1:
        raise RuntimeError('reflectivity out of range: %r' % (R,))
    out.append('reflectivity: %d points, R(Q=0.01) = %.4g' % (len(R), R[0]))

    st.layers[1].fit_entry('thickness')['vary'] = True
    p = FitProblem(st, [dict(Q=Q, R=R * 1.05, dR=0.05 * R, P0=P0, P=P)])
    u = p.x0()
    here = p(u)
    cost = _Cost(p, None)
    pool, token = workers.start(2, p)
    try:
        cost.token = token
        there = pool.map(cost, [u] * 4)
    finally:
        workers.stop(pool, token)
    if not np.allclose(there, here, rtol=1e-12, atol=0):
        raise RuntimeError('worker cost %r != %r' % (there, here))
    out.append('worker pool: 2 processes, cost %.6g' % here)
    return out


def run(path):
    """Write the report to `path`; the exit code."""
    try:
        lines, code = checks() + ['OK'], 0
    except BaseException:
        lines, code = [traceback.format_exc(), 'FAILED'], 1
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return code
