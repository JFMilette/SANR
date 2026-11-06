"""
Licorne's resolution convolution (reflection_m.m `resolut`, reflection.cpp
for mode 1) as a weight matrix: R_res = W @ R on the Q points themselves.

Licorne smears R only at the points it is computed on (the data Q): each
point sums its neighbours within +-3 sigma with Gaussian weights times a
spacing, so the result depends on the grid.  `resolut` is linear in R and
depends only on (q, sigma, mode), so it is written here once as a sparse
matrix W (kernel) that every trial multiplies, rather than looped over per
trial.  Every weight below accumulates exactly what `resolut` adds to RRr,
read line by line against tests/licorne_reference.py (1-based there, as in
the MATLAB source; 0-based here):

  mode 1  (reflection.cpp; the MATLAB mode-1 loop indexes with the
          imaginary unit and crashes): the plain mean of the points within
          +-sigma/2, the point itself included.
  mode 2  ('dQ centred'): R_i dqc_i + sum_j R_j exp(-dq^2 / 2 sigma^2) dl_j,
          over |q_j - q_i| <= 3 sigma, divided by sigma sqrt(2 pi); dl_j is
          the spacing to the neighbour on the side of q_i (rectangle rule),
          dqc_i the half-distance between the two neighbours of q_i.
  mode 3  ('dQ centred to own centres'): the same with dl_j the
          half-distance between the two neighbours of q_j (midpoint rule).
          The first two and last two points have their own branches.

Common to modes 2 and 3: past the first or last point the sum goes on with
the end point's value at the end spacing (constant tails); a point whose
sigma is under half its spacing (dq < dqc/2) is left as is (W_ii = 1); the
sum is NOT normalised, so the rows of W sum to about 1, not exactly.
"""

import numpy as np
from scipy import sparse

MODES = (1, 2, 3)
_SQRT2PI = np.sqrt(2 * np.pi)


def kernel(q, sigma, mode=3):
    """W (scipy.sparse.csr_matrix, N x N) with W @ R = Licorne's resolut(R,
    q, sigma, mode).  q strictly increasing, sigma >= 0 and finite."""
    q = np.asarray(q, dtype=float)
    dq = np.asarray(sigma, dtype=float)
    N = len(q)
    if mode not in MODES:
        raise ValueError('Licorne resolution mode must be 1, 2 or 3')
    if q.ndim != 1 or dq.shape != q.shape:
        raise ValueError('q and sigma must be 1-D arrays of the same length')
    if N > 1 and not np.all(np.diff(q) >= 0):
        raise ValueError('Licorne resolution needs non-decreasing Q')
    # repeated Q is valid inside the grid (stitched TOF angles), but not at
    # either end: the constant tails past an end step by the end spacing,
    # so Licorne's loop would never end (it hangs there too)
    if N > 1 and (q[1] == q[0] or q[-1] == q[-2]):
        raise ValueError('Licorne resolution: the first or last Q is '
                         'repeated (zero end spacing)')
    if not np.all(np.isfinite(dq) & (dq >= 0)):
        raise ValueError('resolution sigma must be finite and >= 0')
    if N < (4 if mode == 3 else 2):
        return sparse.identity(N, format='csr')
    key = (q.tobytes(), dq.tobytes(), mode)
    W = _CACHE.pop(key, None)
    if W is None:
        W = _build(q, dq, mode)
    _CACHE[key] = W                          # the most recent last
    total = sum(w.nnz for w in _CACHE.values())
    while total > CACHE_NNZ and len(_CACHE) > 1:
        total -= _CACHE.pop(next(iter(_CACHE))).nnz
    return W


# kernels already built, shared by every stack (a model switch or a live
# curve builds new Stack objects, whose own caches start empty): (q, sigma,
# mode) -> W, least recently used first, at most CACHE_NNZ stored weights
# in all (about 12 bytes each) beside the newest
_CACHE = {}
CACHE_NNZ = 12_000_000


def _build(q, dq, mode):
    N = len(q)
    rows, cols, vals = [], [], []        # scalars
    blocks = []                          # (row, cols array, weights array)

    def add(i, j, w):
        rows.append(i)
        cols.append(j)
        vals.append(w)

    def add_many(i, js, ws):
        if len(js):
            blocks.append((i, js, ws))

    if mode == 1:
        _mode1(q, dq, add)
    elif mode == 2:
        _mode2(q, dq, add, add_many)
    else:
        _mode3(q, dq, add, add_many)
    r = [np.asarray(rows, dtype=np.int64)]
    c = [np.asarray(cols, dtype=np.int64)]
    v = [np.asarray(vals, dtype=float)]
    for i, js, ws in blocks:
        r.append(np.full(len(js), i, dtype=np.int64))
        c.append(np.asarray(js, dtype=np.int64))
        v.append(ws)
    return sparse.csr_matrix(
        (np.concatenate(v), (np.concatenate(r), np.concatenate(c))),
        shape=(N, N))


def _side(i, q, js, dls, ts, ss, sp, to_end, end, d_end, add_many):
    """One side of row i, as the loops of _mode2 / _mode3 walk it: the
    neighbours js (walked outward from i, at least one) while
    |q_j - q_i| <= ts, each with its spacing dls; if js run `to_end` (the
    last neighbour the loop takes before its tail) and every one is
    within, the constant tail past the end point `end` (spacing d_end)
    goes on from the last of them.  The same operations as the loops, only
    on arrays: qq is |q_i - q_j|, the tail's qq a running sum
    (np.add.accumulate adds in order, as qq += dl does), the weights
    exp(-qq^2 / ss) * dl / sp."""
    if not len(js):
        return
    qq = np.abs(q[i] - q[js])
    keep = np.logical_and.accumulate(qq <= ts)
    k = int(keep.sum())
    if k:
        add_many(i, js[:k], np.exp(-qq[:k] * qq[:k] / ss) * dls[:k] / sp)
    if k < len(js) or not to_end:
        return
    last = qq[-1]
    m = int((ts - last) / d_end) + 2
    tail = np.add.accumulate(np.concatenate(([last], np.full(m, d_end))))[1:]
    tail = tail[np.logical_and.accumulate(tail <= ts)]
    if len(tail):
        add_many(i, np.full(len(tail), end),
                 np.exp(-tail * tail / ss) * d_end / sp)


def _window(q, i, ts):
    """Indices lo, hi bracketing every j with |q_j - q_i| <= ts (with room
    to spare: _side decides each one exactly)."""
    return (int(np.searchsorted(q, q[i] - 2 * ts - 1e-300, 'left')),
            int(np.searchsorted(q, q[i] + 2 * ts + 1e-300, 'right')))


def _mode1(q, dq, add):
    """reflection.cpp mode 1: lower neighbours of every point but the first,
    then the point itself and its upper neighbours (the last point has
    none), all divided by their count."""
    N = len(q)
    lower = [[] for _ in range(N)]
    for i in range(1, N):
        k = 1
        while q[i] - q[i - k] <= dq[i] / 2:
            lower[i].append(i - k)
            k += 1
            if i - k < 0:
                break
    for i in range(N):
        terms = lower[i] + [i]
        if i < N - 1:
            k = 1
            while q[i + k] - q[i] <= dq[i] / 2:
                terms.append(i + k)
                k += 1
                if i + k > N - 1:
                    break
        for j in terms:
            add(i, j, 1.0 / len(terms))


def _skip(i, dqc, dq, add):
    """sigma under half the spacing: the point is left as is."""
    if dq[i] < dqc / 2:
        add(i, i, 1.0)
        return True
    return False


def _tail(i, j, qq, dl, ts, ss, sp, add):
    """The constant tail past an end point: R_j at steps dl from qq on."""
    while qq <= ts:
        add(i, j, np.exp(-qq * qq / ss) * dl / sp)
        qq += dl


def _mode2(q, dq, add, add_many):
    N = len(q)
    # the spacing to the neighbour on the side of q_i: below i, j's gap
    # up (q_j+1 - q_j); above it, j's gap down (q_j - q_j-1)
    gap = np.abs(np.diff(q))
    up = np.append(gap, 0.0)
    down = np.insert(gap, 0, 0.0)
    d0, dn = abs(q[1] - q[0]), abs(q[N - 1] - q[N - 2])
    for i in range(1, N - 1):
        dqc = abs(q[i + 1] - q[i - 1]) / 2
        if _skip(i, dqc, dq, add):
            continue
        sp, ss, ts = dq[i] * _SQRT2PI, 2 * dq[i]**2, 3 * dq[i]
        add(i, i, dqc / sp)
        lo, hi = _window(q, i, ts)
        js = np.arange(i - 1, lo - 1, -1)                # i-1 down to lo
        _side(i, q, js, up[js], ts, ss, sp, lo == 0, 0, d0, add_many)
        js = np.arange(i + 1, min(hi, N))                # i+1 up
        _side(i, q, js, down[js], ts, ss, sp, hi >= N, N - 1, dn, add_many)
    # first point
    dqc = abs(q[1] - q[0])
    if not _skip(0, dqc, dq, add):
        sp, ss, ts = dq[0] * _SQRT2PI, 2 * dq[0]**2, 3 * dq[0]
        add(0, 0, dqc / sp)
        k, qq = 1, abs(q[1] - q[0])
        while qq <= ts:
            add(0, k, np.exp(-qq * qq / ss) * abs(q[k] - q[k - 1]) / sp)
            k += 1
            if k > N - 1:
                break
            qq = abs(q[k] - q[0])
        dl = abs(q[1] - q[0])
        _tail(0, 0, dl, dl, ts, ss, sp, add)
    # last point
    n = N - 1
    dqc = abs(q[n] - q[n - 1])
    if not _skip(n, dqc, dq, add):
        sp, ss, ts = dq[n] * _SQRT2PI, 2 * dq[n]**2, 3 * dq[n]
        add(n, n, dqc / sp)
        k, qq = 1, abs(q[n] - q[n - 1])
        while qq <= ts:
            add(n, n - k,
                np.exp(-qq * qq / ss) * abs(q[n - k + 1] - q[n - k]) / sp)
            k += 1
            if n - k < 0:
                break
            qq = q[n] - q[n - k]
        dl = abs(q[n] - q[n - 1])
        _tail(n, n, dl, dl, ts, ss, sp, add)


def _mode3(q, dq, add, add_many):
    N = len(q)
    n = N - 1
    # each point's own centre spacing, half the distance between its two
    # neighbours (1 .. n-1; the ends unused)
    mid = np.zeros(N)
    mid[1:-1] = np.abs(q[2:] - q[:-2]) / 2
    d0, dn = abs(q[1] - q[0]), abs(q[n] - q[n - 1])
    for i in range(2, N - 2):
        dqc = abs(q[i + 1] - q[i - 1]) / 2
        if _skip(i, dqc, dq, add):
            continue
        sp, ss, ts = dq[i] * _SQRT2PI, 2 * dq[i]**2, 3 * dq[i]
        add(i, i, dqc / sp)
        lo, hi = _window(q, i, ts)
        lo = max(lo, 1)
        js = np.arange(i - 1, lo - 1, -1)                # i-1 down to 1
        _side(i, q, js, mid[js], ts, ss, sp, lo == 1, 0, d0, add_many)
        js = np.arange(i + 1, min(hi, n))                # i+1 up to n-1
        _side(i, q, js, mid[js], ts, ss, sp, hi >= n, n, dn, add_many)
    # first point
    dqc = abs(q[1] - q[0])
    if not _skip(0, dqc, dq, add):
        sp, ss, ts = dq[0] * _SQRT2PI, 2 * dq[0]**2, 3 * dq[0]
        add(0, 0, dqc / sp)
        k, qq = 1, abs(q[1] - q[0])
        while qq <= ts:
            add(0, k, np.exp(-qq * qq / ss)
                * (abs(q[k + 1] - q[k - 1]) / 2) / sp)
            k += 1
            if k > n - 1:
                break
            qq = abs(q[k] - q[0])
        dl = abs(q[1] - q[0])
        _tail(0, 0, dl, dl, ts, ss, sp, add)
    # second point
    dqc = abs(q[2] - q[0]) / 2
    if not _skip(1, dqc, dq, add):
        sp, ss, ts = dq[1] * _SQRT2PI, 2 * dq[1]**2, 3 * dq[1]
        add(1, 1, dqc / sp)
        k, qq = 2, abs(q[2] - q[1])
        while qq <= ts:
            add(1, k, np.exp(-qq * qq / ss)
                * (abs(q[k + 1] - q[k - 1]) / 2) / sp)
            k += 1
            if k > n - 1:
                break
            qq = abs(q[k] - q[1])
        _tail(1, 0, abs(q[1] - q[0]), abs(q[2] - q[0]) / 2, ts, ss, sp, add)
    # point before last
    m = n - 1
    dqc = abs(q[n] - q[m - 1]) / 2
    if not _skip(m, dqc, dq, add):
        sp, ss, ts = dq[m] * _SQRT2PI, 2 * dq[m]**2, 3 * dq[m]
        add(m, m, dqc / sp)
        k, qq = 2, abs(q[m] - q[m - 1])
        while qq <= ts:
            add(m, n - k, np.exp(-qq * qq / ss)
                * (abs(q[n - k + 1] - q[n - k - 1]) / 2) / sp)
            k += 1
            if n - k < 1:
                break
            qq = abs(q[m] - q[n - k])
        _tail(m, n, abs(q[n] - q[m]), abs(q[n] - q[m - 1]) / 2, ts, ss, sp,
              add)
    # last point
    dqc = abs(q[n] - q[m])
    if not _skip(n, dqc, dq, add):
        sp, ss, ts = dq[n] * _SQRT2PI, 2 * dq[n]**2, 3 * dq[n]
        add(n, n, dqc / sp)
        k, qq = 1, abs(q[n] - q[m])
        while qq <= ts:
            add(n, n - k, np.exp(-qq * qq / ss)
                * (abs(q[n - k + 1] - q[n - k - 1]) / 2) / sp)
            k += 1
            if n - k < 1:
                break
            qq = abs(q[n] - q[n - k])
        dl = abs(q[n] - q[m])
        _tail(n, n, dl, dl, ts, ss, sp, add)
