"""
Named channels as (P0, P) polarisation pairs (see model.stack POLARISATION).

A channel is a pair of an incident polarisation P0 and an analyser P,
3-vectors in the sample frame whose length is the efficiency.  The GUI shows
them as two vectors Pi (incident) and Pa (analysed), with Pa = (0, 0, 0)
meaning no analyser (P = None): Pi = (1, 0, 0), Pa = 0 is R+.
"""

import numpy as np

# the named channels, in plot order
CHANNEL_NAMES = ['++', '+-', '-+', '--', '+', '-']
# sign of (P0, P) along the axis for each channel; None = no analyser
_SIGNS = {'++': (1, 1), '+-': (1, -1), '-+': (-1, 1), '--': (-1, -1),
          '+': (1, None), '-': (-1, None)}


def channel_pair(name, n):
    """(P0, P) of channel `name` (one of CHANNEL_NAMES) along the unit
    3-vector n."""
    if name not in _SIGNS:
        raise ValueError('unknown channel %r' % (name,))
    n = np.asarray(n, dtype=float)
    s0, s1 = _SIGNS[name]
    return s0 * n, None if s1 is None else s1 * n


def in_plane(angle):
    """Unit vector in the film plane at `angle` (rad) from sample x; the
    round-off of cos(pi/2) & co. is set to exactly 0."""
    v = np.array([np.cos(angle), np.sin(angle), 0.0])
    return np.where(np.abs(v) < 1e-12, 0.0, v)


def vectors_pair(Pi, Pa):
    """(P0, P) from the GUI's Pi / Pa vectors; Pa = 0 means no analyser."""
    Pa = np.asarray(Pa, dtype=float)
    return np.asarray(Pi, dtype=float), None if not np.any(Pa) else Pa


def pair_vectors(P0, P):
    """Inverse of vectors_pair, as plain lists [Pi, Pa]."""
    return [[float(c) for c in P0],
            [0.0, 0.0, 0.0] if P is None else [float(c) for c in P]]


def default_vectors(n=(1.0, 0.0, 0.0)):
    """{channel: [Pi, Pa]} of every CHANNEL_NAMES along the unit vector n."""
    return {c: pair_vectors(*channel_pair(c, n)) for c in CHANNEL_NAMES}
