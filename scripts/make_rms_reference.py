"""
Freeze the 'rms' roughness scheme's additive profile as a test reference:
tests/data/rms_additive_reference.npz (tests/test_interface_order.py T1, T10).

Run once, from the repo root, with the code to be frozen:

    python scripts/make_rms_reference.py

For each case it stores, under '<case>/<field>':
    stack         the stack as JSON (Stack.to_dict)
    z             depths of the profile
    nsld, rho, theta, phi    Stack.profile(z)
    slab_t, slab_nsld, slab_rho, slab_theta, slab_phi    build_sublayers()
    Q             momentum transfers
    R_res_on, R_res_off      reflectivities of every CHANNEL_NAMES along x,
                             resolution on and off, shape (nQ, 6)

Cases: the stress stacks (stress_<s1>_<s2>_<model>_<smearing>), the
Cr2Te3 session (sessions/session_Cr2Te3.json) and tests/data/show_mistake.json.
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model.polarisation import CHANNEL_NAMES, channel_pair   # noqa: E402
from model.stack import Layer, Stack                         # noqa: E402
import model.stack                                           # noqa: E402

# bit-for-bit reference: the numpy transfer matrix, as the test uses
model.stack.USE_COMPILED = False

OUT = os.path.join(ROOT, 'tests', 'data', 'rms_additive_reference.npz')
SESSIONS = {'session_Cr2Te3': os.path.join(ROOT, 'sessions',
                                            'session_Cr2Te3.json'),
            'show_mistake': os.path.join(ROOT, 'tests', 'data',
                                         'show_mistake.json')}
# (sigma at the top, sigma at the bottom) of the 20 A ferromagnet
STRESS_SIGMAS = [(2, 8), (2, 15), (8, 2), (5, 5)]
Q = np.linspace(0.003, 0.2, 300)
PAIRS = [channel_pair(c, (1.0, 0.0, 0.0)) for c in CHANNEL_NAMES]


def stress_stack(s_top, s_bot, model, smearing):
    """vacuum / FM 20 / spacer 30 / FM 40 / Si; the 20 A FM has sigma
    s_top above and s_bot below, every other interface 3 A (all 5 A in the
    5 / 5 case, so nothing crosses there)."""
    other = 5.0 if (s_top, s_bot) == (5, 5) else 3.0
    L = [Layer('vacuum'),
         Layer('FM', 20, 4e-6, -1e-8, 2e-6, 0.1, s_top, model, 6),
         Layer('spacer', 30, 2e-6, 0, 0, 0.0, s_bot, model, 6),
         Layer('FM2', 40, 6e-6, 0, 1e-6, 0.35, other, model, 6,
               MSLD_phi=0.05),
         Layer('Si', 0, 2.07e-6, 0, 0, 0, other, model, 6)]
    st = Stack(L, magnetic_smearing=smearing)
    st.interface_order = 'additive'     # the reference (new stacks: 'ordered')
    st.resolution['enabled'] = True
    return st


def cases():
    for s1, s2 in STRESS_SIGMAS:
        for model in ('tanh', 'erf'):
            for smearing in ('vector', 'angle', 'step'):
                yield ('stress_%g_%g_%s_%s' % (s1, s2, model, smearing),
                       stress_stack(s1, s2, model, smearing))
    for name, path in SESSIONS.items():
        with open(path) as f:
            st = Stack.from_dict(json.load(f))
        st.resolution['enabled'] = True
        yield name, st


def freeze(st):
    out = {'stack': np.array(json.dumps(st.to_dict()))}
    Z = st._interfaces()
    z = np.linspace(Z[0] - 60.0, Z[-1] + 60.0, 1500)
    out['z'] = z
    for k, v in zip(('nsld', 'rho', 'theta', 'phi'), st.profile(z)):
        out[k] = v
    slabs = st.build_sublayers()
    out['slab_t'] = np.array([s.thickness for s in slabs])
    out['slab_nsld'] = np.array([complex(s.NSLD_real, s.NSLD_img)
                                 for s in slabs])
    for k in ('rho', 'theta', 'phi'):
        out['slab_' + k] = np.array([getattr(s, 'MSLD_' + k) for s in slabs])
    out['Q'] = Q
    out['R_res_on'] = st.reflectivities(Q, PAIRS, warn=False)
    st.resolution['enabled'] = False
    out['R_res_off'] = st.reflectivities(Q, PAIRS, warn=False)
    return out


def main():
    data = {}
    for name, st in cases():
        for k, v in freeze(st).items():
            data['%s/%s' % (name, k)] = v
    np.savez_compressed(OUT, **data)
    print('wrote %s (%d cases)' % (OUT, len({k.split('/')[0]
                                             for k in data})))


if __name__ == '__main__':
    main()
