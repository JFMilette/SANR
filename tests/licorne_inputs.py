"""Inputs of the Licorne oracle (tests/licorne_reference.py) from a Licorne
export, read independently of model.licorne_io: (layers, sub) as
licorne_reference.expandrough takes them, values from profile.dat (6
figures) and roughness_fun / roughness_nbound from parameters.m."""
import re

import numpy as np

_LINE = re.compile(r'^\s*(Substrate|Layers\((\d+)\))\.(\w+)\s*=\s*(.*?);\s*$')


def _fields(path):
    sub, layers = {}, {}
    for line in open(path):
        m = _LINE.match(line)
        if m:
            who, k, field, val = m.groups()
            d = sub if who == 'Substrate' else layers.setdefault(int(k), {})
            d[field] = val.strip("'")
    return sub, [layers[k] for k in sorted(layers)]


def oracle_inputs(folder):
    """(layers, sub) of the export in `folder` (a pathlib.Path)."""
    sub_p, lay_p = _fields(folder / 'parameters.m')
    tab = np.atleast_2d(np.loadtxt(folder / 'profile.dat', comments='#'))
    layers = []
    # angles from parameters.m: exports before 1.2.7 write 0 or stale
    # angles to profile.dat
    for row, p in zip(tab[:-1], lay_p):
        ang = [float(a) for a in p['msld'].strip('[]').split(',')[1:]]
        layers.append({'thickness': row[1], 'nsld': complex(row[2], row[3]),
                       'msld': [row[4]] + ang,
                       'roughness': float(p['roughness']),
                       'fun': p['roughness_fun'],
                       'nbound': int(p['roughness_nbound'])})
    last = tab[-1]
    sub = {'nsld': complex(last[2], last[3]),
           'roughness': float(sub_p['roughness']),
           'fun': sub_p['roughness_fun'],
           'nbound': int(sub_p['roughness_nbound'])}
    return layers, sub


def stack_inputs(st):
    """(layers, sub) of a SANR stack with a vacuum fronting, with Licorne's
    axes = the sample frame (M_L = (sin T cos P, sin T sin P, cos T))."""
    def msld(l):
        u = np.array([np.cos(2*np.pi*l.MSLD_phi) * np.cos(2*np.pi*l.MSLD_theta),
                      np.cos(2*np.pi*l.MSLD_phi) * np.sin(2*np.pi*l.MSLD_theta),
                      np.sin(2*np.pi*l.MSLD_phi)])
        return [l.MSLD_rho, np.degrees(np.arctan2(u[1], u[0])),
                np.degrees(np.arccos(np.clip(u[2], -1, 1)))]

    def rough(l):
        sharp = l.roughness_model == 'none'
        return {'roughness': 0.0 if sharp else l.roughness_sigma,
                'fun': 'tanh' if sharp else l.roughness_model,
                'nbound': int(l.roughness_sublayer)}

    layers = [dict(thickness=l.thickness, nsld=complex(l.NSLD_real, l.NSLD_img),
                   msld=msld(l), **rough(l)) for l in st.layers[1:-1]]
    b = st.backing
    sub = dict(nsld=complex(b.NSLD_real, b.NSLD_img), **rough(b))
    return layers, sub
