"""
Plot the depth profile of a Stack: the exact continuous profile and the
sublayer slabs that the transfer matrix actually sees.

Three panels, sharing a depth axis:
    NSLD real   /   MSLD rho   /   MSLD theta

The solid line is Stack.profile(z), the exact additive-interface profile.
The shaded rectangles are Stack.build_sublayers(), each drawn across its own
thickness at its own value -- literally the slab stack fed to the matrices.
Dashed verticals mark the nominal interface depths.
"""

import numpy as np
import matplotlib.pyplot as plt

from model.stack import Layer, Stack


# ------------------------------------------------------------- the sample --
def make_stack():
    vacuum = Layer('vacuum')

    L1 = Layer('L1', thickness=80.0, NSLD_real=4.0e-6,
               MSLD_rho=1.5e-6, MSLD_theta=0.0,
               roughness_sigma=8.0, roughness_model='tanh', roughness_sublayer=20)

    L2 = Layer('L2', thickness=120.0, NSLD_real=1.5e-6, NSLD_img=-2.0e-8,
               MSLD_rho=0.8e-6, MSLD_theta=0.0,
               roughness_sigma=5.0, roughness_model='tanh', roughness_sublayer=20)

    L3 = Layer('L3', thickness=60.0, NSLD_real=6.0e-6,
               MSLD_rho=0.0, MSLD_theta=0.0,
               roughness_sigma=8.0, roughness_model='tanh', roughness_sublayer=20)

    substrate = Layer('substrate', NSLD_real=2.07e-6,
                      roughness_sigma=4.0, roughness_model='tanh',
                      roughness_sublayer=20)

    return Stack([vacuum, L1, L2, L3, substrate])


# ---------------------------------------------------------------- drawing --
def slab_edges(stack):
    """Depth of every slab boundary; the first slab starts at its own top."""
    lo = stack.windows()[0][0]
    return lo + np.concatenate([[0.0], np.cumsum([s.thickness
                                                  for s in stack.sublayers])])


def draw(stack, scale=1e-6, unit=r'$10^{-6}\,\AA^{-2}$'):
    stack.build_sublayers()
    Z = stack._interfaces()
    e = slab_edges(stack)

    pad = 0.35 * max(Z[-1], 1.0)
    z = np.linspace(min(e[0], Z[0] - pad) - 10, max(e[-1], Z[-1] + pad) + 10, 4000)
    nsld, rho, theta = stack.profile(z)

    fig, ax = plt.subplots(3, 1, figsize=(9.5, 8.2), sharex=True)
    fig.subplots_adjust(hspace=0.12, left=0.11, right=0.97, top=0.94, bottom=0.08)

    panels = [
        (nsld.real / scale, [s.NSLD_real / scale for s in stack.sublayers],
         'NSLD real  (%s)' % unit, '#1f4e79'),
        (rho / scale, [s.MSLD_rho / scale for s in stack.sublayers],
         'MSLD rho  (%s)' % unit, '#2a7f62'),
        (theta * 360, [s.MSLD_theta * 360 for s in stack.sublayers],
         'MSLD theta  (deg)', '#8b5a2b'),
    ]

    for a, (curve, slabvals, label, colour) in zip(ax, panels):
        for Zj in Z:
            a.axvline(Zj, color='0.82', lw=0.8, ls='--', zorder=0)
        a.axhline(0.0, color='0.75', lw=0.7, zorder=0)
        a.bar(e[:-1], slabvals, width=np.diff(e), align='edge', bottom=0.0,
              color=colour, alpha=0.20, edgecolor=colour, linewidth=0.6, zorder=2)
        a.plot(z, curve, lw=2.0, color=colour, zorder=3)
        a.set_ylabel(label)
        a.grid(alpha=0.22)
        a.set_xlim(z[0], z[-1])

    ax[0].set_title('%s   |   %d sublayers' % (
        ' / '.join(l.name for l in stack.layers), len(stack.sublayers)),
        fontsize=10)
    for Zj, l in zip(np.concatenate([[z[0]], Z]), stack.layers):
        ax[0].annotate(l.name, (Zj + 3, 0.96), xycoords=('data', 'axes fraction'),
                       fontsize=8, color='0.45', va='top')
    ax[-1].set_xlabel(r'depth  $z$  ($\AA$)')
    return fig


if __name__ == '__main__':
    st = make_stack()
    st.build_sublayers()

    Q=np.linspace(0.0001, 0.25, 400)

    reflectance=st.calc_reflectance(Q)
    

    print(reflectance.shape)

    # plt.plot(Q, reflectance[:,0], label='R++')
    # plt.plot(Q, reflectance[:,1], label='R+-')
    # plt.plot(Q, reflectance[:,2], label='R-+')
    # plt.plot(Q, reflectance[:,3], label='R--')

    plt.plot(Q, reflectance[:,4], label='R++ lab')
    plt.plot(Q, reflectance[:,5], label='R+- lab')
    plt.plot(Q, reflectance[:,6], label='R-+ lab')
    plt.plot(Q, reflectance[:,7], label='R-- lab')


    plt.yscale('log')
    plt.xlabel(r'Q ($\AA^{-1}$)')
    plt.ylabel('Reflectivity')
    plt.title('Reflectivity of the stack')
    plt.legend()
    plt.show()



        

    draw(st)
    plt.show()