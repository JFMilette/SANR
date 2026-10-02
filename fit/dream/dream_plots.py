"""
Matplotlib figures of a DREAM result (fit.dream.dream_fit) for scripts and reports:
traces, corner plot, posterior predictive band of R (and the spin
asymmetry), SLD-profile band.  The GUI draws the same with pyqtgraph
(fit.dream.dream_window); both take the bands from fit.dream.dream_fit.

Every function draws into a matplotlib Figure it is given (no pyplot):
fig = Figure(); trace(fig, res); fig.savefig('trace.png').

The profile band uses the slabs that enter the transfer matrix
(Stack.build_sublayers), not the continuous profile.
"""

import numpy as np

from fit.dream.dream_fit import predictive_bands, profile_bands

SLD_SCALE = 1e-6
BAND = '#4c8dff'
CHANNEL_COLOURS = ['#4c8dff', '#f87171', '#1b9e77', '#e6ab02', '#b392f0',
                   '#d95f02']


def trace(fig, result):
    """log p and every parameter against generation, one line per chain;
    the adaptation phase (never sampled) shaded."""
    names = result.names or ['x%d' % j for j in range(result.chains.shape[2])]
    axes = fig.subplots(len(names) + 1, 1, sharex=True)
    g = result.generations
    series = [('log p', result.logp)] + [
        (n, result.chains[:, :, j]) for j, n in enumerate(names)]
    for ax, (label, y) in zip(axes, series):
        ax.axvspan(0, result.adapt_until, color='0.5', alpha=0.25, lw=0)
        ax.plot(g, y, lw=0.6)
        ax.set_ylabel(label, rotation=0, ha='right', va='center', fontsize=8)
        ax.tick_params(labelsize=7)
    lp = result.logp[np.isfinite(result.logp)]
    if lp.size:                         # the first generations are far off
        lo = np.percentile(lp[lp.size // 10:], 1)
        axes[0].set_ylim(lo - 0.1 * (lp.max() - lo + 1), lp.max() + 1)
    axes[-1].set_xlabel('generation')
    fig.subplots_adjust(left=0.2, right=0.98, top=0.98, bottom=0.06,
                        hspace=0.15)


def corner(fig, result, discard=0.5, bins=30):
    """1-D histograms on the diagonal, 2-D ones below it."""
    s = result.samples(discard)
    d = s.shape[1]
    names = result.names or ['x%d' % j for j in range(d)]
    axes = np.atleast_2d(fig.subplots(d, d, squeeze=False))
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue
            if i == j:
                ax.hist(s[:, j], bins=bins, color=BAND, histtype='stepfilled',
                        alpha=0.7)
                ax.set_yticks([])
                for q in np.percentile(s[:, j], [16, 50, 84]):
                    ax.axvline(q, color='0.6', lw=0.8, ls='--')
            else:
                ax.hist2d(s[:, j], s[:, i], bins=bins, cmap='Blues')
            if i < d - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(names[j], fontsize=7)
            if j > 0 or i == 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(names[i], fontsize=7)
            ax.tick_params(labelsize=6)
            for t in ax.get_xticklabels():
                t.set_rotation(45)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.98, bottom=0.1,
                        wspace=0.05, hspace=0.05)


def predictive(fig, result, problem, n=200, discard=0.5, seed=0):
    """Data with the median and the 68 / 95 % bands of the model R of n
    posterior draws, and (lighter, dashed edges) the 95 % band of
    replicated data (model + noise dR); below, the spin asymmetry if R+ /
    R- (or R++ / R--) were fitted on the same Q points."""
    b = predictive_bands(result, problem, n, discard, seed)
    asym = b['asymmetry']
    axes = fig.subplots(2 if asym else 1, 1, sharex=True,
                        squeeze=False)[:, 0]
    ax = axes[0]
    for k, c in enumerate(b['channels']):
        col = CHANNEL_COLOURS[k % len(CHANNEL_COLOURS)]
        lo95, lo68, med, hi68, hi95 = c['bands']
        plo, phi = np.maximum(c['predictive'][0], 1e-3 * med), \
            c['predictive'][4]
        ax.fill_between(c['Q'], plo, phi, color=col, alpha=0.12, lw=0.8,
                        ls='--', edgecolor=col)
        ax.errorbar(c['Q'], c['R'], c['dR'], fmt='o', ms=2.5, color=col,
                    alpha=0.6, elinewidth=0.6, label=c['name'])
        ax.fill_between(c['Q'], lo95, hi95, color=col, alpha=0.2, lw=0)
        ax.fill_between(c['Q'], lo68, hi68, color=col, alpha=0.35, lw=0)
        ax.plot(c['Q'], med, color=col, lw=1)
    ax.set_yscale('log')
    ax.set_ylabel('R')
    ax.legend(fontsize=8)
    if asym:
        lo95, _, med, _, hi95 = asym['bands']
        axes[1].fill_between(asym['Q'], asym['predictive'][0],
                             asym['predictive'][4], color=BAND, alpha=0.12,
                             lw=0.8, ls='--', edgecolor=BAND)
        axes[1].errorbar(asym['Q'], asym['A'], asym['dA'], fmt='o', ms=2.5,
                         color='0.6', elinewidth=0.6)
        axes[1].fill_between(asym['Q'], lo95, hi95, color=BAND, alpha=0.3,
                             lw=0)
        axes[1].plot(asym['Q'], med, color=BAND, lw=1)
        axes[1].set_ylabel('spin asymmetry')
    axes[-1].set_xlabel('Q (Å⁻¹)')
    fig.subplots_adjust(left=0.12, right=0.98, top=0.98, bottom=0.08,
                        hspace=0.08)


def profile(fig, result, problem, n=200, discard=0.5, seed=0, npts=1500):
    """Median and 68 / 95 % bands of the nuclear and in-plane magnetic SLD
    of n posterior draws, from their slabs."""
    z, nuc, mag = profile_bands(result, problem, n, discard, seed, npts)
    ax = fig.subplots()
    for bands, col, label in ((nuc, BAND, 'nuclear SLD'),
                              (mag, '#f87171', 'magnetic SLD (in plane)')):
        lo95, lo68, med, hi68, hi95 = bands / SLD_SCALE
        ax.fill_between(z, lo95, hi95, color=col, alpha=0.2, lw=0)
        ax.fill_between(z, lo68, hi68, color=col, alpha=0.35, lw=0)
        ax.plot(z, med, color=col, lw=1, label=label)
    ax.set_xlabel('depth z (Å)')
    ax.set_ylabel('SLD (10⁻⁶ Å⁻²)')
    ax.legend(fontsize=8)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.98, bottom=0.1)
