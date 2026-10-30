"""
Sequential neural posterior estimation of a fit problem with sbi
(plans/npe_plan_v1.md T1, V2), written against sbi 0.27.0
(requirements-npe.txt).

PARAMETERS -- the network models theta = logit(u), u in [0, 1]^d the fit's
normalised parameters (fit.npe.simulator), with the prior pushed to theta
(LogitPrior: the UnitPrior times the Jacobian u (1 - u)): its samples map
back inside the bounds whatever the network does, and nothing leaks out of
the prior.  u is kept within LOGIT_EPS of the bounds.

NETWORK -- a neural spline flow ('nsf', or any of sbi's density estimators)
over theta, conditioned on an embedding of the features: an MLP (EMBEDDING:
hidden widths, output size) on the feature vector, which sbi standardises
with the means and spreads of the first round's simulations, kept in the
network.  Training runs on the CPU in float32; the simulator stays float64.

ROUNDS -- round 1 draws from the prior: its network is an amortised estimator
over the whole prior, kept apart (NPEResult.networks[0]).  Each later round
draws from the prior truncated to the current posterior's highest-density
region at x_o, the measured data (TSNPE, Deistler et al. 2022): the region
(Region) holds all but `quantile` of the posterior mass, its threshold on log q
estimated from SUPPORT_SAMPLES samples (as sbi's get_density_thresholder,
without its progress bar), and the network is trained on every round so far
with the plain maximum-likelihood loss (force_first_round_loss), which the
truncation leaves unbiased.  The truncated prior is drawn by rejection from the
prior while at least PRIOR_ACCEPT of its draws are kept; below that (sharp
posteriors in many dimensions) by importance resampling from a defensive
envelope: ENVELOPE_PRIOR of the prior and the rest the posterior q itself,
whose draws fall in the region by its definition (all but `quantile` of them),
however many parameters there are.  The weights prior / envelope of the draws
inside the region are at most 1 / ENVELOPE_PRIOR, and the round's draws are
taken from a pool of at least 10 n of them, without replacement, by weight, the
pool grown until its effective size is 2 n (at most 100 n draws; 'over' 1 when
that was not reached).  This is an approximation of the truncated prior, close
when the pool is large; where the region is mostly a flow's thin tails it is
rough (tests/test_npe_train.py), which only makes that round's proposal a
little off: the loss stays the plain maximum likelihood and DREAM decides.  The
exact rejection costs one log q per prior draw (about 1e7 per round of 2e4 at
the switch).  Each round says which was used and its acceptance (the effective
sample size per draw for the envelope).

RESULT -- NPEResult holds every round's network, the prior, the observed
features x_o, the names and bounds, the data hash, the fit problem
(fit.saved.problem_state), the coarse settings and fidelity report, the git
hash, sbi / torch versions, the seed and each round's statistics.  save()
writes it with torch.save; load_npe() reads it and refuses a problem whose
names, bounds or data differ.  sample() and log_prob() are in u; samples are
proposals for DREAM (fit.npe.seed), never results.

CALIBRATION -- calibrate(): simulation-based calibration of a round's
network (default the amortised round 1) on prior draws with sbi's run_sbc /
check_sbc: the rank of each true parameter among posterior samples should be
uniform; passed when every parameter's KS p-value exceeds 0.05 / d.  Only a
calibrated round-1 network may be used for scenario studies
(scripts/npe_scenarios.py).
"""

import copy
import functools
import json
import time
import warnings

import numpy as np
import sbi
import torch
from scipy.special import expit, logit
from torch import nn
from torch.distributions import Distribution, constraints
from torch.utils.data import default_collate

from sbi.diagnostics import check_sbc, run_sbc
from sbi.inference import NPE, DirectPosterior
from sbi.neural_nets import posterior_nn

from fit.npe.simulator import Simulator, UnitPrior

LOGIT_EPS = 1e-6
EMBEDDING = ((256, 256), 64)
FLOW = dict(model='nsf', hidden_features=64, num_transforms=5)
PRIOR_ACCEPT = 2e-3
ENVELOPE_PRIOR = 0.3
SUPPORT_SAMPLES = 100_000       # posterior samples for the threshold
CHUNK = 100_000                 # rows of one log q evaluation
SBC_PASS = 0.05
FORMAT = 'sanr-npe-1'
# sbi / nflows warnings that say nothing here: the features of draws whose
# critical edge crosses a low-Q point are far from the others (asinh keeps
# them finite), nflows' deprecated call, the SBC batch size
QUIET = ('Data has extreme outliers', 'torch.triangular_solve',
         'Capping max_sampling_batch_size')


def quiet(func):
    """func with the QUIET warnings ignored."""
    @functools.wraps(func)
    def wrapped(*args, **kw):
        with warnings.catch_warnings():
            for m in QUIET:
                warnings.filterwarnings('ignore', message='.*' + m)
            return func(*args, **kw)
    return wrapped


def to_theta(u):
    return logit(np.clip(u, LOGIT_EPS, 1 - LOGIT_EPS))


def to_u(theta):
    return expit(np.asarray(theta, dtype=float))


def log_jacobian(u):
    """log du/dtheta of each row."""
    u = np.clip(u, LOGIT_EPS, 1 - LOGIT_EPS)
    return np.sum(np.log(u) + np.log1p(-u), axis=-1)


class LogitPrior(Distribution):
    """The fit's prior (fit.npe.simulator UnitPrior) in theta = logit(u)."""

    arg_constraints = {}
    support = constraints.independent(constraints.real, 1)
    has_rsample = False

    def __init__(self, unit, seed=None):
        self.unit = unit
        self.rng = np.random.default_rng(seed)
        super().__init__(batch_shape=torch.Size(),
                         event_shape=torch.Size([unit.d]),
                         validate_args=False)

    def sample(self, sample_shape=torch.Size()):
        shape = torch.Size(sample_shape)
        u = self.unit.sample(int(shape.numel()), self.rng)
        return torch.as_tensor(to_theta(u), dtype=torch.float32).reshape(
            *shape, self.unit.d)

    def log_prob(self, theta):
        th = theta.detach().cpu().numpy().astype(float)
        flat = th.reshape(-1, self.unit.d)
        u = to_u(flat)
        lp = self.unit.log_prob(u) + log_jacobian(u)
        return torch.as_tensor(lp.reshape(th.shape[:-1]), dtype=torch.float32)

    @property
    def mean(self):
        return self.sample((10_000,)).mean(0)

    @property
    def variance(self):
        return self.sample((10_000,)).var(0)


class _NoTracker:
    """sbi's Tracker protocol doing nothing: its default writes TensorBoard
    logs into ./sbi-logs."""

    log_dir = None

    def log_metric(self, name, value, step=None):
        pass

    def log_metrics(self, metrics, step=None):
        pass

    def log_params(self, params):
        pass

    def add_figure(self, name, figure, step=None):
        pass

    def flush(self):
        pass


def embedding_net(n_in, hidden=EMBEDDING[0], n_out=EMBEDDING[1]):
    """The MLP of NETWORK."""
    layers, w = [], n_in
    for h in hidden:
        layers += [nn.Linear(w, h), nn.ReLU()]
        w = h
    layers.append(nn.Linear(w, n_out))
    return nn.Sequential(*layers)


def versions():
    return dict(sbi=sbi.__version__, torch=torch.__version__,
                numpy=np.__version__)


# -- truncated proposals ------------------------------------------------------
class Region:
    """The region holding all but `quantile` of the posterior's mass at its
    default x (see ROUNDS): log q of theta, and whether theta is inside."""

    def __init__(self, posterior, quantile, n=SUPPORT_SAMPLES):
        self.posterior = posterior
        with torch.no_grad():
            s = posterior.sample((n,), show_progress_bars=False)
            self.t = float(torch.quantile(posterior.log_prob(s), quantile))

    def log_q(self, theta):
        """log q of each row of theta (numpy), in chunks: a flow's
        activations for millions of rows at once would take gigabytes."""
        out = []
        with torch.no_grad():
            for c in np.array_split(theta, max(1, -(-len(theta) // CHUNK))):
                out.append(self.posterior.log_prob(
                    torch.as_tensor(c, dtype=torch.float32)).numpy())
        return np.concatenate(out).astype(float) if out else np.zeros(0)

    def __call__(self, theta):
        """Inside or not, for a tensor theta (as sbi's thresholder)."""
        with torch.no_grad():
            return self.posterior.log_prob(theta) > self.t


def density_region(posterior, quantile, n=SUPPORT_SAMPLES):
    return Region(posterior, quantile, n)


def truncated_draws(prior, posterior, region, n, rng, pilot=20_000):
    """(u (n, d), info): draws of the prior truncated to the Region
    `region` (see ROUNDS); info {'proposal', 'acceptance', 'over'}."""
    unit = prior.unit

    def inside(theta):
        return region.log_q(theta) > region.t

    u = unit.sample(pilot, rng)
    ok = inside(to_theta(u))
    acc = ok.mean()
    if acc >= PRIOR_ACCEPT:
        got = [u[ok]]
        total, kept = pilot, int(ok.sum())
        while kept < n:
            m = int(min(2 * (n - kept) / acc + 1000, 2_000_000))
            u = unit.sample(m, rng)
            ok = inside(to_theta(u))
            got.append(u[ok])
            total, kept = total + m, kept + int(ok.sum())
        return np.vstack(got)[:n], dict(proposal='prior',
                                        acceptance=kept / total, over=0)
    # the envelope: ENVELOPE_PRIOR of the prior, the rest the posterior
    pool_t, pool_w = [], []
    total = 0
    m = max(10 * n, pilot)
    while True:
        k = int((rng.random(m) < ENVELOPE_PRIOR).sum())
        with torch.no_grad():
            tq = posterior.sample((m - k,), show_progress_bars=False).numpy()
        t = np.vstack([to_theta(unit.sample(k, rng)), tq.astype(float)])
        lq = region.log_q(t)
        ok = lq > region.t
        uk = to_u(t[ok])
        lp = unit.log_prob(uk) + log_jacobian(uk)
        lg = np.logaddexp(np.log(ENVELOPE_PRIOR) + lp,
                          np.log1p(-ENVELOPE_PRIOR) + lq[ok])
        pool_t.append(t[ok])
        pool_w.append(lp - lg)
        total += m
        lw = np.concatenate(pool_w)
        w = np.exp(lw - lw.max()) if lw.size else lw
        ess = w.sum() ** 2 / (w ** 2).sum() if w.size else 0.0
        if ess >= 2 * n or total >= 100 * n:
            break
    if np.count_nonzero(w) < n:
        raise RuntimeError('the truncated prior cannot be drawn: %d '
                           'weighted draws for %d' % (np.count_nonzero(w), n))
    # n of the pool without replacement, by weight (Efraimidis-Spirakis)
    with np.errstate(divide='ignore'):
        key = np.log(rng.random(len(w))) / w
    pick = np.argsort(-key)[:n]
    return to_u(np.concatenate(pool_t)[pick]), dict(
        proposal='envelope', acceptance=float(ess / total),
        over=int(ess < 2 * n))


# -- result -------------------------------------------------------------------
class NPEResult:
    """Every round's network and what it was trained from (see RESULT)."""

    def __init__(self, networks, prior, x_o, meta):
        self.networks = list(networks)
        self.prior = prior
        self.x_o = np.asarray(x_o, dtype=np.float32)
        self.meta = meta
        self.simulator = None      # the training Simulator, if known

    @property
    def names(self):
        return self.meta['names']

    @property
    def rounds(self):
        return self.meta['rounds']

    def posterior(self, round=-1):
        """sbi DirectPosterior of a round's network (in theta), its default
        x the measured data."""
        p = DirectPosterior(self.networks[round], self.prior)
        return p.set_default_x(torch.as_tensor(self.x_o[None]))

    def sample(self, n, round=-1, x=None, seed=None):
        """n samples (n, d) in u of a round's posterior at the features x
        (default: the measured data)."""
        if seed is not None:
            torch.manual_seed(seed)
        x = self.x_o if x is None else np.asarray(x, dtype=np.float32)
        with torch.no_grad():
            th = self.posterior(round).sample(
                (n,), x=torch.as_tensor(x[None]), show_progress_bars=False)
        return to_u(th.numpy())

    def log_prob(self, u, round=-1, x=None):
        """log q(u | x) of each row of u (density in u)."""
        u = np.atleast_2d(u)
        x = self.x_o if x is None else np.asarray(x, dtype=np.float32)
        th = torch.as_tensor(to_theta(u), dtype=torch.float32)
        with torch.no_grad():
            lq = self.posterior(round).log_prob(
                th, x=torch.as_tensor(x[None])).numpy().astype(float)
        return lq - log_jacobian(u)

    def save(self, path):
        torch.save(dict(format=FORMAT, networks=self.networks,
                        unit=self.prior.unit, x_o=self.x_o,
                        meta=json.dumps(self.meta, default=_plain)), path)

    def check(self, problem):
        """ValueError unless problem (a PosteriorProblem, or a FitProblem
        or JointProblem) has these names, bounds and data."""
        from fit.dream.dream_fit import data_hash
        sim = Simulator(problem)
        if sim.names != list(self.names):
            raise ValueError('the network was trained for the parameters '
                             '%s, not %s' % (self.names, sim.names))
        if not (np.array_equal(sim.lo, self.meta['lo']) and
                np.array_equal(sim.hi, self.meta['hi'])):
            raise ValueError('the network was trained with other bounds')
        if data_hash(sim.posterior) != self.meta['data_hash']:
            raise ValueError('the network was trained on other data')

    def posterior_problem(self):
        """The PosteriorProblem it was trained for, rebuilt from the
        stored fit problem (fit.saved) and noise options."""
        from fit.dream.dream_fit import PosteriorProblem
        from fit.saved import problem_from_state
        return PosteriorProblem(problem_from_state(self.meta['problem']),
                                **self.meta.get('noise', {}))


def _plain(obj):
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError('%s is not JSON serialisable' % type(obj).__name__)


def load_npe(path, problem=None):
    """The NPEResult saved at path, checked against problem if given."""
    d = torch.load(path, weights_only=False)
    if not isinstance(d, dict) or d.get('format') != FORMAT:
        raise ValueError('%s is not a SANR NPE file' % path)
    res = NPEResult(d['networks'], LogitPrior(d['unit']), d['x_o'],
                    json.loads(d['meta']))
    if problem is not None:
        res.check(problem)
    return res


# -- training -----------------------------------------------------------------
def _round_stats(inference, t0):
    s = inference._summary
    k = s['epochs_trained'][-1]
    return dict(epochs=int(k),
                best_validation_loss=float(s['best_validation_loss'][-1]),
                validation_loss=[float(v) for v in s['validation_loss'][-k:]],
                training_loss=[float(v) for v in s['training_loss'][-k:]],
                seconds=time.monotonic() - t0)


@quiet
def train(simulator, *, rounds=5, sims_per_round=20_000, truncation=True,
          quantile=1e-4, seed=None, n_workers=1, embedding=EMBEDDING,
          flow=None, training=None, problem=None, coarse=None,
          callback=None, cancelled=None):
    """NPEResult of sequential NPE on simulator (fit.npe.simulator, maybe
    of a coarse problem) for its measured data (see ROUNDS).

    problem: the full-fidelity problem (default: the simulator's), whose
        names, bounds, data hash, noise options and state are stored;
        coarse: the choose_coarse report.
    embedding: (hidden widths, output size) of the MLP; flow: sbi
        posterior_nn arguments (default FLOW); training: NPE.train
        arguments (early stopping on the validation loss by default).
    callback(event, info): 'simulate' (done, total), 'train' (round),
        'epoch' after each epoch of a training ({'round', 'epoch',
        'validation_loss', 'training_loss' (this round's so far),
        'since_best' (epochs without a better validation loss),
        'patience' (after how many it stops)}) and 'round' (that round's
        statistics) as it goes; cancelled() True stops (RuntimeError
        'stopped'), also within a training: it is asked before every
        batch."""
    from fit.dream.dream_fit import _git_hash, data_hash
    from fit.saved import problem_state

    cb = callback or (lambda event, info: None)
    stop = cancelled or (lambda: False)
    rng = np.random.default_rng(seed)
    torch.manual_seed(int(rng.integers(2**31)))
    full = Simulator(problem if problem is not None
                     else simulator.posterior).posterior
    prior = LogitPrior(simulator.prior, int(rng.integers(2**31)))
    x_o = simulator.observed_features()
    builder = posterior_nn(
        z_score_theta='independent', z_score_x='independent',
        embedding_net=embedding_net(len(x_o), *embedding),
        **dict(FLOW, **(flow or {})))
    inference = NPE(prior=prior, density_estimator=builder,
                    show_progress_bars=False, tracker=_NoTracker())
    opts = dict(training or {})
    opts['force_first_round_loss'] = True
    patience = opts.get('stop_after_epochs', 20)    # sbi's default
    live = dict(round=0, start=0, seen=0)        # the epochs said so far

    def report_epoch():
        # a finished epoch not said yet (sbi 0.27 appends its losses to the
        # trainer's _summary as it goes)
        s = getattr(inference, '_summary', {})
        v = s.get('validation_loss', [])
        if len(v) > live['seen']:
            live['seen'] = len(v)
            cb('epoch', dict(
                round=live['round'] + 1, epoch=len(v) - live['start'],
                validation_loss=[float(a) for a in v[live['start']:]],
                training_loss=[float(a) for a in
                               s.get('training_loss', [])[live['start']:]],
                since_best=int(getattr(
                    inference, '_epochs_since_last_improvement', 0)),
                patience=patience))

    def collate(batch):
        # every batch of the training: the one place a stop gets through,
        # and where the epoch before is seen (the last one: after it)
        if stop():
            raise RuntimeError('stopped')
        report_epoch()
        return default_collate(batch)

    opts['dataloader_kwargs'] = dict(opts.get('dataloader_kwargs') or {},
                                     collate_fn=collate)
    meta = dict(names=list(full.names), lo=full.lo, hi=full.hi,
                data_hash=data_hash(full),
                noise=dict(sigma_free=full.sigma_free,
                           floor_free=full.floor_free),
                problem=problem_state(full.fit), coarse=coarse,
                git=_git_hash(), versions=versions(), seed=seed,
                rounds=[], truncation=bool(truncation), quantile=quantile,
                sims_per_round=sims_per_round, embedding=embedding,
                flow=dict(FLOW, **(flow or {})),
                channels=simulator.channel_names, sizes=simulator.sizes,
                scales=simulator.scales, calibration=None)
    networks = []
    posterior = inside = None
    for r in range(rounds):
        if stop():
            raise RuntimeError('stopped')
        t0 = time.monotonic()
        if r == 0 or not truncation:
            u = simulator.prior.sample(sims_per_round, rng)
            info = dict(proposal='prior', acceptance=1.0, over=0)
        else:
            u, info = truncated_draws(prior, posterior, inside,
                                      sims_per_round, rng)
        u, x = simulator.draw(
            u, rng, n_workers=n_workers, cancelled=stop,
            progress=lambda done, total: cb('simulate', (r, done, total)))
        n_failed = sims_per_round - len(u)
        if len(u) < 0.5 * sims_per_round:
            raise RuntimeError('the model failed at %d of %d draws of round '
                               '%d' % (n_failed, sims_per_round, r + 1))
        if stop():
            raise RuntimeError('stopped')
        cb('train', r)
        live.update(round=r, start=len(
            getattr(inference, '_summary', {}).get('validation_loss', [])))
        live['seen'] = live['start']
        inference.append_simulations(
            torch.as_tensor(to_theta(u), dtype=torch.float32),
            torch.as_tensor(x))
        est = inference.train(**opts)
        report_epoch()
        networks.append(copy.deepcopy(est))
        posterior = DirectPosterior(est, prior).set_default_x(
            torch.as_tensor(x_o[None]))
        stats = dict(round=r + 1, n_sims=len(u), n_failed=n_failed, **info,
                     **_round_stats(inference, t0))
        if truncation and r < rounds - 1:
            inside = density_region(posterior, quantile)
        stats['seconds'] = time.monotonic() - t0
        meta['rounds'].append(stats)
        cb('round', stats)
    return NPEResult(networks, prior, x_o, meta)


def run_npe(problem, *, coarse=True, threshold=None, fidelity_draws=100,
            seed=None, n_workers=1, callback=None, cancelled=None, **kw):
    """choose_coarse (when `coarse`), then train on the chosen simulator:
    the NPEResult of problem (a PosteriorProblem, or a FitProblem /
    JointProblem).  kw go to train."""
    from fit.dream.dream_fit import PosteriorProblem
    from fit.npe.simulator import FIDELITY_THRESHOLD, choose_coarse
    cb = callback or (lambda event, info: None)
    full = Simulator(problem).posterior
    report = None
    train_on = full
    if coarse:
        cb('fidelity', 'choosing the training simulator…')
        fit, report = choose_coarse(
            full.fit, FIDELITY_THRESHOLD if threshold is None else threshold,
            fidelity_draws, seed, n_workers,
            progress=lambda text: cb('fidelity', text))
        if fit is not full.fit:              # the coarse model, same noise
            train_on = PosteriorProblem(fit, sigma_free=full.sigma_free,
                                        floor_free=full.floor_free)
        cb('coarse', report)
    sim = Simulator(train_on, seed=seed)
    res = train(sim, seed=seed, n_workers=n_workers, problem=full,
                coarse=report, callback=callback, cancelled=cancelled, **kw)
    res.simulator = sim                     # what it learnt (not saved)
    return res


@quiet
def calibrate(result, simulator, n=300, n_posterior=500, round=0, seed=None,
              n_workers=1):
    """Simulation-based calibration of a round's network (see
    CALIBRATION): {'round', 'n', 'ks_pvals' [per parameter], 'c2st_ranks',
    'c2st_dap', 'passed'}; stored in result.meta['calibration']."""
    rng = np.random.default_rng(seed)
    if seed is not None:
        torch.manual_seed(seed)
    u = simulator.prior.sample(n, rng)
    u, x = simulator.draw(u, rng, n_workers=n_workers)
    theta = torch.as_tensor(to_theta(u), dtype=torch.float32)
    post = DirectPosterior(result.networks[round], result.prior)
    ranks, dap = run_sbc(theta, torch.as_tensor(x), post,
                         num_posterior_samples=n_posterior,
                         show_progress_bar=False)
    check = check_sbc(ranks, theta, dap, num_posterior_samples=n_posterior)
    ks = check['ks_pvals'].numpy().astype(float)
    out = dict(round=round + 1, n=len(u), ks_pvals=ks.tolist(),
               c2st_ranks=check['c2st_ranks'].numpy().astype(float).tolist(),
               c2st_dap=check['c2st_dap'].numpy().astype(float).tolist(),
               passed=bool(np.all(ks > SBC_PASS / len(ks))))
    result.meta['calibration'] = out
    return out
