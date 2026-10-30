"""
Neural posterior estimation (NPE) as a front end for DREAM(ZS)
(plans/npe_plan_v1.md): a network trained on simulations of the fit
problem proposes every region of parameter space compatible with the data;
DREAM with the exact likelihood (fit.dream.dream_fit) then samples the
posterior from there.  The network only proposes, DREAM decides: no result
is ever reported from the network's samples alone.

  simulator  Simulator: prior draws, the fit problem's model, the
             likelihood's noise, features; coarse_problem / fidelity /
             choose_coarse: a cheaper training simulator with a measured gap
  train      sequential NPE with truncated proposals (sbi), NPEResult,
             save / load_npe, calibrate (simulation-based calibration)
  seed       seed_chains: DREAM starting states spread over the modes the
             network found, with the exact posterior
  symmetry   probe: which transformations of the parameters leave the
             model unchanged (mirror modes)
  npe_window the NPE tab of the main window (GUI)

torch and sbi are optional (requirements-npe.txt): only train (and what
calls it) imports them, so the rest of SANR runs without them; available()
says whether they are installed.
"""

import importlib.util

REQUIRED = ('torch', 'sbi', 'sklearn')


def missing():
    """The optional packages NPE needs that are not installed."""
    return [m for m in REQUIRED if importlib.util.find_spec(m) is None]


def available():
    """torch, sbi and scikit-learn are installed."""
    return not missing()
