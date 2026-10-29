"""
Fit problems as plain JSON, for the fit and DREAM tabs a session keeps
(fit.panel FitPanel.fit_tabs_state) and the runs "Save chains…" writes
(fit.dream.dream_fit.save_result).

problem_state gives what a fit.problem.FitProblem or fit.joint.JointProblem
was built from (FitProblem.inputs: every channel's points as given, dQ
included, the Q range, Licorne grid and error floor; its cost and target;
its stack as the fit had it, i.e. with the thickness spread's node count
frozen, and the models of a mixture, which Stack.to_dict leaves out; the
names of the model it was taken from and of the dataset fitted,
FitProblem.model_name and dataset_name), and
problem_from_state builds the same problem again.  Arrays become lists;
NaN and ±inf stay floats (Python's json reads its own NaN and Infinity).
"""

import numpy as np

from fit.joint import JointProblem
from fit.problem import FitProblem
from model.stack import Stack


def plain(v):
    """v with numpy arrays and scalars (in dicts, lists and tuples) as
    plain lists and numbers."""
    if isinstance(v, dict):
        return {k: plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, (np.ndarray, np.generic)):
        return v.tolist()
    return v


def stack_state(st):
    """Stack.to_dict with the linked mixture models (Stack._mix_stacks)."""
    d = st.to_dict()
    if st._mix_stacks:
        d['mix_stacks'] = {n: m.to_dict() for n, m in st._mix_stacks.items()}
    return d


def stack_from_state(d):
    d = dict(d)
    mix = d.pop('mix_stacks', {})
    st = Stack.from_dict(d)
    st._mix_stacks = {n: Stack.from_dict(m) for n, m in mix.items()}
    return st


def _array(v):
    return None if v is None else np.asarray(v, dtype=float)


def problem_state(problem):
    """Plain-JSON state of a FitProblem or JointProblem (module
    docstring)."""
    if isinstance(problem, JointProblem):
        return dict(parts=[[n, problem_state(p)] for n, p
                           in zip(problem.part_names, problem.parts)],
                    modes=problem.modes, fixed=plain(problem.fixed),
                    sources=problem.sources)
    i = problem.inputs
    return plain(dict(stack=stack_state(problem.stack), data=i['data'],
                      model=problem.model_name,
                      dataset=problem.dataset_name,
                      cost=problem.cost, target=problem.target,
                      qmin=float(i['qmin']), qmax=float(i['qmax']),
                      grid=i['grid'], error_floor=i['error_floor']))


def problem_from_state(d):
    """Inverse of problem_state."""
    if 'parts' in d:
        return JointProblem([(n, problem_from_state(p)) for n, p in d['parts']],
                            d['modes'], d['fixed'], d.get('sources'))
    data = [dict(e, **{k: _array(e.get(k)) for k in
                       ('Q', 'R', 'dR', 'dQ', 'P0', 'P')})
            for e in d['data']]
    problem = FitProblem(stack_from_state(d['stack']), data, d['cost'],
                         d['qmin'], d['qmax'], _array(d['grid']),
                         d['error_floor'], d['target'])
    problem.model_name = d.get('model')
    problem.dataset_name = d.get('dataset')
    return problem
