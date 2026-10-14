"""
Worker processes that receive the fit problem ONCE, when each starts,
instead of with every task (fit.de.de_fit's pool, fit.dream.dream_fit's).

multiprocessing pickles the mapped function with every task, and a cost
function holds its problem: a joint fit of three datasets with the Licorne
resolution pickles to ~1.2 MB, sent again with every chunk of every DE
generation and with every chain of every DREAM generation.  start() gives
the problem to each process as it starts and returns a token; a wrapper
that holds the problem pickles to that token alone while it is set
(Wrapper), and the worker takes its own copy back (lookup).

The problem goes through a temporary file, its path in the pool's
initializer arguments: a 'spawn' process gets those through a pipe, and
1.2 MB in it made each start wait for the process before to read it (the
pool took ~1.2 s to start instead of returning at once).  stop() ends the
pool and removes the file.
"""

import itertools
import multiprocessing
import os
import pickle
import tempfile

_HELD = {}                        # token -> problem, in a worker process
_FILES = {}                       # token -> its file, in the parent
_TOKENS = itertools.count()


def _hold(token, path):
    with open(path, 'rb') as f:
        _HELD[token] = pickle.load(f)


def start(n, problem):
    """(pool of n 'spawn' processes that each hold `problem`, its token)."""
    token = '%d-%d' % (os.getpid(), next(_TOKENS))
    fd, path = tempfile.mkstemp(prefix='sanr-fit-', suffix='.pickle')
    with os.fdopen(fd, 'wb') as f:
        pickle.dump(problem, f, protocol=pickle.HIGHEST_PROTOCOL)
    _FILES[token] = path
    try:
        pool = multiprocessing.get_context('spawn').Pool(
            n, initializer=_hold, initargs=(token, path))
    except BaseException:
        _remove(token)
        raise
    return pool, token


def stop(pool, token):
    """End a pool from start() at once and remove its file."""
    try:
        pool.terminate()
        pool.join()
    finally:
        _remove(token)


def _remove(token):
    path = _FILES.pop(token, None)
    if path is not None:
        try:
            os.remove(path)
        except OSError:
            pass


def lookup(token):
    """The problem a worker holds for `token`; None elsewhere (Pool.terminate
    unpickles the tasks never sent in the parent, only to drop them)."""
    return _HELD.get(token)


class Wrapper:
    """Base of a picklable function of self.problem: pickled with its
    problem, or with only self.token once it is set (the token of a pool
    from start() holding that problem).  Subclasses list the attributes
    left out of the pickle in DROP (set to None there)."""

    DROP = ()
    token = None

    def __getstate__(self):
        d = dict(self.__dict__)
        for k in self.DROP:
            d[k] = None
        if self.token is not None:
            d['problem'] = None
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        if self.problem is None:
            self.problem = lookup(self.token)
