# cython: language_level=3, boundscheck=False, wraparound=False
# cython: cdivision=True, initializedcheck=False
"""
Compiled transfer-matrix product of model.stack (Stack.build_transfer_matrix):
the slab matrices of Stack._slab_matrices (Majkrzak Eq. 1.128), multiplied
deepest-leftmost, in one loop per Q over the slabs, with no temporary arrays.
Equal to the numpy product to rounding (tests/test_transfer.py).

COLLINEAR -- when every magnetic slab's in-plane M is along one axis (its
angle the same, or half a turn apart: rho -> -rho), each 2x2 block of every
slab matrix is a I + b J with the same J = [[0, mu*], [mu, 0]] (J^2 = 1), so
the product splits into the two scalar eigenchannels, two 2x2 products
instead of one 4x4, and is put back together at the end.

The complex sqrt and exp are written with real functions: Apple's csqrt and
cexp are several times slower.  Built by setup.py (python setup.py
build_ext --inplace); without it model.stack falls back to numpy.
"""

import numpy as np

from libc.math cimport sqrt, hypot, exp, expm1, sin, cos, fabs, copysign, M_PI

ctypedef double complex cplx

# angles closer than this (turns) count as the same axis
COLLINEAR_TOL = 1e-12


cdef inline cplx _csqrt(cplx z) noexcept nogil:
    """Principal square root, the sign of its imaginary part that of z's
    (C99 csqrt, as numpy); the small part from the large one, no
    cancellation."""
    cdef double x = z.real, y = z.imag, m, a, b
    if x == 0 and y == 0:
        return 0
    m = hypot(x, y)
    if x >= 0:
        a = sqrt(0.5*(m + x))
        b = y / (2*a)
    else:
        b = copysign(sqrt(0.5*(m - x)), y)
        a = y / (2*b)
    return a + 1j*b


cdef inline void _chan(cplx S, double d, cplx *c, cplx *sS,
                       cplx *sdS) noexcept nogil:
    """One eigenchannel of a slab: c = cosh(S d), sS = sinh(S d) S,
    sdS = sinh(S d) / S (d at S = 0)."""
    cdef double x = S.real*d, y = S.imag*d, E, ex, shx, chx, cy, sy, n2
    if fabs(x) < 1:
        E = expm1(x)                     # sinh x accurate at small x
        ex = E + 1
        shx = 0.5*(E + E/ex)
    else:
        ex = exp(x)
        shx = 0.5*(ex - 1/ex)
    chx = 0.5*(ex + 1/ex)
    cy = cos(y)
    sy = sin(y)
    cdef cplx sh = shx*cy + 1j*(chx*sy)
    c[0] = chx*cy + 1j*(shx*sy)
    sS[0] = sh*S
    n2 = S.real*S.real + S.imag*S.imag
    if n2 == 0:
        sdS[0] = d
    else:
        sdS[0] = sh*S.conjugate() / n2


cdef void _general(const cplx[::1] nsld, const double[::1] rho,
                   const double[::1] theta, const double[::1] d,
                   const cplx[::1] Q, cplx[:, :, ::1] out) noexcept nogil:
    cdef Py_ssize_t nS = nsld.shape[0], k, s, i, j
    cdef double fp = 4*M_PI, ang
    cdef cplx q2, m, mc, c1, sS1, s1S, c3, sS3, s3S
    cdef cplx cp, cm, sp, sm, tp, tm
    cdef cplx A[16]
    cdef cplx P[16]
    cdef cplx T[16]
    for k in range(Q.shape[0]):
        q2 = Q[k]*Q[k] / 4
        for i in range(16):
            P[i] = 1 if i % 5 == 0 else 0
        for s in range(nS):
            _chan(_csqrt(fp*(nsld[s] + rho[s]) - q2), d[s], &c1, &sS1, &s1S)
            _chan(_csqrt(fp*(nsld[s] - rho[s]) - q2), d[s], &c3, &sS3, &s3S)
            ang = 2*M_PI*theta[s]
            m = cos(ang) + 1j*sin(ang)
            mc = m.conjugate()
            cp = 0.5*(c1 + c3)
            cm = 0.5*(c1 - c3)
            sp = 0.5*(sS1 + sS3)
            sm = 0.5*(sS1 - sS3)
            tp = 0.5*(s1S + s3S)
            tm = 0.5*(s1S - s3S)
            A[0] = cp;    A[1] = mc*cm; A[2] = tp;    A[3] = mc*tm
            A[4] = m*cm;  A[5] = cp;    A[6] = m*tm;  A[7] = tp
            A[8] = sp;    A[9] = mc*sm; A[10] = cp;   A[11] = mc*cm
            A[12] = m*sm; A[13] = sp;   A[14] = m*cm; A[15] = cp
            for i in range(4):
                for j in range(4):
                    T[4*i + j] = (A[4*i]*P[j] + A[4*i + 1]*P[4 + j]
                                  + A[4*i + 2]*P[8 + j]
                                  + A[4*i + 3]*P[12 + j])
            for i in range(16):
                P[i] = T[i]
        for i in range(4):
            for j in range(4):
                out[k, i, j] = P[4*i + j]


cdef inline void _step(cplx *a, cplx c, cplx sS, cplx sdS) noexcept nogil:
    """a <- [[c, sdS], [sS, c]] a for a 2x2 channel product a."""
    cdef cplx t0 = c*a[0] + sdS*a[2], t1 = c*a[1] + sdS*a[3]
    a[2] = sS*a[0] + c*a[2]
    a[3] = sS*a[1] + c*a[3]
    a[0] = t0
    a[1] = t1


cdef void _collinear(const cplx[::1] nsld, const double[::1] rho, double theta,
                     const double[::1] d, const cplx[::1] Q,
                     cplx[:, :, ::1] out) noexcept nogil:
    """rho signed along the common axis at angle theta (see COLLINEAR)."""
    cdef Py_ssize_t nS = nsld.shape[0], k, s, i, j
    cdef double fp = 4*M_PI, ang = 2*M_PI*theta
    cdef cplx m = cos(ang) + 1j*sin(ang), mc = m.conjugate()
    cdef cplx q2, c, sS, sdS, al, be
    cdef cplx a[4]
    cdef cplx b[4]
    for k in range(Q.shape[0]):
        q2 = Q[k]*Q[k] / 4
        a[0] = 1; a[1] = 0; a[2] = 0; a[3] = 1
        b[0] = 1; b[1] = 0; b[2] = 0; b[3] = 1
        for s in range(nS):
            _chan(_csqrt(fp*(nsld[s] + rho[s]) - q2), d[s], &c, &sS, &sdS)
            _step(a, c, sS, sdS)
            if rho[s] != 0:
                _chan(_csqrt(fp*(nsld[s] - rho[s]) - q2), d[s], &c, &sS, &sdS)
            _step(b, c, sS, sdS)
        # block (i, j) of the 4x4: al I + be J
        for i in range(2):
            for j in range(2):
                al = 0.5*(a[2*i + j] + b[2*i + j])
                be = 0.5*(a[2*i + j] - b[2*i + j])
                out[k, 2*i, 2*j] = al
                out[k, 2*i, 2*j + 1] = mc*be
                out[k, 2*i + 1, 2*j] = m*be
                out[k, 2*i + 1, 2*j + 1] = al


def axis(rho, theta):
    """(angle, signed rho) when the magnetic slabs are collinear (see
    COLLINEAR), else None."""
    rho = np.asarray(rho, dtype=float)
    mag = rho != 0
    if not mag.any():
        return 0.0, rho
    t = np.asarray(theta, dtype=float)
    ref = t[mag][0]
    off = np.mod(t - ref + 0.25, 1.0) - 0.25      # in [-0.25, 0.75)
    flip = np.abs(off - 0.5) <= COLLINEAR_TOL
    if not np.all((np.abs(off) <= COLLINEAR_TOL) | flip | ~mag):
        return None
    return float(ref), np.where(flip, -rho, rho)


def product(nsld, rho, theta, d, Q):
    """Product matrix A_N ... A_1 (deepest leftmost) of the slabs (top ->
    bottom) at each Q, shape (nQ, 4, 4); arguments as Stack._slab_matrices."""
    cdef const cplx[::1] n = np.ascontiguousarray(nsld, dtype=np.complex128)
    cdef const double[::1] t = np.ascontiguousarray(theta, dtype=float)
    cdef const double[::1] th = np.ascontiguousarray(d, dtype=float)
    cdef const cplx[::1] q = np.ascontiguousarray(Q, dtype=np.complex128)
    cdef const double[::1] r
    cdef double angle
    out = np.empty((q.shape[0], 4, 4), dtype=np.complex128)
    cdef cplx[:, :, ::1] o = out
    col = axis(rho, theta)
    if col is None:
        r = np.ascontiguousarray(rho, dtype=float)
        with nogil:
            _general(n, r, t, th, q, o)
    else:
        r = np.ascontiguousarray(col[1], dtype=float)
        angle = col[0]
        with nogil:
            _collinear(n, r, angle, th, q, o)
    return out
