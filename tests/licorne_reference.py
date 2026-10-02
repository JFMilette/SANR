"""Licorne 1.4.2 reference (test oracle): a line-by-line Python port of the
MATLAB code path (CodeOpt=1) -- expandrough.m, roughsublayer.m and
reflection_m.m (reflection_s, spin_av, resolut) -- plus reflection.cpp
resolut mode 1.  Validated against Licorne's own profile_sublayers.dat
exports (fixture1, v127_chi3_137, v127_r2_6_508) to their 6 significant
figures.  Used only by the tests; never imported by model/."""
import numpy as np
from scipy.special import erfinv, erf

A1 = 2*erfinv(0.5); A2 = 2*np.arctanh(0.5); D1 = erfinv(0.97); D2 = np.arctanh(0.97)


def _int(a, b, coeff, g):                 # int_tanh / int_erf: 50-pt rectangle
    x = np.linspace(a, b, 50)
    return np.sum(g(coeff*x))*(x[1]-x[0])


def roughsublayer(tf, Sigma, N, fun):
    tf = np.array(tf, dtype=complex)
    Sigma = Sigma*1.3
    g, a, d = (erf, A1, D1) if fun == 'erf' else (np.tanh, A2, D2)
    ta, tb = tf[0, 0].real, tf[1, 0].real
    t0 = tN1 = None
    La = d*Sigma/a; Lb = La; Coeff = a/Sigma
    if La < ta/2: t0 = ta/2-La; Ls_a = 1
    else: La = ta/2; Ls_a = 0
    if Lb < tb/2: tN1 = tb/2-Lb; Ls_b = 1
    else: Lb = tb/2; Ls_b = 0
    x = np.linspace(-La+(La+Lb)/(2*N), Lb-(La+Lb)/(2*N), N)
    t = np.ones(N)*(La+Lb)/N
    fa, fb = tf[0, 1:].copy(), tf[1, 1:].copy()
    if not ((ta == tb) or (Ls_a and Ls_b)):
        I = _int(-La, Lb, Coeff, g)
        if ta > tb: fb = fa+2*Lb*(fb-fa)/(Lb+La+I)
        else:       fa = fb-2*La*(fb-fa)/(Lb+La-I)
    f = np.zeros((N, tf.shape[1]-1), dtype=complex)
    for k in range(tf.shape[1]-1):
        if k < 2:                          # nsld and |M| smeared
            f[:, k] = fa[k] if fa[k] == fb[k] else (fb[k]-fa[k])*(g(Coeff*x)+1)/2+fa[k]
        else:                              # angles: step at x = 0
            f[:, k] = np.where(x > 0, tf[1, k+1], tf[0, k+1])
    out = np.column_stack([t, f])
    if t0 is not None: out = np.vstack([np.r_[t0, tf[0, 1:]], out])
    if tN1 is not None: out = np.vstack([out, np.r_[tN1, tf[1, 1:]]])
    return out


def _row(L):
    return np.r_[L['thickness'], L['nsld'], L['msld']].astype(complex)


def _smooth(L):
    return L['roughness'] == 0 or L['fun'] == 'NC'


def expandrough(layers, sub):
    """Returns array rows [t, nsld, rho, phi_deg, theta_deg]."""
    L = layers; N = len(L)
    if _smooth(L[0]):
        tf1 = _row(L[0])[None, :]
    else:
        tf = [np.r_[L[0]['thickness']*1.5, 0, 0, L[0]['msld'][1], L[0]['msld'][2]], _row(L[0])]
        tf1 = roughsublayer(tf, L[0]['roughness'], L[0]['nbound'], L[0]['fun'])
    MM = []
    same = lambda r, Lk: (r[1] == Lk['nsld'] and r[2] == Lk['msld'][0] and
                          r[3] == Lk['msld'][1] and r[4] == Lk['msld'][2])
    for k in range(1, N):
        a, b = L[k-1], L[k]
        if _smooth(a) and _smooth(b):
            MM.append(tf1); tf1 = _row(b)[None, :]
        elif not _smooth(a) and _smooth(b):
            if same(tf1[-1], a):
                tf1 = tf1.copy(); tf1[-1, 0] += a['thickness']/2; MM.append(tf1)
            else:
                MM.append(np.vstack([tf1, np.r_[a['thickness']/2, a['nsld'], a['msld']]]))
            tf1 = _row(b)[None, :]
        elif not _smooth(a) and not _smooth(b):
            tf2 = tf1
            tf1 = roughsublayer([_row(a), _row(b)], b['roughness'], b['nbound'], b['fun'])
            if np.all(tf1[0, 1:] == tf2[-1, 1:]):
                tf2 = tf2.copy(); tf2[-1, 0] += tf1[0, 0]; tf1 = tf1[1:]
            MM.append(tf2)
        else:
            tf1 = roughsublayer([_row(a), _row(b)], b['roughness'], b['nbound'], b['fun'])
            if same(tf1[0], a):
                tf1 = tf1.copy(); tf1[0, 0] += a['thickness']/2
            else:
                MM.append(np.r_[a['thickness']/2, a['nsld'], a['msld']][None, :])
    LN = L[-1]
    subrow = np.r_[LN['thickness']*1.5, sub['nsld'], 0, LN['msld'][1], LN['msld'][2]]
    if _smooth(LN) and _smooth(sub):
        MM.append(tf1)
    elif not _smooth(LN) and _smooth(sub):
        if same(tf1[-1], LN):
            tf1 = tf1.copy(); tf1[-1, 0] += LN['thickness']/2; MM.append(tf1)
        else:
            MM.append(np.vstack([tf1, np.r_[LN['thickness']/2, LN['nsld'], LN['msld']]]))
    elif not _smooth(LN) and not _smooth(sub):
        tf2 = tf1
        tf1 = roughsublayer([_row(LN), subrow], sub['roughness'], sub['nbound'], sub['fun'])
        if np.all(tf1[0, 1:] == tf2[-1, 1:]):
            tf2 = tf2.copy(); tf2[-1, 0] += tf1[0, 0]; tf1 = tf1[1:]
        MM.append(tf2); MM.append(tf1)
    else:
        tf1 = roughsublayer([_row(LN), subrow], sub['roughness'], sub['nbound'], sub['fun'])
        if same(tf1[0], LN):
            tf1 = tf1.copy(); tf1[0, 0] += LN['thickness']/2
        else:
            MM.append(np.r_[LN['thickness']/2, LN['nsld'], LN['msld']][None, :])
        MM.append(tf1)
    M = np.vstack(MM)
    if M[0, 1] == 0 and M[0, 2] == 0 and not _smooth(L[0]): M = M[1:]
    if M[-1, 1] == sub['nsld'] and M[-1, 2] == 0 and not _smooth(sub): M = M[:-1]
    return M


def reflection_s(q, rows, sub_nsld):
    """Supermatrix of reflection_m.m. rows: [t, nsld, rho, phi_deg, theta_deg]."""
    k = q/2+0j; k2 = k**2
    sm = np.sqrt(k2-4*np.pi*sub_nsld)
    I2 = np.eye(2)
    S = np.broadcast_to(np.eye(4, dtype=complex), (len(q), 4, 4)).copy()
    pauli = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]
    for r in rows:
        th, n = r[0].real, r[1]
        rho, ph, te = r[2].real, np.radians(r[3].real), np.radians(r[4].real)
        B = 4*np.pi*rho*np.array([np.sin(te)*np.cos(ph), np.sin(te)*np.sin(ph), np.cos(te)])
        A = 4*np.pi*n; Bm = np.linalg.norm(B)

        def F(fun):
            if Bm == 0:
                v = fun(A); return v[:, None, None]*I2
            vp, vm = fun(A+Bm), fun(A-Bm)
            Fp, Fm = (vp+vm)/2, (vp-vm)/(2*Bm)
            Bs = sum(B[i]*pauli[i] for i in range(3))
            return Fp[:, None, None]*I2+Fm[:, None, None]*Bs
        mom = lambda X: np.sqrt(k2-X)
        Ms = F(lambda X: np.sin(mom(X)*th)); Mc = F(lambda X: np.cos(mom(X)*th))
        M12 = F(lambda X: (k2-X)**-0.5) @ Ms
        M21 = -F(mom) @ Ms
        Mt = np.zeros((len(q), 4, 4), complex)
        Mt[:, :2, :2] = Mc; Mt[:, :2, 2:] = M12; Mt[:, 2:, :2] = M21; Mt[:, 2:, 2:] = Mc
        S = Mt @ S
    S11, S12, S21, S22 = S[:, :2, :2], S[:, :2, 2:], S[:, 2:, :2], S[:, 2:, 2:]
    v = lambda a: a[:, None, None]
    Down = v(1j*sm)*S11+v(k*sm)*S12-S21+v(1j*k)*S22
    Up = v(-1j*sm)*S11+v(k*sm)*S12+S21+v(1j*k)*S22
    return np.linalg.inv(Down) @ Up


def spin_av(R, n1, n2, pe=1.0, ae=1.0):
    s = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]
    d1 = np.eye(2)+pe*sum(n1[i]*s[i] for i in range(3))
    d2 = np.eye(2)+ae*sum(n2[i]*s[i] for i in range(3))
    Rh = np.conj(np.swapaxes(R, 1, 2))
    return np.real(np.trace(d1 @ Rh @ d2 @ R, axis1=1, axis2=2))/4


def resolut3(RR, q, dq):
    """res_mode 3 ('dQ centred to own centres'), interior points only
    (the edge points of the MATLAB code are left as computed there)."""
    N = len(q); out = RR.copy(); ps = np.sqrt(2*np.pi)
    for ii in range(2, N-2):
        dqc = abs(q[ii+1]-q[ii-1])/2
        if dq[ii] < dqc/2: continue
        sp = dq[ii]*ps; ss = 2*dq[ii]**2; ts = 3*dq[ii]
        acc = RR[ii]*dqc
        k = 1; qq = abs(q[ii]-q[ii-1]); dl = abs(q[ii]-q[ii-2])/2; Rk = RR[ii-1]
        while qq <= ts:
            acc += Rk*np.exp(-qq*qq/ss)*dl; k += 1; ik = ii-k
            if ik < 1: dl = abs(q[1]-q[0]); qq += dl; Rk = RR[0]
            else: qq = abs(q[ii]-q[ik]); dl = abs(q[ik+1]-q[ik-1])/2; Rk = RR[ik]
        k = 1; qq = abs(q[ii+1]-q[ii]); dl = abs(q[ii+2]-q[ii])/2; Rk = RR[ii+1]
        while qq <= ts:
            acc += Rk*np.exp(-qq*qq/ss)*dl; k += 1; ik = ii+k
            if ik > N-2: dl = abs(q[N-1]-q[N-2]); qq += dl; Rk = RR[N-1]
            else: qq = abs(q[ik]-q[ii]); dl = abs(q[ik+1]-q[ik-1])/2; Rk = RR[ik]
        out[ii] = acc/sp
    return out


def resolut(RR, q, dq, res_mode):
    """Full port of reflection_m.m `resolut` (modes 2, 3) and reflection.cpp
    mode 1 (the MATLAB mode-1 loop indexes with the imaginary unit and
    crashes).  Written with 1-based arrays so it reads line by line against
    the MATLAB source."""
    N = len(q); Nm1 = N-1
    q = np.r_[np.nan, q]; dq = np.r_[np.nan, dq]; RR = np.r_[np.nan, RR]
    RRr = RR.copy()
    if res_mode == 1:
        den = np.zeros(N+1); RRr = np.zeros(N+1)
        for ii in range(2, N+1):
            k = 1
            while q[ii]-q[ii-k] <= dq[ii]/2:
                den[ii] += 1; RRr[ii] += RR[ii-k]; k += 1
                if ii-k < 1: break
        den[N] += 1; RRr[N] = (RRr[N]+RR[N])/den[N]
        for ii in range(1, N):
            k = 1; den[ii] += 1; RRr[ii] += RR[ii]
            while q[ii+k]-q[ii] <= dq[ii]/2:
                den[ii] += 1; RRr[ii] += RR[ii+k]; k += 1
                if ii+k > N: break
            RRr[ii] /= den[ii]
        return RRr[1:]
    ps = np.sqrt(2*np.pi)
    if res_mode == 2:
        for ii in range(2, Nm1+1):
            dqc = abs(q[ii+1]-q[ii-1])/2
            if dq[ii] < dqc/2: RRr[ii] = RR[ii]; continue
            sp = dq[ii]*ps; ss = 2*dq[ii]**2
            RRr[ii] = RR[ii]*dqc/sp
            k = 1; qq = abs(q[ii]-q[ii-1]); dl = qq; Rk = RR[ii-k]; ts = 3*dq[ii]
            while qq <= ts:
                RRr[ii] += Rk*np.exp(-qq*qq/ss)*dl/sp; k += 1
                if ii-k < 1: dl = abs(q[2]-q[1]); qq += dl; Rk = RR[1]
                else: qq = abs(q[ii]-q[ii-k]); dl = abs(q[ii-k+1]-q[ii-k]); Rk = RR[ii-k]
            k = 1; qq = abs(q[ii+1]-q[ii]); dl = qq; Rk = RR[ii+k]
            while qq <= ts:
                RRr[ii] += Rk*np.exp(-qq*qq/ss)*dl/sp; k += 1
                if ii+k > N: dl = abs(q[N]-q[Nm1]); qq += dl; Rk = RR[N]
                else: qq = abs(q[ii+k]-q[ii]); dl = abs(q[ii+k]-q[ii+k-1]); Rk = RR[ii+k]
        # first point
        dqc = abs(q[2]-q[1])
        if dq[1] < dqc/2: RRr[1] = RR[1]
        else:
            sp = dq[1]*ps; ss = 2*dq[1]**2; RRr[1] = RR[1]*dqc/sp
            k = 2; qq = abs(q[2]-q[1]); ts = 3*dq[1]
            while qq <= ts:
                RRr[1] += RR[k]*np.exp(-qq*qq/ss)*abs(q[k]-q[k-1])/sp; k += 1
                if k > N: break
                qq = abs(q[k]-q[1])
            qq = abs(q[2]-q[1]); dl = qq
            while qq <= ts:
                RRr[1] += RR[1]*np.exp(-qq*qq/ss)*dl/sp; qq += dl
        # last point
        dqc = abs(q[N]-q[Nm1])
        if dq[N] < dqc/2: RRr[N] = RR[N]
        else:
            sp = dq[N]*ps; ss = 2*dq[N]**2; RRr[N] = RR[N]*dqc/sp
            k = 1; qq = abs(q[N]-q[Nm1]); ts = 3*dq[N]
            while qq <= ts:
                RRr[N] += RR[N-k]*np.exp(-qq*qq/ss)*abs(q[N-k+1]-q[N-k])/sp; k += 1
                if N-k < 1: break
                qq = q[N]-q[N-k]
            qq = abs(q[N]-q[Nm1]); dl = qq
            while qq <= ts:
                RRr[N] += RR[N]*np.exp(-qq*qq/ss)*dl/sp; qq += dl
        return RRr[1:]
    if res_mode == 3:
        for ii in range(3, Nm1):
            dqc = abs(q[ii+1]-q[ii-1])/2
            if dq[ii] < dqc/2: RRr[ii] = RR[ii]; continue
            sp = dq[ii]*ps; ss = 2*dq[ii]**2
            RRr[ii] = RR[ii]*dqc
            k = 1; qq = abs(q[ii]-q[ii-1]); dl = abs(q[ii]-q[ii-2])/2; Rk = RR[ii-k]; ts = 3*dq[ii]
            while qq <= ts:
                RRr[ii] += Rk*np.exp(-qq*qq/ss)*dl; k += 1; ik = ii-k
                if ik < 2: dl = abs(q[2]-q[1]); qq += dl; Rk = RR[1]
                else: qq = abs(q[ii]-q[ik]); dl = abs(q[ik+1]-q[ik-1])/2; Rk = RR[ik]
            k = 1; qq = abs(q[ii+1]-q[ii]); dl = abs(q[ii+2]-q[ii])/2; Rk = RR[ii+k]
            while qq <= ts:
                RRr[ii] += Rk*np.exp(-qq*qq/ss)*dl; k += 1; ik = ii+k
                if ik > N-1: dl = abs(q[N]-q[Nm1]); qq += dl; Rk = RR[N]
                else: qq = abs(q[ik]-q[ii]); dl = abs(q[ik+1]-q[ik-1])/2; Rk = RR[ik]
            RRr[ii] /= sp
        # first point
        dqc = abs(q[2]-q[1])
        if dq[1] < dqc/2: RRr[1] = RR[1]
        else:
            sp = dq[1]*ps; ss = 2*dq[1]**2; RRr[1] = RR[1]*dqc/sp
            k = 2; qq = abs(q[2]-q[1]); ts = 3*dq[1]
            while qq <= ts:
                RRr[1] += RR[k]*np.exp(-qq*qq/ss)*(abs(q[k+1]-q[k-1])/2)/sp; k += 1
                if k > N-1: break
                qq = abs(q[k]-q[1])
            qq = abs(q[2]-q[1]); dl = qq
            while qq <= ts:
                RRr[1] += RR[1]*np.exp(-qq*qq/ss)*dl/sp; qq += dl
        # second point
        dqc = abs(q[3]-q[1])/2
        if dq[2] < dqc/2: RRr[2] = RR[2]
        else:
            sp = dq[2]*ps; ss = 2*dq[2]**2; RRr[2] = RR[2]*dqc/sp
            k = 3; qq = abs(q[3]-q[2]); ts = 3*dq[2]
            while qq <= ts:
                RRr[2] += RR[k]*np.exp(-qq*qq/ss)*(abs(q[k+1]-q[k-1])/2)/sp; k += 1
                if k > N-1: break
                qq = abs(q[k]-q[2])
            qq = abs(q[2]-q[1]); dl = abs(q[3]-q[1])/2
            while qq <= ts:
                RRr[2] += RR[1]*np.exp(-qq*qq/ss)*dl/sp; qq += dl
        # point before last
        dqc = abs(q[N]-q[Nm1-1])/2
        if dq[N-1] < dqc/2: RRr[N-1] = RR[N-1]
        else:
            sp = dq[N-1]*ps; ss = 2*dq[N-1]**2; RRr[N-1] = RR[N-1]*dqc/sp
            k = 2; qq = abs(q[N-1]-q[Nm1-1]); ts = 3*dq[N-1]
            while qq <= ts:
                RRr[N-1] += RR[N-k]*np.exp(-qq*qq/ss)*(abs(q[N-k+1]-q[N-k-1])/2)/sp; k += 1
                if N-k < 2: break
                qq = abs(q[N-1]-q[N-k])
            qq = abs(q[N]-q[Nm1]); dl = abs(q[N]-q[Nm1-1])/2
            while qq <= ts:
                RRr[N-1] += RR[N]*np.exp(-qq*qq/ss)*dl/sp; qq += dl
        # last point
        dqc = abs(q[N]-q[Nm1])
        if dq[N] < dqc/2: RRr[N] = RR[N]
        else:
            sp = dq[N]*ps; ss = 2*dq[N]**2; RRr[N] = RR[N]*dqc/sp
            k = 1; qq = abs(q[N]-q[Nm1]); ts = 3*dq[N]
            while qq <= ts:
                RRr[N] += RR[N-k]*np.exp(-qq*qq/ss)*(abs(q[N-k+1]-q[N-k-1])/2)/sp; k += 1
                if N-k < 2: break
                qq = abs(q[N]-q[N-k])
            qq = abs(q[N]-q[Nm1]); dl = qq
            while qq <= ts:
                RRr[N] += RR[N]*np.exp(-qq*qq/ss)*dl/sp; qq += dl
        return RRr[1:]
    raise ValueError('res_mode must be 1, 2 or 3')


def licorne_R(q, dq, layers, sub, pol, an, norm=1.0, background=0.0, res_mode=3):
    """End to end, as reflection_m.m: expandrough -> supermatrix -> spin_av
    -> resolut -> *Norm_factor + Background.  pol / an: Licorne 3-vectors."""
    R = reflection_s(np.asarray(q, float), expandrough(layers, sub), sub['nsld'])
    RR = spin_av(R, pol, an)
    return resolut(RR, np.asarray(q, float), np.asarray(dq, float), res_mode)*norm + background
