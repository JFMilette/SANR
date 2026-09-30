"""Independent implementation of Ruehm, Toperverg, Dosch PRB 60, 16073 (1999):
Eq. 7 (supermatrix via matrix functions of p-hat), Eq. 9 product, Eq. 10 R-hat,
Eq. 2 / Eq. 5 reflectivity with density matrices."""
import numpy as np
from scipy.linalg import cosm, sinm, inv
sx = np.array([[0,1],[1,0]],complex); sy = np.array([[0,-1j],[1j,0]]); sz = np.array([[1,0],[0,-1]],complex)
sig = np.array([sx,sy,sz]); I2 = np.eye(2)

def S_slab(p0, N, rho, Bdir, d):
    # q_c-hat^2 = 4 pi (N + rho B0.sigma)  (spin || B sees N+rho)
    qc2 = 4*np.pi*(N*I2 + rho*np.tensordot(Bdir, sig, 1))
    w_, V_ = np.linalg.eig(p0**2*I2 - qc2 + 0j); p = V_ @ np.diag(np.sqrt(w_)) @ inv(V_)
    c, s = cosm(p*d), sinm(p*d)
    return np.block([[c, inv(p)@s], [-p@s, c]])          # Eq. 7

def R_hat(p0, layers, N_sub):
    S = np.eye(4, dtype=complex)
    for (N, rho, B, d) in layers:                         # top -> bottom
        S = S_slab(p0, N, rho, B, d) @ S                  # S_tot = S_n ... S_1
    S11, S12, S21, S22 = S[:2,:2], S[:2,2:], S[2:,:2], S[2:,2:]
    ps = np.sqrt(p0**2 - 4*np.pi*N_sub + 0j)
    D = (S22 + 1j/p0*S21)*p0 + (S11 - 1j*p0*S12)*ps
    return inv(D) @ ((S22 - 1j/p0*S21)*p0 - (S11 + 1j*p0*S12)*ps)   # Eq. 10

def refl_eq5(R, P0, P):
    R0 = np.trace(R); Rv = np.array([np.trace(R@s) for s in sig])
    P0 = np.asarray(P0, float); P = np.asarray(P, float)
    return (1/8*(abs(R0)**2*(1+P0@P) + np.vdot(Rv,Rv).real*(1-P0@P))
            + 1/4*np.real(np.conj(R0)*Rv@(P0+P) + (np.conj(Rv)@P0)*(Rv@P))
            - 1/4*np.imag(np.conj(R0)*Rv@np.cross(P0,P)
                          + 0.5*np.cross(np.conj(Rv),Rv)@(P0-P)))

def refl_eq2(R, P0, P):
    r0 = 0.5*(I2 + np.tensordot(P0, sig, 1)); r = 0.5*(I2 + np.tensordot(P, sig, 1))
    return np.trace(r @ R @ r0 @ R.conj().T).real
