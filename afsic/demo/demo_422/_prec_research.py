"""Preconditioner research: spectrum of P^{-1}A vs mu_s.

Measures how the eigenvalues of P_LDU^{-1}A spread as mu_s grows, quantifying
the 'dropped-term ~ mu_s dt/rho_s' mechanism that makes the block-LDU stagnate
at stiff solids.  Also reports the FGMRES residual history.
"""
import os, sys
import numpy as np
from scipy.sparse.linalg import eigs, LinearOperator
os.environ.setdefault('NX', '32'); os.environ.setdefault('NY', '32')
os.environ.setdefault('SOLID_H', '0.02')
from config import make_config
from immersed import ImmersedFEM
from linops import MumpsFactor, fgmres
from scipy.sparse import bmat

def build(mu, dt=0.01):
    cfg = make_config(); cfg['num_steps'] = 1; cfg['dt'] = dt; cfg['mu_s'] = mu
    cfg['frozen'] = 2
    s = ImmersedFEM(cfg); s.X[:s.n_u] = s._initial_velocity()
    s.solve_monolithic()
    Xd = s.X.copy()
    s.compute_interaction(Xd[s.n_u + s.n_p:]); s.assemble_mixed_mass()
    W = Xd[s.n_u + s.n_p:]; f_el, A_uW = s.assemble_elastic(W)
    A = s.apply_bc(s.build_monolithic(A_uW))
    n_u, n_p, n_s = s.n_u, s.n_p, s.n_s
    s11 = cfg['p_stab'] * s.Mp
    F = s.apply_bc(bmat([[s.K, s.Bt], [s.B, s11]], format='csr'))
    fl = MumpsFactor(F); ms = MumpsFactor(s.M_s)
    A_s = A[:n_u, n_u + n_p:]; M_wu = A[n_u + n_p:, :n_u]
    def Minv(x): return dt * ms.solve(np.ascontiguousarray(x))
    def p_inv(r):
        r = np.asarray(r, dtype=float)
        zw0 = Minv(r[n_u + n_p:]); yu = r[:n_u] - A_s @ zw0
        zu_zp = fl.solve(np.ascontiguousarray(np.concatenate([yu, r[n_u:n_u + n_p]])))
        return np.concatenate([zu_zp, zw0 + Minv(M_wu @ zu_zp[:n_u])])
    return A, p_inv, Xd, s

for mu in [0.1, 1.0, 10.0, 100.0]:
    A, p_inv, Xd, s = build(mu)
    N = A.shape[0]
    # P^{-1}A as an operator (dense-ish; ARPACK needs only matvec)
    def op(v):
        return p_inv(np.asarray(A @ v, dtype=float))
    L = LinearOperator((N, N), matvec=op, dtype=float)
    # --- spectrum: 12 largest-magnitude eigenvalues ---
    try:
        ev = eigs(L, k=12, which='LM', maxiter=2000, tol=1e-6)
        ev = np.sort(np.abs(ev[0]))
        print(f'mu_s={mu:6}: |eig| min={ev.min():.3f} max={ev.max():.2f} '
              f'(dist from 1: {np.abs(ev - 1).max():.2f})')
    except Exception as e:
        print(f'mu_s={mu:6}: eigs failed: {e}')
    # --- FGMRES residual history on a random rhs ---
    rng = np.random.default_rng(0)
    rhs = A @ (rng.standard_normal(N) * 1e-2)
    hist = []
    x, info = fgmres(A, rhs, M=p_inv, rtol=1e-8, atol=1e-14, restart=50,
                     maxiter=300, callback=lambda it, b: hist.append(b))
    print(f'         FGMRES: info={info} iters={len(hist)} final_beta={hist[-1] if hist else 0:.2e}')
