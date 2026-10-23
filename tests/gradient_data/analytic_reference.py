"""Exact state and parameter derivatives of the analytic A-B binary (jansson-sensitivities explainer, Section 11), with sympy.

Copied from thoughts/shared/research/jansson-sensitivities/prototypes/cython-proto/verify_R83145.py (R = 8.3145, as
pycalphad); its last section prints the constants in analytic.py.
Run: uv run --no-project --with sympy --with numpy --with scipy python analytic_reference.py
"""
import numpy as np
import sympy as sp

R = 8.3145
T_, *TH = sp.symbols('T th0:5')
# alpha: (A,B)1, L0 = th0 + th1*T, L1 = th2
ya = sp.symbols('ya_A ya_B')
Ga = (R*T_*(ya[0]*sp.log(ya[0]) + ya[1]*sp.log(ya[1]))
      + ya[0]*ya[1]*((TH[0] + TH[1]*T_) + TH[2]*(ya[0] - ya[1])) + 2000*ya[1] - 1.0*T_*ya[1])
# beta: (A,B)1(A,B,VA)1, L(A,B:VA) = th3, G(A:B) = th4 - 2T
yb = sp.symbols('yb_A1 yb_B1 yb_A2 yb_B2 yb_V2')
s1, s2 = yb[:2], yb[2:]
Gend = {('A', 'A'): 9000, ('A', 'B'): TH[4] - 2*T_, ('B', 'A'): 15000, ('B', 'B'): 7000,
        ('A', 'V'): 6200 + 2*T_, ('B', 'V'): 6000}
Gb = sum(s1[i]*s2[j]*Gend[(a, b)] for i, a in enumerate('AB') for j, b in enumerate('ABV'))
Gb += R*T_*sum(y*sp.log(y) for y in yb) + s1[0]*s1[1]*s2[2]*TH[3] + 8000*s2[0]*s2[1]
phases = {
    'alpha': dict(y=ya, G=Ga, m=[ya[0], ya[1]], J=[[1, 1]]),
    'beta': dict(y=yb, G=Gb, m=[yb[0] + yb[2], yb[1] + yb[3]], J=[[1, 1, 0, 0, 0], [0, 0, 1, 1, 1]]),
}
theta0 = np.array([-13800.0, 2.0, 1500.0, -3000.0, -16000.0])
nth, nc = len(TH), 2

# compiled phase functions
for ph in phases.values():
    y, G = ph['y'], ph['G']
    args = (T_, *TH, *y)
    f = lambda e: sp.lambdify(args, e, 'numpy')
    H = G - T_*sp.diff(G, T_)
    ph['n'] = len(y)
    ph['Jm'] = np.array(ph['J'], float)
    ph['G_f'] = f(G)
    ph['H_f'] = f(H)
    ph['Gy_f'] = f(sp.Matrix([sp.diff(G, yi) for yi in y]))
    ph['Gyy_f'] = f(sp.hessian(G, y))
    ph['GT_f'] = f(sp.diff(G, T_))
    ph['GyT_f'] = f(sp.Matrix([sp.diff(G, yi, T_) for yi in y]))
    ph['Gth_f'] = f(sp.Matrix([sp.diff(G, t) for t in TH]))
    ph['Gyth_f'] = f(sp.Matrix([[sp.diff(G, yi, t) for t in TH] for yi in y]))
    ph['Hy_f'] = f(sp.Matrix([sp.diff(H, yi) for yi in y]))
    ph['HT_f'] = f(sp.diff(H, T_))
    ph['Hth_f'] = f(sp.Matrix([sp.diff(H, t) for t in TH]))
    ph['my'] = np.array([[float(sp.diff(mk, yi)) for yi in y] for mk in ph['m']])  # c x n, constant
names = list(phases)
P = [phases[k] for k in names]

def ev(ph, key, T, th, y):
    return np.array(ph[key](T, *th, *y), float).squeeze()

# conditions: N = 1, X(B) = xB, T fixed (T is an unknown with a row T - T0 = 0)
def amount_rows(xB):
    """a.n = r rows: N and X(B)."""
    return np.array([[1.0, 1.0], [-xB, 1.0 - xB]]), np.array([1.0, 0.0])

# full unknown vector: [mu(2), T, per phase: y, eta, F]
def unpack(z):
    mu, T = z[:2], z[2]
    out, k = [], 3
    for ph in P:
        n, s = ph['n'], ph['Jm'].shape[0]
        out.append((z[k:k+n], z[k+n:k+n+s], z[k+n+s]))
        k += n + s + 1
    return mu, T, out

def residual(z, th, T0, xB):
    """Full KKT residual: stationarity, site constraints, hyperplane (mu.m - G), amounts, T row."""
    mu, T, per = unpack(z)
    A, r = amount_rows(xB)
    res, n_el = [], np.zeros(2)
    for ph, (y, eta, F) in zip(P, per):
        res += list(ev(ph, 'Gy_f', T, th, y) - ph['my'].T @ mu - ph['Jm'].T @ eta)
        res += list(ph['Jm'] @ y - 1)
        res.append(mu @ (ph['my'] @ y) - ev(ph, 'G_f', T, th, y))
        n_el += F * (ph['my'] @ y)
    res += list(A @ n_el - r)
    res.append(T - T0)
    return np.array(res)

def jac_fd(fun, z, h=1e-7):
    f0 = fun(z)
    Jm = np.zeros((f0.size, z.size))
    for i in range(z.size):
        dz = np.zeros_like(z); dz[i] = h * max(1.0, abs(z[i]))
        Jm[:, i] = (fun(z + dz) - fun(z - dz)) / (2 * dz[i])
    return Jm

def full_jacobian(z, th, T0, xB):
    """Analytic Jacobian of the full KKT residual in z."""
    mu, T, per = unpack(z)
    A, _ = amount_rows(xB)
    nz = z.size
    Jf = np.zeros((nz, nz))
    row, col = 0, 3
    amt_row = sum(ph['n'] + ph['Jm'].shape[0] + 1 for ph in P)
    for ph, (y, eta, F) in zip(P, per):
        n, s = ph['n'], ph['Jm'].shape[0]
        Hyy = ev(ph, 'Gyy_f', T, th, y)
        Jf[row:row+n, col:col+n] = Hyy
        Jf[row:row+n, 0:2] = -ph['my'].T
        Jf[row:row+n, 2] = ev(ph, 'GyT_f', T, th, y)
        Jf[row:row+n, col+n:col+n+s] = -ph['Jm'].T
        Jf[row+n:row+n+s, col:col+n] = ph['Jm']
        hp = row + n + s
        m = ph['my'] @ y
        Jf[hp, 0:2] = m
        Jf[hp, 2] = -ev(ph, 'GT_f', T, th, y)
        Jf[hp, col:col+n] = ph['my'].T @ mu - ev(ph, 'Gy_f', T, th, y)
        Jf[amt_row:amt_row+2, col:col+n] = F * (A @ ph['my'])
        Jf[amt_row:amt_row+2, col+n+s] = A @ m
        row += n + s + 1
        col += n + s + 1
    Jf[-1, 2] = 1.0
    return Jf

def full_dtheta(z, th, T0, xB):
    """dF/dtheta of the full residual."""
    mu, T, per = unpack(z)
    out, row = np.zeros((z.size, nth)), 0
    for ph, (y, eta, F) in zip(P, per):
        n, s = ph['n'], ph['Jm'].shape[0]
        out[row:row+n] = ev(ph, 'Gyth_f', T, th, y).reshape(n, nth)
        out[row+n+s] = -ev(ph, 'Gth_f', T, th, y)
        row += n + s + 1
    return out

def solve(z, th, T0, xB, tol=1e-11):
    for _ in range(200):
        f = residual(z, th, T0, xB)
        if np.linalg.norm(f) < tol:
            return z
        dz = -np.linalg.solve(full_jacobian(z, th, T0, xB), f)
        _, _, per = unpack(z)
        _, _, dper = unpack(dz)
        a = 1.0
        for (y, _, _), (dy, _, _) in zip(per, dper):
            neg = dy < 0
            if np.any(neg):
                a = min(a, 0.9 * np.min(-y[neg] / dy[neg]))
        z = z + a * dz
    raise RuntimeError(np.linalg.norm(f))

def start(th, T0, xB, nsample=40000):
    """Initial guess from the lower convex hull of sampled phase points (LP)."""
    from scipy.optimize import linprog
    rng = np.random.default_rng(1)
    pts = []
    for i, ph in enumerate(P):
        for _ in range(nsample):
            y = np.concatenate([rng.dirichlet(0.3 * np.ones(len(np.nonzero(r)[0]))) for r in ph['Jm']])
            m = ph['my'] @ y
            pts.append((i, y, m / m.sum(), ev(ph, 'G_f', T0, th, y) / m.sum()))
    Gs = np.array([q[3] for q in pts]); xs = np.array([q[2] for q in pts])
    res = linprog(Gs, A_eq=xs.T, b_eq=[1 - xB, xB], bounds=(0, None), method='highs')
    mu = res.eqlin.marginals
    z = [mu, [T0]]
    for i, ph in enumerate(P):
        act = [k for k in np.nonzero(res.x > 1e-9)[0] if pts[k][0] == i]
        k = max(act, key=lambda k: res.x[k]) if act else None
        y = pts[k][1] if k is not None else np.full(ph['n'], 0.5)
        frac = sum(res.x[k] for k in act)
        z += [y, np.zeros(ph['Jm'].shape[0]), [frac / (ph['my'] @ y).sum()]]
    return np.concatenate(z), [pts[k][0] for k in np.nonzero(res.x > 1e-9)[0]]

T0, xB = 900.0, 0.30
z, active = start(theta0, T0, xB)
print("LP active phases", active, "start", z)
z = solve(z, theta0, T0, xB)
mu, T, per = unpack(z)
print('mu', mu, 'F', [p[2] for p in per])
print('y_alpha', per[0][0], 'y_beta', per[1][0])
assert np.allclose(full_jacobian(z, theta0, T0, xB), jac_fd(lambda q: residual(q, theta0, T0, xB), z), rtol=1e-5, atol=1e-4)

# stability sanity: tangent plane mu.m - G <= 0 on a sampled grid of each phase (global min check)
rng = np.random.default_rng(0)
for ph, nm in zip(P, names):
    ys = [np.concatenate([rng.dirichlet(np.ones(len(np.nonzero(r)[0]))) for r in ph['Jm']]) for _ in range(20000)]
    best = max(mu @ (ph['my'] @ y) - ev(ph, 'G_f', T, theta0, y) for y in ys)
    print(f'max sampled driving force {nm}: {best:.4g} J/fu (<= 0: no point below the tangent plane)')

# ---------- full IFT and central finite differences
Jfull = full_jacobian(z, theta0, T0, xB)
dz_full = -np.linalg.solve(Jfull, full_dtheta(z, theta0, T0, xB))
def fd(fun, h):
    cols = []
    for j in range(nth):
        tp, tm = theta0.copy(), theta0.copy()
        dh = h * max(1.0, abs(theta0[j]))
        tp[j] += dh; tm[j] -= dh
        cols.append((fun(solve(z.copy(), tp, T0, xB), tp) - fun(solve(z.copy(), tm, T0, xB), tm)) / (2 * dh))
    return np.array(cols).T
dz_fd = fd(lambda q, th: q, 1e-5)
print('full IFT vs FD, max rel err:', np.max(np.abs(dz_full - dz_fd)) / np.max(np.abs(dz_fd)))

# ---------- condensed system: unknowns x = [dmu(2), dT, dF_alpha, dF_beta]
def phase_E(ph, T, th, y):
    """yy block of the inverse of the bordered phase matrix [[H, J^T], [J, 0]]."""
    n, s = ph['n'], ph['Jm'].shape[0]
    K = np.zeros((n + s, n + s))
    K[:n, :n] = ev(ph, 'Gyy_f', T, th, y)
    K[:n, n:] = ph['Jm'].T
    K[n:, :n] = ph['Jm']
    return np.linalg.inv(K)[:n, :n]

A, r = amount_rows(xB)
E = [phase_E(ph, T, theta0, y) for ph, (y, _, _) in zip(P, per)]
p = len(P)
m_unk = nc + 1 + p
M = np.zeros((m_unk, m_unk))
S = np.zeros((m_unk, nth))           # dF/dtheta of the condensed residual (b = -S in the code)
bT = np.zeros(m_unk)                  # input T0: row T - T0 -> dF/dT0 = -1
for i, (ph, (y, _, F)) in enumerate(zip(P, per)):
    my, Ei = ph['my'], E[i]
    M[i, :nc] = my @ y
    M[i, nc] = -ev(ph, 'GT_f', T, theta0, y)
    S[i] = -ev(ph, 'Gth_f', T, theta0, y)
    M[p:p+2, :nc] += F * A @ my @ Ei @ my.T
    M[p:p+2, nc] += F * A @ my @ Ei @ (-ev(ph, 'GyT_f', T, theta0, y))
    M[p:p+2, nc+1+i] = A @ (my @ y)
    S[p:p+2] += F * A @ my @ Ei @ (-ev(ph, 'Gyth_f', T, theta0, y).reshape(ph['n'], nth))
M[-1, nc] = 1.0
bT[-1] = -1.0
X = -np.linalg.solve(M, S)            # tangent sensitivities, one column per parameter

# pull the same quantities out of the full IFT
def full_to_condensed(dz):
    mu_, T_, per_ = unpack(dz)
    return np.concatenate([mu_, [T_], [q[2] for q in per_]])
Xfull = np.array([full_to_condensed(dz_full[:, j]) for j in range(nth)]).T
print('condensed vs full IFT (dmu, dT, dF), max abs err:', np.max(np.abs(X - Xfull)))
for i, (ph, (y, _, F)) in enumerate(zip(P, per)):
    Gyth = ev(ph, 'Gyth_f', T, theta0, y).reshape(ph['n'], nth)
    dy = E[i] @ (ph['my'].T @ X[:nc] - np.outer(ev(ph, 'GyT_f', T, theta0, y), X[nc]) - Gyth)
    dy_full = np.array([unpack(dz_full[:, j])[2][i][0] for j in range(nth)]).T
    print(f'  {names[i]} dy/dtheta condensed vs full, max abs err: {np.max(np.abs(dy - dy_full)):.2e}')
    if ph['n'] > 2:
        Hinv = np.linalg.inv(ev(ph, 'Gyy_f', T, theta0, y))
        dy_wrong = Hinv @ (ph['my'].T @ X[:nc] - Gyth)
        print(f'  {names[i]} with plain H^-1: max abs err {np.max(np.abs(dy_wrong - dy_full)):.3e}, '
              f'sublattice sums of dy {ph["Jm"] @ dy_wrong[:, 0]} (must be 0); E_yy sums {ph["Jm"] @ dy[:, 0]}')

# Schur complement of the full Jacobian over each phase's (y, eta) block equals M
idx_y, idx_x, col = [], [0, 1, 2], 3
rows_y, rows_x, row = [], [], 0
for ph in P:
    n, s = ph['n'], ph['Jm'].shape[0]
    idx_y += list(range(col, col + n + s)); idx_x.append(col + n + s)
    rows_y += list(range(row, row + n + s)); rows_x.append(row + n + s)
    col += n + s + 1; row += n + s + 1
rows_x += [row, row + 1, row + 2]
Jxx = Jfull[np.ix_(rows_x, idx_x)]; Jxy = Jfull[np.ix_(rows_x, idx_y)]
Jyx = Jfull[np.ix_(rows_y, idx_x)]; Jyy = Jfull[np.ix_(rows_y, idx_y)]
schur = Jxx - Jxy @ np.linalg.solve(Jyy, Jyx)
# reorder to M's columns [mu, T, F...] (idx_x is [mu, T, F_alpha, F_beta] already) and rows [hp..., amounts, T]
print('Schur complement vs condensed M, max abs err:', np.max(np.abs(schur - M)))

# ---------- outputs Q and their condensed gradients
def outputs(zz, th):
    mu_, T_, per_ = unpack(zz)
    Hsys = sum(F * ev(ph, 'H_f', T_, th, y) for ph, (y, _, F) in zip(P, per_))
    Gsys = sum(F * ev(ph, 'G_f', T_, th, y) for ph, (y, _, F) in zip(P, per_))
    yb_, Fb = per_[1][0], per_[1][2]
    return np.array([mu_[1], Hsys, Fb * (yb_[0] + yb_[1] + yb_[2] + yb_[3]), yb_[4], Gsys])
qnames = ['MU(B)', 'H (system)', 'NP(beta)', 'y(beta,VA#2)', 'G (system)']

def output_gradients():
    """Per output: dQ/dx (condensed, site-fraction terms folded in) and explicit dQ/dtheta."""
    gx, gth = [], []
    for q in range(5):
        g_mu, g_T, g_F = np.zeros(nc), 0.0, np.zeros(p)
        g_y = [np.zeros(ph['n']) for ph in P]
        g_t = np.zeros(nth)
        for i, (ph, (y, _, F)) in enumerate(zip(P, per)):
            if q == 0 and i == 0:
                g_mu[1] = 1.0
            if q == 1:
                g_F[i] = ev(ph, 'H_f', T, theta0, y); g_y[i] = F * ev(ph, 'Hy_f', T, theta0, y)
                g_T += F * ev(ph, 'HT_f', T, theta0, y); g_t += F * ev(ph, 'Hth_f', T, theta0, y)
            if q == 4:
                g_F[i] = ev(ph, 'G_f', T, theta0, y); g_y[i] = F * ev(ph, 'Gy_f', T, theta0, y)
                g_T += F * ev(ph, 'GT_f', T, theta0, y); g_t += F * ev(ph, 'Gth_f', T, theta0, y)
            if i == 1 and q == 2:
                g_F[i] = y[:4].sum(); g_y[i] = F * np.array([1, 1, 1, 1, 0.0])
            if i == 1 and q == 3:
                g_y[i] = np.array([0, 0, 0, 0, 1.0])
        # fold dy = E(my^T dmu - GyT dT - Gyth dtheta) into the condensed gradient
        for i, (ph, (y, _, F)) in enumerate(zip(P, per)):
            Ey = E[i] @ g_y[i]       # E is symmetric
            g_mu += ph['my'] @ Ey
            g_T += -ev(ph, 'GyT_f', T, theta0, y) @ Ey
            g_t += -ev(ph, 'Gyth_f', T, theta0, y).reshape(ph['n'], nth).T @ Ey
        gx.append(np.concatenate([g_mu, [g_T], g_F])); gth.append(g_t)
    return np.array(gx), np.array(gth)

gx, gth = output_gradients()
forward = gth + gx @ X                          # tangent (forward) mode
lam = np.linalg.solve(M.T, gx.T)                # adjoint: one transposed solve per output
adjoint = gth - lam.T @ S
dq_fd = fd(outputs, 1e-5)
print('\nper output: max rel err   forward-vs-FD   adjoint-vs-forward')
for q in range(5):
    sc = np.max(np.abs(dq_fd[q]))
    print(f'  {qnames[q]:14s} {np.max(np.abs(forward[q]-dq_fd[q]))/sc:.2e}   {np.max(np.abs(adjoint[q]-forward[q]))/np.max(np.abs(forward[q])):.2e}')
print('dot-product test g.x_j vs -lambda.s_j:', np.max(np.abs(gx @ X - (-(lam.T @ S)))))

# ---------- the same M serves a condition: dQ/dT0 (heat capacity of the equilibrium for Q=H)
xT = -np.linalg.solve(M, bT)
def outputs_T(h):
    zp = solve(z.copy(), theta0, T0 + h, xB); zm = solve(z.copy(), theta0, T0 - h, xB)
    return (outputs(zp, theta0) - outputs(zm, theta0)) / (2 * h)
print('dQ/dT: tangent', gx @ xT, '\n       FD     ', outputs_T(1e-3))
print('       adjoint', -(lam.T @ bT))

# ---------- shadow prices for Q = G with element-amount rows N_A, N_B
Mn = M.copy(); Sn = S.copy()
An = np.eye(2)
Mn[p:p+2] = 0; Sn[p:p+2] = 0
for i, (ph, (y, _, F)) in enumerate(zip(P, per)):
    my, Ei = ph['my'], E[i]
    Mn[p:p+2, :nc] += F * An @ my @ Ei @ my.T
    Mn[p:p+2, nc] += F * An @ my @ Ei @ (-ev(ph, 'GyT_f', T, theta0, y))
    Mn[p:p+2, nc+1+i] = An @ (my @ y)
    Sn[p:p+2] += F * An @ my @ Ei @ (-ev(ph, 'Gyth_f', T, theta0, y).reshape(ph['n'], nth))
lam_G = np.linalg.solve(Mn.T, gx[4])
print('\nQ = G, rows (hp_alpha, hp_beta, N_A, N_B, T): lambda =', lam_G)
print('   mu =', mu, ' dG/dT =', gx[4] @ xT)
# lambda_phi = implicit part of dQ/d(uniform shift delta of G^phi): check by re-solving with G^phi + delta
lam_all = np.linalg.solve(Mn.T, gx.T)
for i in range(p):
    def shifted(delta):
        old_f = P[i]['G_f']
        P[i]['G_f'] = lambda *a, f=old_f: f(*a) + delta
        zz = solve(z.copy(), theta0, T0, xB)
        P[i]['G_f'] = old_f
        q = outputs(zz, theta0)
        return q, unpack(zz)[2][i][2]
    (qp, Fp), (qm, Fm) = shifted(1e-2), shifted(-1e-2)
    implicit = (qp - qm) / 2e-2    # outputs() uses the unshifted formulas: equilibrium response only
    print(f'  shift {names[i]}: implicit dQ/ddelta FD {implicit},\n      lambda_phi           {lam_all[i]}')
env = sum(F * ev(ph, 'Gth_f', T, theta0, y) for ph, (y, _, F) in zip(P, per))
print('envelope: dG/dtheta = sum F dG/dtheta|_y:', np.max(np.abs(env - forward[4])))
Mfix = Mn[:-1][:, [0, 1, 3, 4]]                 # T fixed: drop T row and column
perm = [2, 3, 0, 1]                              # amounts first, then hyperplanes
print('symmetric after row permutation (N_A, N_B rows, fixed T):', np.allclose(Mfix[perm], Mfix[perm].T))
print('condition number of M:', np.linalg.cond(M), ' size', M.shape)

# ---------- per-phase form: dQ/dtheta = dQ/dtheta|explicit + sum_phi [lam_phi G_theta + z_phi . G_y_theta]
def raw_qy_qth(q):
    """dQ/dy per phase and dQ/dtheta at fixed state (before folding)."""
    qy, qt = [np.zeros(ph['n']) for ph in P], np.zeros(nth)
    for i, (ph, (y, _, F)) in enumerate(zip(P, per)):
        if q == 1:
            qy[i] = F * ev(ph, 'Hy_f', T, theta0, y); qt += F * ev(ph, 'Hth_f', T, theta0, y)
        if q == 4:
            qy[i] = F * ev(ph, 'Gy_f', T, theta0, y); qt += F * ev(ph, 'Gth_f', T, theta0, y)
        if i == 1 and q == 2:
            qy[i] = F * np.array([1, 1, 1, 1, 0.0])
        if i == 1 and q == 3:
            qy[i] = np.array([0, 0, 0, 0, 1.0])
    return qy, qt
for q in range(5):
    qy, qt = raw_qy_qth(q)
    l = lam[:, q]
    tot = qt.copy()
    for i, (ph, (y, _, F)) in enumerate(zip(P, per)):
        z_ = E[i] @ (F * ph['my'].T @ A.T @ l[p:p+2] - qy[i])
        tot += l[i] * ev(ph, 'Gth_f', T, theta0, y) + ev(ph, 'Gyth_f', T, theta0, y).reshape(ph['n'], nth).T @ z_
    print(f'per-phase form {qnames[q]:14s} max abs diff vs adjoint: {np.max(np.abs(tot - adjoint[q])):.2e}')

# ---------- constants for tests/gradient_data/analytic.py
if __name__ == '__main__':
    dys = []
    for i, (ph, (y, _, F)) in enumerate(zip(P, per)):
        Gyth = ev(ph, 'Gyth_f', T, theta0, y).reshape(ph['n'], nth)
        dys.append((E[i] @ (ph['my'].T @ X[:nc] - np.outer(ev(ph, 'GyT_f', T, theta0, y), X[nc]) - Gyth)).T.tolist())
    constants = {
        'MU': mu.tolist(), 'Y': {'ALPHA': per[0][0].tolist(), 'BETA': per[1][0].tolist()},
        'DMU_DTHETA': X[:nc].T.tolist(), 'DY_DTHETA': {'ALPHA': dys[0], 'BETA': dys[1]},
        # alpha has one atom per formula unit; beta's NP(beta) output is in moles of atoms
        'DNP_DTHETA': {'ALPHA': X[nc + 1].tolist(), 'BETA': forward[2].tolist()},
        'DHM_DTHETA': forward[1].tolist(), 'DGM_DTHETA': forward[4].tolist(),
    }
    print('\n# ---------- constants')
    for key, value in constants.items():
        print(f'{key} = {value!r}')
