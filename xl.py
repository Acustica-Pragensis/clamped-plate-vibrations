"""Xing & Liu (2009) symplectic solution for fully clamped plates.

Shared by the eigenmode scripts.  The plate is centred at the origin with
sides Lx x Ly; Xing & Liu measure from a clamped edge, so the 1-D profiles
are written in x_edge = x + Lx/2 and y_edge = y + Ly/2.
"""

import csv
from pathlib import Path

import numpy as np
from scipy.optimize import root as _root

QMAX = 18.0
NSEED = 40
MODES_CSV = Path('data/comsol_modes.csv')


def _sech(t):
    """Overflow-safe 1/cosh(t) for t > 0."""
    return 2.0 * np.exp(-t) / (1.0 + np.exp(-2.0 * t))


def xl_eqs(q, Lx, Ly):
    """Residuals of the two CCCC eigenvalue equations (Table 2 of Xing & Liu),
    divided by cosh(b2*Lx) resp. cosh(a2*Ly) so they stay finite for higher
    modes.  q = (beta1, alpha1).

    The pair of equations is closed by  k^2 = beta1^2 + alpha1^2,  i.e.
        beta2^2  = beta1^2 + 2 alpha1^2 ,   alpha2^2 = alpha1^2 + 2 beta1^2 .
    """
    b1, a1 = q
    k = np.hypot(b1, a1)
    b2 = np.sqrt(k * k + a1 * a1)               # beta2^2  = k^2 + alpha1^2
    a2 = np.sqrt(k * k + b1 * b1)               # alpha2^2 = k^2 + beta1^2
    Fx = (b1 * b1 - b2 * b2) * np.sin(b1 * Lx) * np.tanh(b2 * Lx) \
        + 2.0 * b1 * b2 * (np.cos(b1 * Lx) - _sech(b2 * Lx))
    Fy = (a1 * a1 - a2 * a2) * np.sin(a1 * Ly) * np.tanh(a2 * Ly) \
        + 2.0 * a1 * a2 * (np.cos(a1 * Ly) - _sech(a2 * Ly))
    return np.array([Fx, Fy])


def xl_eigenpairs(Lx, Ly, qmax=QMAX, nseed=NSEED):
    """All CCCC eigenpairs (beta1, alpha1) with 0 < beta1, alpha1 < qmax,
    sorted by k.  Grid of initial guesses + Newton; the Jacobian is nearly
    diagonal, so the basins of attraction are wide."""
    found = []
    for s1 in np.linspace(0.5, qmax, nseed):
        for s2 in np.linspace(0.5, qmax, nseed):
            sol = _root(xl_eqs, [s1, s2], args=(Lx, Ly), method='hybr', tol=1e-13)
            if not sol.success:
                continue
            b1, a1 = sol.x
            if not (0.3 < b1 < qmax + 1.0 and 0.3 < a1 < qmax + 1.0):
                continue                                  # beam limit
            if np.max(np.abs(xl_eqs(sol.x, Lx, Ly))) > 1e-7 * (1.0 + b1 * b1 + a1 * a1):
                continue
            if not any(abs(b1 - B1) < 1e-6 and abs(a1 - A1) < 1e-6 for B1, A1 in found):
                found.append((b1, a1))
    found.sort(key=lambda p: np.hypot(p[0], p[1]))
    return found


def xl_profiles(b1, a1, Lx, Ly):
    """1-D profile functions X(x_edge), Y(y_edge), in the equivalent form of
    Rdzanek et al. (3b)-(4a): X(0) = X'(0) = 0.  Returns X, Y and k."""
    k = np.hypot(b1, a1)
    b2 = np.sqrt(k * k + a1 * a1)
    a2 = np.sqrt(k * k + b1 * b1)
    kb = (np.cos(b1 * Lx) - np.cosh(b2 * Lx)) / ((b2 / b1) * np.sin(b1 * Lx) - np.sinh(b2 * Lx))
    ka = (np.cos(a1 * Ly) - np.cosh(a2 * Ly)) / ((a2 / a1) * np.sin(a1 * Ly) - np.sinh(a2 * Ly))

    def X(xe):
        return (np.cos(b1 * xe) - np.cosh(b2 * xe)
                - kb * ((b2 / b1) * np.sin(b1 * xe) - np.sinh(b2 * xe)))

    def Y(ye):
        return (np.cos(a1 * ye) - np.cosh(a2 * ye)
                - ka * ((a2 / a1) * np.sin(a1 * ye) - np.sinh(a2 * ye)))

    return X, Y, k


def xl_halfwaves(f, L, n=20001):
    """Number of half-waves = number of sign changes on (0, L) + 1, i.e. the
    mode index m or n of Eqs. (42)-(45)."""
    v = f(np.linspace(0.0, L, n))
    signs = np.sign(np.where(np.abs(v) > 1e-6 * np.max(np.abs(v)), v, 0.0))
    signs = signs[signs != 0.0]
    return int(np.count_nonzero(np.diff(signs) != 0)) + 1


def xl_modes_for_plate(Lx, Ly, modes, qmax=QMAX, nseed=NSEED):
    """Map requested 'mn' half-wave tags to (beta1, alpha1, k) at Lx x Ly."""
    out = {}
    for b1, a1 in xl_eigenpairs(Lx, Ly, qmax=qmax, nseed=nseed):
        Xf, Yf, k = xl_profiles(b1, a1, Lx, Ly)
        tag = f"{xl_halfwaves(Xf, Lx)}{xl_halfwaves(Yf, Ly)}"
        if tag in modes:
            out.setdefault(tag, (b1, a1, k))            # lowest k per (m, n)

    missing = [tag for tag in modes if tag not in out]
    if missing:
        raise RuntimeError(f"eigenpairs not found for modes {missing}: "
                           f"increase qmax/nseed in xl_eigenpairs")
    return out


XL_CACHE = {}


def xl_pairs(ba, qmax=QMAX, nseed=NSEED):
    """Eigenpairs at Lx = 1, Ly = ba, sorted by k. Solved once per ratio."""
    key = round(ba, 6)
    if key not in XL_CACHE:
        XL_CACHE[key] = xl_eigenpairs(1.0, key, qmax=qmax, nseed=nseed)
    return XL_CACHE[key]


def xl_field(pair, ba, Xn, Yn):
    """Product-form field on the unit plate.

    Xn, Yn in [-0.5, 0.5].  X profile lives on [0, 1] (Lx = 1), the Y
    profile on [0, ba] (Ly = ba), so Yn must be scaled by ba."""
    b1, a1 = pair
    Xf, Yf, _ = xl_profiles(b1, a1, 1.0, ba)
    w = Xf(Xn + 0.5) * Yf((Yn + 0.5) * ba)
    m = np.max(np.abs(w))
    return w / m if m > 0 else w


def load_mode_table(path=MODES_CSV):
    """{ba: [f11, f12, ...]} in ascending frequency order (column order)."""
    with open(path, newline='') as fh:
        rows = list(csv.reader(fh))
    out = {}
    for r in rows[1:]:
        if not r or not r[0].strip():
            continue
        ba = round(float(r[0]), 6)
        labels = [c.strip() for c in r[1:]
                  if c.strip() and c.strip().lower() != 'nan']
        out[ba] = labels
    return out


TABLE = load_mode_table()
ASSIGN_CACHE = {}


def xl_assign(ba):
    """{label: (pair, k)} -- i-th label gets the i-th smallest k.

    Valid because f ~ k^2 for fixed material and thickness, so the COMSOL
    frequency order and the Xing & Liu k order are the same order."""
    key = round(ba, 6)
    if key in ASSIGN_CACHE:
        return ASSIGN_CACHE[key]
    if key not in TABLE:
        raise KeyError(f'ba={key} not in {MODES_CSV}')

    pairs = xl_pairs(key)
    labels = TABLE[key]
    if len(pairs) < len(labels):
        raise RuntimeError(f'ba={key}: {len(labels)} labels but only '
                           f'{len(pairs)} eigenpairs -- raise qmax/nseed')

    out = {lab: (pairs[i], float(np.hypot(*pairs[i])))
           for i, lab in enumerate(labels)}
    ASSIGN_CACHE[key] = out
    return out


def xl_mode_frequencies(a, ba, h, E, nu, rho, modes=None):
    """Eigenfrequencies [Hz] of the labelled modes from the X&L eigenvalues.

    a  : half side along x [m] (the data column a)
    ba : aspect ratio b/a
    h  : plate thickness [m]
    E, nu, rho : material constants (Pa, -, kg/m^3)

    xl_assign() works with Lx = 1, so its k is dimensionless; the physical
    wavenumber is k/Lx with Lx = 2a.  With k^4 = rho*h*omega^2/D this gives
        f = k^2 / (2 pi Lx^2) * sqrt(D / (rho h)).
    """
    a, ba, h = float(a), float(ba), float(h)
    Lx = 2.0 * a
    D = E * h ** 3 / (12.0 * (1.0 - nu ** 2))
    scale = np.sqrt(D / (rho * h)) / (2.0 * np.pi * Lx ** 2)

    assign = xl_assign(ba)
    if modes is not None:
        assign = {lab: assign[lab] for lab in modes}
    return {lab: k * k * scale for lab, (_pair, k) in assign.items()}
