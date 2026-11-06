# =============================================================================
#  CCCC unit plate: present closed-form approximations, versus the
#  Xing & Liu (2009) symplectic solution, both compared with the FEM export
# =============================================================================

# ------------------------------------------------------- 1. imports, parameters
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from xl import xl_eigenpairs, xl_modes_for_plate, xl_profiles


matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'      # Computer Modern math


DATA_FILE = "data/em/chosen_modes/unit_sq_W.csv"    # single file, one column per mode
MODES = ['11', '13', '31', '33']    # modes to compare
extent = (-0.5, 0.5, -0.5, 0.5)     # plate centred at the origin


# --------------------------------------------------------------- 2. data loading
data = np.loadtxt(DATA_FILE, delimiter=",", comments="%")

N = int(round(np.sqrt(data.shape[0])))      # flat listing of an N*N grid
x_fem = data[:, 0].reshape(N, N)            # centred: x in [-a, a]
y_fem = data[:, 1].reshape(N, N)            # centred: y in [-b, b]

fem_fields = {}                             # 'mn' -> FEM field, unit amplitude
for mn, col in zip(MODES, [2, 3, 4, 5]):    # column order of the header
    w = data[:, col].reshape(N, N)
    fem_fields[mn] = w / np.max(np.abs(w))  # unit amplitude

print(f"{DATA_FILE}: {N} x {N} grid, "
      f"x in [{x_fem.min():.4f}, {x_fem.max():.4f}], "
      f"y in [{y_fem.min():.4f}, {y_fem.max():.4f}]")
print()


# FEM eigenfrequencies of the unit plate, ascending as returned by COMSOL
FEM_FREQ_HZ = np.array([
    73.65207189674939,
    150.2072757099038,
    150.20728790506138,
    221.4627460801601,
    269.266151328439,
    270.5453400394452,
    337.6381050062418,
    337.63820147095754,
    430.75949944116195,
    430.75960467808244,
    450.2147736953175,
    495.4572480109068,
])

# which entry of FEM_FREQ_HZ is which mode (1-based index)
FEM_MODE_INDEX = {'11': 1, '13': 5, '31': 6, '33': 11}
# ------------------------------------------- 3. our approximations, Eqs. (42)-(45)
def mode11(X, Y):
    phi_x = (1 - 4 * X**2)**2
    phi_y = (1 - 4 * Y**2)**2
    Phi = (phi_x * phi_y)
    return Phi


def mode13(X, Y):
    phi_x = (1 - 4 * X**2)**1.65
    phi_y = (1 - (6.83 * Y)**2) * (np.cos(np.pi * Y)**2) / 1.1
    Phi = (phi_x * phi_y)
    return Phi


def mode31(X, Y):
    phi_x = (1 - (6.83 * X)**2) * (np.cos(np.pi * X)**2) / 1.1
    phi_y = (1 - 4 * Y**2)**1.65
    Phi = (phi_x * phi_y)
    return Phi


def mode33(X, Y):
    phi_x = (1 - (6.6 * X)**2) * (np.cos(np.pi * X)**2)
    phi_y = (1 - (6.6 * Y)**2) * (np.cos(np.pi * Y)**2)
    Phi = (phi_x * phi_y)
    return Phi


own_fields = {
    '11': mode11(x_fem, y_fem),
    '13': mode13(x_fem, y_fem),
    '31': mode31(x_fem, y_fem),
    '33': mode33(x_fem, y_fem),
}


# ------------------------------------------- 4. Xing & Liu (2009) solution, CCCC
#  Geometry: plate centred at the origin with sides 2a x 2b, i.e.
#  x in [-a, a], y in [-b, b]; Xing & Liu measure from a clamped edge, so
#  x_edge = x + a in [0, Lx] with Lx = 2a, Ly = 2b.
a_half = np.max(np.abs(x_fem))              # a (half side)
b_half = np.max(np.abs(y_fem))              # b
Lx, Ly = 2.0 * a_half, 2.0 * b_half         # full sides, Xing & Liu's a and b
x_edge = x_fem + a_half
y_edge = y_fem + b_half


# validation against the paper's own geometry
print("Self-test of the Xing & Liu solver against their Table 5 (a=1, b=1.2):")
ref_ka = [5.49, 7.42, 8.30, 9.56, 9.76, 11.31, 11.39, 12.23, 12.23, 13.52]
ref_b1 = [4.33, 3.90, 7.72, 7.45, 3.68, 10.93, 7.21, 10.77, 3.56, 7.04]
pairs_test = xl_eigenpairs(1.0, 1.2)
for i in range(min(5, len(pairs_test))):
    b1, a1 = pairs_test[i]
    print(f"  mode {i+1}:  ka = {np.hypot(b1, a1):5.2f} (ref {ref_ka[i]:5.2f})"
          f"    b1*a = {b1 * 1.0:5.2f} (ref {ref_b1[i]:5.2f})")
print()


# eigenpairs of the plate under study
XL_MODES = xl_modes_for_plate(Lx, Ly, MODES)   # 'mn' -> (beta1, alpha1, k)

print(f"Xing & Liu eigenvalues, plate {Lx:.3f} x {Ly:.3f} (2a x 2b):")
for mn in MODES:
    b1, a1, k = XL_MODES[mn]
    b2 = np.sqrt(k * k + a1 * a1)
    a2 = np.sqrt(k * k + b1 * b1)
    print(f"  mode {mn}:  k*2a = {k * Lx:6.3f}   k = {k:6.3f}"
          f"   b1 = {b1:6.3f}   b2 = {b2:6.3f}"
          f"   a1 = {a1:6.3f}   a2 = {a2:6.3f}")
print()

# separability diagnostic: X(x)*Y(y) is not an exact solution for CCCC
def _d2(f, x, h=1e-5):
    return (f(x + h) - 2.0 * f(x) + f(x - h)) / h ** 2


b1, a1, k = XL_MODES['11']
Xf, Yf, _ = xl_profiles(b1, a1, Lx, Ly)
xg = np.linspace(0.15 * Lx, 0.85 * Lx, 8)
yg = np.linspace(0.15 * Ly, 0.85 * Ly, 8)
c1 = _d2(Xf, xg) / Xf(xg)
c2 = _d2(Yf, yg) / Yf(yg)
print("Separability check (mode 11): X(x)*Y(y) solves the biharmonic equation only")
print("if X''/X and Y''/Y are separately constant, which a clamped profile is not.")
print(f"  X''/X on [0.15, 0.85]*2a:  min {c1.min():8.2f}  max {c1.max():8.2f}"
      f"   (-b1^2 = {-b1**2:8.2f})")
print(f"  Y''/Y on [0.15, 0.85]*2b:  min {c2.min():8.2f}  max {c2.max():8.2f}"
      f"   (-a1^2 = {-a1**2:8.2f})")
print()

# mode shapes
xl_fields = {}
for mn in MODES:
    b1, a1, k = XL_MODES[mn]
    Xf, Yf, _ = xl_profiles(b1, a1, Lx, Ly)
    w = Xf(x_edge) * Yf(y_edge)
    w /= np.max(np.abs(w))                      # eigenfunctions have free amplitude
    if np.sum(w * fem_fields[mn]) < 0:
        w = -w                                  # ... and a free sign
    xl_fields[mn] = w

# ------------------------------------------------- 5. eigenvalue comparison
# Xing & Liu give a dimensionless wavenumber; the FEM export gives frequencies.
# Since k^4 = rho*h*omega^2/D, k scales as sqrt(f), hence
#     (k*2a)_n / (k*2a)_11 = sqrt(f_n / f_11),
# which needs no material data.  The absolute scale is anchored on the
# Xing & Liu value for mode 11, so the table tests how the FEM spectrum is
# *spaced*; any common offset of the whole spectrum is absorbed by the anchor.
k11_xl = XL_MODES['11'][2] * Lx
f11 = FEM_FREQ_HZ[FEM_MODE_INDEX['11'] - 1]

print("Eigenvalues, k*2a (2a = full side, anchored on mode 11):")
print(f"  {'mode':>4} {'f_fem [Hz]':>13} {'k*2a FEM':>10} {'k*2a X&L':>10}"
      f" {'diff':>9} {'f_n/f_11':>9}")
for mn in MODES:
    f = FEM_FREQ_HZ[FEM_MODE_INDEX[mn] - 1]
    k_fem = k11_xl * np.sqrt(f / f11)
    k_xl = XL_MODES[mn][2] * Lx
    print(f"  {mn:>4} {f:13.3f} {k_fem:10.3f} {k_xl:10.3f}"
          f" {100 * (k_xl - k_fem) / k_fem:8.2f}% {f / f11:9.4f}")
print()

for mn in MODES:
    print(f"  max|Eqs.(42)-(45)| mode {mn}: {np.max(np.abs(own_fields[mn])):.4f}")


# ------------------------------------------------------------ 5. error measures
def err_w(w_fem, w_pred):
    return 100 * np.sum(np.abs(w_fem - w_pred)) / np.sum(np.abs(w_fem))

print(f"Mode-shape errors, % of the FEM field")
print(f"  {'mode':>4} | {'Eqs.(42)-(45)':>14} | {'Xing & Liu':>12}")
for mn in MODES:
    fem = fem_fields[mn]
    print(f"  {mn:>4} | {err_w(fem, own_fields[mn]):13.3f}% | "
          f"{err_w(fem, xl_fields[mn]):11.3f}%")

# ----------------------------------------------------------------- 6. figures
for mn in MODES:
    fem = fem_fields[mn]
    own = own_fields[mn]
    w_xl = xl_fields[mn]

    d_own = fem - own
    d_xl = fem - w_xl
    d_xo = w_xl - own
    v_lim = max(np.max(np.abs(d_own)), np.max(np.abs(d_xl)), np.max(np.abs(d_xo)))
    norm_sym = Normalize(vmin=-v_lim, vmax=v_lim)      # 0 is white in PiYG_r

    plt.figure(figsize=(17, 4))

    plt.subplot(1, 4, 1)
    im1 = plt.imshow(fem, extent=extent, vmin=-1, vmax=1, origin='lower', cmap='RdBu_r')
    plt.title('FEM')
    plt.xlabel(r'$\bar{x}$', fontsize=12)
    plt.ylabel(r'$\bar{y}$', fontsize=12)
    plt.colorbar(im1, fraction=0.045)

    plt.subplot(1, 4, 2)
    im2 = plt.imshow(d_own, extent=extent, origin='lower', cmap='PiYG_r', norm=norm_sym)
    plt.title('FEM $-$ Eqs. (42)-(45)')
    plt.xlabel(r'$\bar{x}$', fontsize=12)
    plt.colorbar(im2, fraction=0.045, format='%.3f')

    plt.subplot(1, 4, 3)
    im3 = plt.imshow(d_xl, extent=extent, origin='lower', cmap='PiYG_r', norm=norm_sym)
    plt.title('FEM $-$ Xing & Liu')
    plt.xlabel(r'$\bar{x}$', fontsize=12)
    plt.colorbar(im3, fraction=0.045, format='%.3f')

    plt.subplot(1, 4, 4)
    im4 = plt.imshow(d_xo, extent=extent, origin='lower', cmap='PiYG_r', norm=norm_sym)
    plt.title('Xing & Liu $-$ Eqs. (42)-(45)')
    plt.xlabel(r'$\bar{x}$', fontsize=12)
    plt.colorbar(im4, fraction=0.045, format='%.3f')

    plt.tight_layout()
    # plt.savefig(f'figs/mode{mn}_xl.png', dpi=300)
    plt.show()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    im1 = plt.imshow(fem, extent=extent, vmin=-1, vmax=1, origin='lower', cmap='RdBu_r')
    # plt.title('FEM')
    plt.xlabel(r'$\bar{x}$', fontsize=12)
    plt.ylabel(r'$\bar{y}$', fontsize=12)
    plt.colorbar(im1, fraction=0.045)

    plt.subplot(1, 3, 2)
    im2 = plt.imshow(d_own, extent=extent, origin='lower', cmap='PiYG_r', norm=norm_sym)
    # plt.title('FEM $-$ Eqs. (42)-(45)')
    plt.xlabel(r'$\bar{x}$', fontsize=12)
    plt.colorbar(im2, fraction=0.045, format='%.3f')

    plt.subplot(1, 3, 3)
    im3 = plt.imshow(d_xl, extent=extent, origin='lower', cmap='PiYG_r', norm=norm_sym)
    # plt.title('FEM $-$ Xing & Liu')
    plt.xlabel(r'$\bar{x}$', fontsize=12)
    plt.colorbar(im3, fraction=0.045, format='%.3f')

    plt.tight_layout()
    plt.savefig(f'figs/em/mode{mn}_xl.pdf', dpi=300)
    plt.show()