import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' # Computer Modern math

DATA_FILE = "data/em/chosen_modes/unit_sq_W.csv"    # single file, one column per mode
X_FILE    = "data/em/chosen_modes/unit_sq_X.csv"    # single file, one column per mode

data = np.loadtxt(DATA_FILE, delimiter=",", comments="%")
col13 = 3 # mode 13
col31 = 4 # mode 31

N = int(round(np.sqrt(data.shape[0])))      # flat listing of an N*N grid
x_fem = data[:, 0].reshape(N, N)            # centred: x in [-a, a]
y_fem = data[:, 1].reshape(N, N)            # centred: y in [-b, b]

w_fem13 = data[:, col13].reshape(N, N)
w_fem13 = w_fem13 / np.max(np.abs(w_fem13))

w_fem31 = data[:, col31].reshape(N, N)
w_fem31 = w_fem31 / np.max(np.abs(w_fem31))

print(f"{DATA_FILE}: {N} x {N} grid, "
      f"x in [{x_fem.min():.4f}, {x_fem.max():.4f}], "
      f"y in [{y_fem.min():.4f}, {y_fem.max():.4f}]")
print()

extent = (x_fem.min(), x_fem.max(), y_fem.min(), y_fem.max())


def plot_comparison(w_fem, w_pred, extent, title):
    """FEM / prediction / difference, same layout as em_7."""
    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.imshow(w_fem, extent=extent, origin='lower', vmin=-1, vmax=1, cmap='RdBu_r')
    plt.colorbar(fraction=0.045)
    plt.axis('scaled')
    plt.title('FEM')
    plt.xlabel(r'$\bar{x}$')
    plt.ylabel(r'$\bar{y}$')
    plt.tight_layout()

    plt.subplot(132)
    plt.imshow(w_pred, extent=extent, origin='lower', vmin=-1, vmax=1, cmap='RdBu_r')
    plt.colorbar(fraction=0.045)
    plt.axis('scaled')
    plt.title('Prediction')
    plt.xlabel(r'$\bar{x}$')
    plt.tight_layout()

    plt.subplot(133)
    plt.imshow(w_fem - w_pred, extent=extent, origin='lower', cmap='PiYG')
    plt.colorbar(fraction=0.045)
    plt.axis('scaled')
    plt.title('Difference')
    plt.xlabel(r'$\bar{x}$')
    plt.tight_layout()

    plt.suptitle(title)
    plt.show()


# ======================================================================
# 0. No refit of 13/31: use the mode 11 and mode 33 profiles instead
# ======================================================================
def mode11_profile(X):
    return (1 - 4 * X**2)**2


def mode33_profile(X):
    return (1 - (6.6 * X)**2) * (np.cos(np.pi * X)**2)


def mode13_from_11_33(X, Y):
    phi_x = mode11_profile(X)
    phi_y = mode33_profile(Y) * 0.91
    return phi_x * phi_y


def mode31_from_11_33(X, Y):
    phi_x = mode33_profile(X) * 0.91
    phi_y = mode11_profile(Y)
    return phi_x * phi_y


w_pred13_11_33 = mode13_from_11_33(x_fem, y_fem)
w_pred31_11_33 = mode31_from_11_33(x_fem, y_fem)

err_13_11_33 = 100 * np.sum(np.abs(w_fem13 - w_pred13_11_33)) / np.sum(np.abs(w_fem13))
err_31_11_33 = 100 * np.sum(np.abs(w_fem31 - w_pred31_11_33)) / np.sum(np.abs(w_fem31))
print(f"err 13 (11 x 33 profiles, no refit): {err_13_11_33:.4f} %")
print(f"err 31 (11 x 33 profiles, no refit): {err_31_11_33:.4f} %")

plot_comparison(w_fem13, w_pred13_11_33, extent,
                f"Mode 13, 11 x 33 profiles, no refit (err = {err_13_11_33:.3f} %)")
plot_comparison(w_fem31, w_pred31_11_33, extent,
                f"Mode 31, 11 x 33 profiles, no refit (err = {err_31_11_33:.3f} %)")


# ======================================================================
# 1. Refit modes 13 and 31 (same as em_7)
# ======================================================================
data = np.loadtxt(X_FILE, delimiter=",", comments="%")

x_fem31 = data[:, 2]
x_fem31 = x_fem31 / np.max(np.abs(x_fem31))

x_fem13 = data[:, 3]
x_fem13 = -x_fem13 / np.max(np.abs(x_fem13))


# fit mode 31 with (1-4*x^2)^n and find n
def model_31(x, n):
    # We use np.abs or clip to handle potential precision issues near boundaries
    base = 1 - 4 * x**2
    return np.power(np.maximum(base, 0), n)

x_axis = np.linspace(-0.5, 0.5, N)

try:
    # Initial guess n=1.5 (wider than n=2)
    popt, pcov = curve_fit(model_31, x_axis, x_fem31, p0=[1.5])
    n_fit = popt[0]
    
    print(f"Fitted exponent n: {n_fit:.6f}")
    
    # Generate the fit for plotting
    y_fit = model_31(x_axis, n_fit)
    
    # Calculate relative RMS error
    rms = np.sqrt(np.sum((x_fem31 - y_fit)**2) / np.sum(x_fem31**2))
    print(f"Rel. RMS Error: {rms:.4%}")

except Exception as e:
    print(f"Fit failed: {e}")

# Visual check
plt.plot(x_axis, x_fem31, 'g.', alpha=0.3, label='Target X31')
plt.plot(x_axis, y_fit, 'r-', label=f'Fit (n={n_fit:.3f})')
plt.legend()
plt.show()

def mode_13_model(X, p1):
    # p1: Envelope width (was 6.6 for red)
    phi = (1 - (p1 * X)**2) * (np.cos(np.pi * X)**2)
    return phi

try:
    # Initial guess: p1=5.0 (wider than 6.6)
    popt, pcov = curve_fit(mode_13_model, x_axis, x_fem13, p0=[6.0])
    p1_fit= popt[0]
    
    print(f"Fitted X13 Parameters:")
    print(f"p1 (Envelope): {p1_fit:.4f}")

    y_fit = mode_13_model(x_axis, *popt)
    rms = np.sqrt(np.sum((x_fem13 - y_fit)**2) / np.sum(x_fem13**2))
    print(f"Rel. RMS Error: {rms:.4%}")

except Exception as e:
    print(f"Fit failed: {e}")

# Visual check
plt.plot(x_axis, x_fem13, 'g.', alpha=0.3, label='Target X13')
plt.plot(x_axis, y_fit, 'r-', label=f'Fit ({p1_fit:.3f})')
plt.legend()
plt.show()

def mode13(X,Y):

    phi_x = (1-4*X**2)**1.65
    phi_y = (1 - (6.83*Y)**2) * (np.cos(np.pi*Y)**2)*0.91

    print(np.max(np.abs(phi_x)),np.max(np.abs(phi_y)))

    Phi = (phi_x * phi_y) 
    return Phi

def mode31(X,Y):
    phi_x = (1 - (6.68*X)**2) * (np.cos(np.pi*X)**2) *0.91
    phi_y = (1-4*Y**2)**1.65

    print(np.max(np.abs(phi_x)),np.max(np.abs(phi_y)))

    Phi = (phi_x * phi_y) 
    return Phi

w_pred13 = mode13(x_fem, y_fem)

w_pred31 = mode31(x_fem, y_fem)

err_13 = 100 * np.sum(np.abs(w_fem13 - w_pred13)) / np.sum(np.abs(w_fem13))
print(f"err 13: {err_13:.4f} %")

err_31 = 100 * np.sum(np.abs(w_fem31 - w_pred31)) / np.sum(np.abs(w_fem31))
print(f"err 31: {err_31:.4f} %")

plot_comparison(w_fem13, w_pred13, extent, 'Mode 13, refitted')
plot_comparison(w_fem31, w_pred31, extent, 'Mode 31, refitted')
