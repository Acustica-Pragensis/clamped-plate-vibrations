import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' # Computer Modern math

DATA_FILE = "data/em/chosen_modes/unit_sq_W.csv"    # single file, one column per mode

data = np.loadtxt(DATA_FILE, delimiter=",", comments="%")
col = 5 # mode 33

N = int(round(np.sqrt(data.shape[0])))      # flat listing of an N*N grid
x_fem = data[:, 0].reshape(N, N)            # centred: x in [-a, a]
y_fem = data[:, 1].reshape(N, N)            # centred: y in [-b, b]

w_fem = data[:, col].reshape(N, N)
w_fem = w_fem / np.max(np.abs(w_fem))  # unit amplitude

print(f"{DATA_FILE}: {N} x {N} grid, "
      f"x in [{x_fem.min():.4f}, {x_fem.max():.4f}], "
      f"y in [{y_fem.min():.4f}, {y_fem.max():.4f}]")
print()

def mode33(X,Y):

    phi_x = (1 - (6.6*X)**2) * (np.cos(np.pi*X)**2)
    phi_y = (1 - (6.6*Y)**2) * (np.cos(np.pi*Y)**2)

    Phi = (phi_x * phi_y) 
    return Phi


w_pred = mode33(x_fem, y_fem)
# print(np.max(np.abs(w_pred)))
err1 = 100 * np.sum(np.abs(w_fem - w_pred)) / np.sum(np.abs(w_fem))
print(err1)

plt.figure(figsize=(12, 4))
extent = (x_fem.min(), x_fem.max(), y_fem.min(), y_fem.max())

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
# plt.ylabel(r'$\bar{y}$')
plt.tight_layout()

plt.subplot(133)
plt.imshow(w_fem - w_pred, extent=extent, origin='lower',  cmap='PiYG')
plt.colorbar(fraction=0.045)
plt.axis('scaled')
plt.title('Difference')
plt.xlabel(r'$\bar{x}$')
# plt.ylabel(r'$\bar{y}$')
plt.tight_layout()
# plt.savefig(f'mode33_fit.pdf', dpi=300)

plt.show()

from scipy.optimize import minimize_scalar

def objective33(p1_x13, X, Y, target):
    phi_x = (1 - (p1_x13 * X)**2) * (np.cos(np.pi * X)**2)
    phi_y = (1 - (p1_x13 * Y)**2) * (np.cos(np.pi * Y)**2)
    
    Phi = phi_x * phi_y
   
    return 100 * np.sqrt(np.sum((target - Phi)**2) / np.sum(target**2))


# 1. Fit for Mode 33 (Adjusting the 6.6)
res33 = minimize_scalar(objective33, bounds=(4, 10), args=(x_fem, y_fem, w_fem), method='bounded')
best_66 = res33.x


print(f"Optimized coefficient for Mode 33 (replaced 6.6): {best_66:.6f}")
print(f"Final Error 1: {res33.fun:.4f}%")