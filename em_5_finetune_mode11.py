import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' # Computer Modern math

DATA_FILE = "data/em/chosen_modes/unit_sq_W.csv"    # single file, one column per mode

data = np.loadtxt(DATA_FILE, delimiter=",", comments="%")
col = 2 # mode 11

N = int(round(np.sqrt(data.shape[0])))      # flat listing of an N*N grid
x_fem = data[:, 0].reshape(N, N)            # centred: x in [-a, a]
y_fem = data[:, 1].reshape(N, N)            # centred: y in [-b, b]

w_fem = data[:, col].reshape(N, N)
w_fem = w_fem / np.max(np.abs(w_fem))  # unit amplitude

print(f"{DATA_FILE}: {N} x {N} grid, "
      f"x in [{x_fem.min():.4f}, {x_fem.max():.4f}], "
      f"y in [{y_fem.min():.4f}, {y_fem.max():.4f}]")
print()

def mode11(X,Y):

    phi_x = (1-4*X**2)**2
    phi_y = (1-4*Y**2)**2

    Phi = (phi_x * phi_y)
    return Phi

w_pred = mode11(x_fem, y_fem)
err1 = 100 * np.sum(np.abs(w_fem - w_pred)) / np.sum(np.abs(w_fem))

print(err1)

plt.figure(figsize=(12, 4))
extent = (-0.5, 0.5, -0.5, 0.5)

plt.subplot(131)
plt.imshow(w_fem, extent=extent, origin='lower',  vmin=-1, vmax=1, cmap='RdBu_r')
plt.colorbar(fraction=0.045)
plt.axis('scaled')
# plt.ylabel('V1')
plt.title('FEM')
plt.tight_layout()
plt.xlabel(r'$\bar{x}$')
plt.ylabel(r'$\bar{y}$')
plt.tight_layout()


plt.subplot(132)
plt.imshow(w_pred, extent=extent, origin='lower',  vmin=-1, vmax=1, cmap='RdBu_r')
plt.colorbar(fraction=0.045)
plt.axis('scaled')
plt.title('Prediction')
plt.tight_layout()
plt.xlabel(r'$\bar{x}$')
# plt.ylabel('$y/(2b)$')
plt.tight_layout()

plt.subplot(133)
plt.imshow(w_fem - w_pred, extent=extent, origin='lower',  cmap='PiYG')
plt.colorbar(fraction=0.045)
plt.axis('scaled')
plt.title('Difference')
plt.tight_layout()
plt.xlabel(r'$\bar{x}$')
# plt.ylabel('$y/(2b)$')
plt.tight_layout()

# plt.savefig(f'mode11_fit.pdf', dpi=300)

plt.show()
