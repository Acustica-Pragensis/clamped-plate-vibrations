import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' # Computer Modern math

# ------------------------------------------------------- 1. imports, parameters
DATA_FILE = "data/em/chosen_modes/unit_sq_W.csv"    # 2D fields, one column per mode
X_FILE = "data/em/chosen_modes/unit_sq_X.csv"       # 1D profiles along x
Y_FILE = "data/em/chosen_modes/unit_sq_Y.csv"       # 1D profiles along y
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

w_fem11 = fem_fields['11']
w_fem13 = fem_fields['13']
w_fem31 = fem_fields['31']
w_fem33 = fem_fields['33']

# ---- 1D profiles: data/unit_sq_X.csv and data/unit_sq_Y.csv
#   header: % cln1x,W11,W13,W31,W33
#   column 0 = coordinate, columns 1-4 = W11, W13, W31, W33
prof_x = np.loadtxt(X_FILE, delimiter=",", comments="%")
prof_y = np.loadtxt(Y_FILE, delimiter=",", comments="%")

x_axis = prof_x[:, 0]               # x of the extracted profiles
y_axis = prof_y[:, 0]               # y of the extracted profiles


x_11 = prof_x[:, 1] / np.max(np.abs(prof_x[:, 1]))
x_13 = prof_x[:, 2] / np.max(np.abs(prof_x[:, 2]))
x_31 = -prof_x[:, 3] / np.max(np.abs(prof_x[:, 3]))
x_33 = prof_x[:, 4] / np.max(np.abs(prof_x[:, 4]))

y_11 = prof_y[:, 1] / np.max(np.abs(prof_y[:, 1]))
y_13 = prof_y[:, 2] / np.max(np.abs(prof_y[:, 2]))
y_31 = -prof_y[:, 3] / np.max(np.abs(prof_y[:, 3]))
y_33 = prof_y[:, 4] / np.max(np.abs(prof_y[:, 4]))

print(f"{X_FILE}: {prof_x.shape[0]} points, "
      f"x in [{x_axis.min():.4f}, {x_axis.max():.4f}]")
print(f"{Y_FILE}: {prof_y.shape[0]} points, "
      f"y in [{y_axis.min():.4f}, {y_axis.max():.4f}]")
print()
x_axis = np.linspace(-0.5, 0.5, N)


wr_11 = np.dot(y_11.reshape(-1, 1), x_11.reshape(1, -1))
wr_13 = np.dot(y_13.reshape(-1, 1), x_13.reshape(1, -1))
wr_31 = np.dot(y_31.reshape(-1, 1), x_31.reshape(1, -1))
wr_33 = np.dot(y_33.reshape(-1, 1), x_33.reshape(1, -1))


x11_fit = (1-4*x_axis**2)**2
x13_fit = (1-4*x_axis**2)**1.58
x31_fit = (1 - (6.83*x_axis)**2) * (np.cos(np.pi*x_axis)**2)/1.1
x33_fit = (1 - (6.6*x_axis)**2) * (np.cos(np.pi*x_axis)**2) 

y11_fit = (1-4*x_axis**2)**2
y13_fit = (1 - (6.83*x_axis)**2) * (np.cos(np.pi*x_axis)**2)/1.1
y31_fit = (1-4*x_axis**2)**1.58
y33_fit = (1 - (6.6*x_axis)**2) * (np.cos(np.pi*x_axis)**2)


modes = [
    ('11', w_fem11, wr_11),  # Mode name, FEM data, Reconstruction, vmin
    ('13', w_fem13, wr_13),
    ('31', w_fem31, wr_31),
    ('33', w_fem33, wr_33)
]
from matplotlib.colors import Normalize
for mn, fem, rec in modes:
    plt.figure(figsize=(12, 4))
    
    # 1. FEM Plot
    plt.subplot(1, 3, 1)
    im1 = plt.imshow(fem, extent=extent, vmin=-1, vmax=1, origin='lower', cmap='RdBu_r')
    plt.title(f"FEM")
    plt.xlabel(r'$\bar{x}$', fontsize=12)
    plt.ylabel(r'$\bar{y}$', fontsize=12)
    plt.colorbar(im1, fraction=0.045)
    
    # 2. XY Reconstruction Plot
    plt.subplot(1, 3, 2)
    im2 = plt.imshow(rec, extent=extent, vmin=-1, vmax=1, origin='lower', cmap='RdBu_r')
    plt.title('XY Reconstruction')
    plt.xlabel(r'$\bar{x}$', fontsize=12)
    plt.colorbar(im2, fraction=0.045)
    
    # 3. Difference Plot
    plt.subplot(1, 3, 3)
    diff = fem - rec
    
    # Calculate the maximum absolute deviation to ensure symmetry
    v_limit = np.max(np.abs(diff))
    
    # Use standard Normalize with symmetric bounds
    # This keeps 0 exactly in the middle of the PiYG_r map (white)
    norm_sym = Normalize(vmin=-v_limit, vmax=v_limit)
        
    im3 = plt.imshow(diff, extent=extent, origin='lower', cmap='PiYG_r', norm=norm_sym)
    plt.title('Difference')
    plt.xlabel(r'$\bar{x}$', fontsize=12)   
    cbar = plt.colorbar(im3, fraction=0.045, format='%.3f') 

    plt.tight_layout()
    plt.savefig(f'figs/em/mode{mn}_varsep.pdf', dpi=300)
    plt.show()

plt.figure(figsize=(7,4))

plt.subplot(121)
plt.plot(x_axis, x_11, "b", label='X11')
plt.plot(x_axis, x_13, color="orange",label='X13')
plt.plot(x_axis, x_31, "g", label='X31')
plt.plot(x_axis, x_33, "r", label='X33')
plt.xlabel(r'$\bar{x}$')
plt.legend()

plt.subplot(122)
plt.plot(x_axis, y_11, "b", linestyle='-', label='Y11')
plt.plot(x_axis, y_13,color="orange", linestyle='-',label='Y13')
plt.plot(x_axis, y_31,  "g",linestyle='-', label='Y31')
plt.plot(x_axis, y_33, "r", linestyle='-', label='Y33')
plt.ylabel(r'$\bar{y}$')
plt.legend()
plt.show()

plt.figure(figsize=(6,9))
plt.subplot(411)
plt.plot(x_axis, x_11, "b", label='X')
plt.plot(x_axis, y_11, color = "orange", linestyle='--', label='Y')
plt.legend()
plt.title('$X_{mn}(\\bar{x})$, $Y_{mn}(\\bar{y})$')
plt.ylabel('Scaled Deflection')
plt.tight_layout()

plt.subplot(412)
plt.plot(x_axis, x_13, "b", label='X')
plt.plot(x_axis, y_13, color = "orange", linestyle='--', label='Y')
plt.legend()
# plt.title('$X_{mn}(\\bar{x})$, $Y_{mn}(\\bar{y})$')
plt.ylabel('Scaled Deflection')
plt.tight_layout()

plt.subplot(413)
plt.plot(x_axis, x_31, "b", label='X')
plt.plot(x_axis, y_31, color = "orange", linestyle='--', label='Y')
plt.legend()
# plt.title('$X_{mn}(\\bar{x})$, $Y_{mn}(\\bar{y})$')
plt.ylabel('Scaled Deflection')
plt.tight_layout()

plt.subplot(414)
plt.plot(x_axis, x_33, "b", label='X')
plt.plot(x_axis, y_33, color = "orange", linestyle='--', label='Y')
plt.legend()
plt.xlabel('$x/(2a)$, $y/(2b)$')
plt.ylabel('Scaled Deflection')
plt.tight_layout()
# plt.savefig(f'var_separation_XY.png', dpi=300)

plt.show()



plt.figure(figsize=(6,9))
plt.subplot(421)
plt.plot(x_axis, x_11, "b", label='X11')
plt.plot(x_axis, x11_fit, color = "orange", linestyle='--', label='X11 fit')
plt.legend()
plt.title('$X_{mn}(x)$')
plt.ylabel('Scaled Deflection')
plt.tight_layout()

plt.subplot(422)
plt.plot(x_axis, y_11, "b", label='Y11')
plt.plot(x_axis, y11_fit, color = "orange", linestyle='--', label='Y11 fit')
plt.legend()
plt.title('$Y_{mn}(y)$')
plt.tight_layout()

plt.subplot(423)
plt.plot(x_axis, x_13, "b", label='X13')
plt.plot(x_axis, x13_fit, color = "orange", linestyle='--', label='X13 fit')
plt.legend()
plt.ylabel('Scaled Deflection')
plt.tight_layout()

plt.subplot(424)
plt.plot(x_axis, y_13, "b", label='Y13')
plt.plot(x_axis, y13_fit, color = "orange", linestyle='--', label='Y13 fit')
plt.legend()
plt.tight_layout()

plt.subplot(425)
plt.plot(x_axis, x_31, "b", label='X31')
plt.plot(x_axis, x31_fit, color = "orange", linestyle='--', label='X31 fit')
plt.legend()
plt.ylabel('Scaled Deflection')
plt.tight_layout()

plt.subplot(426)
plt.plot(x_axis, y_31, "b", label='Y31')
plt.plot(x_axis, y31_fit, color = "orange", linestyle='--', label='Y31 fit')
plt.legend()
plt.ylabel('Scaled Deflection')
plt.tight_layout()

plt.subplot(427)
plt.plot(x_axis, x_33, "b", label='X33')
plt.plot(x_axis, x33_fit, color = "orange", linestyle='--', label='X33 fit')
plt.legend()
plt.xlabel('$x/(2a)$')
plt.ylabel('Scaled Deflection')
plt.tight_layout()

plt.subplot(428)
plt.plot(x_axis, y_33, "b", label='Y33')
plt.plot(x_axis, y33_fit, color = "orange", linestyle='--', label='Y33 fit')
plt.legend()
plt.xlabel('$y/(2b)$')
plt.ylabel('Scaled Deflection')
plt.tight_layout()
# plt.savefig(f'var_separation_XY_fit.png', dpi=300)
plt.show()