# load data with 9 selected modes (f11..f33) for each training sample (a,ba,h) and reshape into one row per (sample, mode)
import numpy as np
import matplotlib.pyplot as plt

csv_file = "data/ef/ef_selected_train.csv"
save_file= "data/ef/ef_pysr.csv"

data = np.loadtxt(csv_file, delimiter=",", skiprows=1)

n_freq = 9  # f11..f33

# --- reshape into one row per (sample, mode) ---
f  = data[:, 3:3 + n_freq].ravel()           # frequency
a  = np.repeat(data[:, 1], n_freq)
b  = np.repeat(data[:, 0], n_freq)*a
h  = np.repeat(data[:, 2], n_freq)
m  = np.tile(np.repeat([1, 2, 3], 3), len(data))
n  = np.tile([1, 2, 3],      3 * len(data))

DATA = np.column_stack([f, a, b, h, m, n])

np.savetxt(save_file, DATA,
           fmt=["%.1f", "%.1e", "%.3e", "%.1e", "%d", "%d"],
           header="f,a,b,h,m,n", comments="")
print(DATA.shape)