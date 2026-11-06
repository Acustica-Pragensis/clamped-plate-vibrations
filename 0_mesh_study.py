#!/usr/bin/env python3
"""
Mesh convergence study of COMSOL eigenfrequencies.

Metric: relative mesh error per mode w.r.t. the finest mesh (reference),
summarised by the worst mode, plus the successive change between
neighbouring meshes and an estimate of the observed convergence order.
"""

import re
import numpy as np
import matplotlib.pyplot as plt

version = "v3"
CSV_PATH = f"data/mesh_study_{version}.csv"

# load ------------------------------------------------------------------
data = np.genfromtxt(CSV_PATH, delimiter=",", names=True)

# frequency columns: any column named f<integer>, ordered by mode number
freq_cols = sorted((c for c in data.dtype.names if re.fullmatch(r"f\d+", c)),
                   key=lambda c: int(c[1:]))
mode_numbers = np.array([int(c[1:]) for c in freq_cols])

# coarsest -> finest
order = np.argsort(data["elems_per_side"])
elems = np.asarray(data["elems_per_side"], dtype=int)[order]
freqs = np.column_stack([data[c] for c in freq_cols])[order]   # shape (n_meshes, n_modes)

# plot ------------------------------------------------------------------
plt.figure(figsize=(10, 6))
for n, f in zip(elems, freqs):
    plt.plot(mode_numbers, f, "o-", ms=4, label=f"# = {n}")
plt.xlabel("mode number")
plt.ylabel("eigenfrequency / Hz")
plt.xticks(mode_numbers)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.yscale('log')
plt.savefig(f"figs/mesh_convergence_{version}.png", dpi=150)
plt.show()

# difference to the finest mesh --------------------------------------------
f_ref = freqs[-1]                                  # finest mesh = reference
rel = (freqs - f_ref) / f_ref * 100.0              # signed, / %

plt.figure()
for n, e in zip(elems, rel):
    plt.plot(mode_numbers, e, "o-", ms=4, label=f"# = {n}")
plt.axhline(0.0, color="k", lw=0.8)
plt.xlabel("mode number")
plt.ylabel("frequency difference to the finest mesh / %")
plt.xticks(mode_numbers)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f"figs/mesh_convergence_diff_{version}.png", dpi=150)
plt.show()

# difference to the finest mesh vs refinement -------------------------------
plt.figure()
for i, j in enumerate(mode_numbers):
    plt.loglog(elems, rel[:, i], "o-", ms=4, label=f"mode {j}")
plt.xlabel("#")
plt.ylabel("frequency difference to the finest mesh / %")
plt.grid(alpha=0.3, which="both")
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.savefig(f"figs/mesh_convergence_vs_elems_{version}.png", dpi=150)
plt.show()

# metric 1: vs reference ---------------------------------------------------
f_ref = freqs[-1]                                    # finest mesh
rel_to_ref = np.abs(freqs - f_ref) / f_ref * 100.0   # / %, (n_meshes, n_modes)

print("Eigenfrequency error relative to the finest mesh / %")
print(f"reference: elems_per_side = {elems[-1]}\n")

head = f"{'elems/side':>10} | " + " ".join(f"{m:>9d}" for m in mode_numbers)
print(head)
print("-" * len(head))
for i, n in enumerate(elems):
    print(f"{n:>10d} | " + " ".join(f"{e:>9.3f}" for e in rel_to_ref[i]))

print()
for i, n in enumerate(elems):
    j = int(np.argmax(rel_to_ref[i]))
    print(f"elems_per_side={n:>4d}: max = {rel_to_ref[i].max():8.4f} % "
          f"(mode {mode_numbers[j]}), mean = {rel_to_ref[i].mean():8.4f} %")

# metric 2: successive change ---------------------------------------------
succ = np.abs(np.diff(freqs, axis=0)) / freqs[:-1] * 100.0   # / %

print("\nSuccessive change between neighbouring meshes / %")
for i in range(len(elems) - 1):
    j = int(np.argmax(succ[i]))
    print(f"{elems[i]:>4d} -> {elems[i + 1]:>4d}: max = {succ[i].max():8.4f} % "
          f"(mode {mode_numbers[j]}), mean = {succ[i].mean():8.4f} %")

# observed order of convergence -------------------------------------------
if len(elems) >= 3:
    e_prev = np.abs(freqs[-2] - freqs[-3])           # last two refinements
    e_last = np.abs(freqs[-1] - freqs[-2])
    r = elems[-1] / elems[-2]                        # refinement ratio (h ~ 1/N)
    mask = (e_prev > 0) & (e_last > 0) & (r != 1)
    if mask.any():
        p = np.log(e_prev[mask] / e_last[mask]) / np.log(r)
        print(f"\nObserved convergence order (finest three meshes): "
              f"median p = {np.median(p):.2f}  (range {p.min():.2f} .. {p.max():.2f})")