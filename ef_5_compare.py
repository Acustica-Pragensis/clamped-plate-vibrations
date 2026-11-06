import importlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import hof_equations as hof_models

# ------------------------- switches ---------------------------------
OUT_CSV   = "data/ef/compare_results_vs_FEM.csv"
SAVE_FIGS = False
SUBSET    = None       # None = all; or e.g. ["f_hof_cx11", "f_hof_cx12"]

# ------------------------- material / plate -------------------------
E     = 170e9       # Pa
nu    = 0.28        # -
rho_p = 2329.0      # kg/m^3

D_plate = lambda h: E * h**3 / (12.0 * (1.0 - nu**2))

# ------------------------- data -------------------------------------
data = np.loadtxt("data/ef/ef_selected_train.csv", delimiter=",", skiprows=1)
ba = data[:, 0]
a  = data[:, 1]
b  = ba * a
h  = data[:, 2]

# (mode m,n) -> column index in the FEM data ef_selected_train.csv
FEM_COLS = {(1,1):3, (1,2):4, (1,3):5,
            (2,1):6, (2,2):7, (2,3):8,
            (3,1):9, (3,2):10, (3,3):11}
MODES = sorted(FEM_COLS)

ok = np.isfinite(a) & np.isfinite(ba) & np.isfinite(h) & (ba > 0) & (a > 0) & (h > 0)

# --- our old model --------------------------------------------------
f_rect = lambda a, b, h, m, n, E, nu, rho: (
    0.201 / a**2
    * (1 + 1.394 * (a / b - 1) / ((m / n)**2 + 0.825))
    * np.sqrt((m + n)**5.60 / (m * n)**1.16)
    * np.sqrt(E * h**2 / (12.0 * (1.0 - nu**2) * rho))
)

# --- Mitchell & Hazell, eqs (1), (6), (7) ---------------------------
delta_m = lambda a, b, m, n: ((n * a / (m * b))**2 + 2.0)**-1 + 0.017 / m
delta_n = lambda a, b, m, n: ((m * b / (n * a))**2 + 2.0)**-1 + 0.017 / n

f_mn_MH = lambda a, b, h, m, n, E, nu, rho: (
    np.pi / (2.0 * a**2) * np.sqrt(E * h**2 / (12.0 * (1.0 - nu**2) * rho))
    * ((m + delta_m(a, b, m, n))**2
       + (n + delta_n(a, b, m, n))**2 * (a / b)**2)
    / 4
)

pct_err = lambda pred, ref: np.abs((pred - ref) / ref) * 100.0

# ------------------------- comparison -------------------------------
rows      = []
per_ba    = {}
per_ba_mh = {}

hdr = f"{'mode':>6} " \
      f"{'our FEM%':>9} {'MH-FEM%':>9}"
print(hdr); print("-" * len(hdr))

for (m, n) in MODES:
    mf, nf = float(m), float(n)

    f_fem = data[:, FEM_COLS[(m, n)]]
    good  = ok & np.isfinite(f_fem)

    f_usr = f_rect(a, b, h, mf, nf, E, nu, rho_p)
    f_mh  = f_mn_MH(a, b, h, mf, nf, E, nu, rho_p)

    d_m = delta_m(a, b, mf, nf)
    d_n = delta_n(a, b, mf, nf)

    e_uf = pct_err(f_usr, f_fem)
    e_mf = pct_err(f_mh,  f_fem)

    for i in np.flatnonzero(good):
        rows.append([m, n, a[i], h[i], ba[i], f_fem[i], f_usr[i], f_mh[i], e_uf[i], e_mf[i], d_m[i], d_n[i]])
        per_ba.setdefault(round(float(ba[i]), 6), []).append(e_uf[i])
        per_ba_mh.setdefault(round(float(ba[i]), 6), []).append(e_mf[i])

    print(f"({m},{n:>1})  {e_uf[good].mean():9.2f} {e_mf[good].mean():9.2f}")

rows = np.array(rows)

# ------------------------- save (formatted) -------------------------
HEADER = "m,n,a,h,ba,f_FEM_Hz,f_rect_Hz,f_MH_Hz,err_rect_FEM_pct,err_MH_FEM_pct,delta_m,delta_n"

def fmt_ba(x, nd=2):
    """1.10 -> '1.1' | 1.00 -> '1.' | 1.125 -> '1.13'  (max nd decimals)"""
    return f"{x:.{nd}f}".rstrip("0")

table = [[f"{r[0]:#.0f}",      # m    -> "1."
          f"{r[1]:#.0f}",      # n    -> "1."
          f"{r[2]:.6g}",       # a    -> "1"  / "0.5"
          f"{r[3]:.6g}",       # h    -> "0.01"
          fmt_ba(r[4]),        # ba   -> "1.1"
          f"{r[5]:.0f}",       # f_FEM_Hz
          f"{r[6]:.0f}",       # f_rect_Hz
          f"{r[7]:.0f}",       # f_MH_Hz
          f"{r[8]:.6f}",       # err_rect_FEM_pct
          f"{r[9]:.6f}",
          f"{r[10]:.6f}",      # delta_m
          f"{r[11]:.6f}"       # delta_n
          ]       # err_MH_FEM_pct
         for r in rows]

np.savetxt(OUT_CSV, np.array(table, dtype=object), delimiter=",",
           header=HEADER, comments="",
           fmt="%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s")

# # ------------------------- summary ----------------------------------
# print("\nmean |error| vs FEM [%] per mode")
# print(f"{'mode':>6} {'ours':>10} {'MH':>10}")
# for (m, n) in MODES:
#     s = rows[(rows[:, 0] == m) & (rows[:, 1] == n)]
#     print(f"{m},{n:>1} {s[:,8].mean():10.3f} {s[:,9].mean():10.3f}")

# e = rows[:, 8]
# print(f"\noverall f_rect vs FEM : mean {e.mean():.3f} % | median {np.median(e):.3f} % | max {e.max():.3f} %")
# e = rows[:, 9]
# print(f"overall MH     vs FEM : mean {e.mean():.3f} % | median {np.median(e):.3f} % | max {e.max():.3f} %")


def buckets_by_ba(per_ba):
    keys = sorted(per_ba)
    return [f"{k:.2f}" for k in keys], [per_ba[k] for k in keys]

# ------------------------- Hall-of-Fame models ----------------------
importlib.reload(hof_models)

fem       = {md: data[:, FEM_COLS[md]] for md in MODES}
unique_ba = np.unique(ba)

def err_by_ba(pred):
    """pred(md, mask) -> predictions; returns (labels, buckets, flat) over all modes."""
    labels, buckets, flat = [], [], []
    for bv in unique_ba:
        mask = (ba == bv)
        e = []
        for md in MODES:
            p = pred(md, mask)
            e.extend(np.abs((p - fem[md][mask]) / fem[md][mask]) * 100)
        labels.append(f"{bv:.2f}")
        buckets.append(np.array(e))
        flat.extend(e)
    return labels, buckets, np.array(flat)

# --- compute MH once, before the HoF loop ---
mh_labels, mh_buckets, mh_flat = err_by_ba(
    lambda md, mask: f_mn_MH(a[mask], b[mask], h[mask],
                             float(md[0]), float(md[1]), E, nu, rho_p))
mh_med = np.array([np.median(bk) for bk in mh_buckets])

# --- add an optional reference to the helper ---
def boxplot_by_ba(name, labels, buckets, fname=None, title=None, ref=mh_med):
    plt.figure(figsize=(6, 4))
    plt.boxplot(buckets, tick_labels=labels, patch_artist=True,
                showfliers=False,
                boxprops=dict(facecolor='lightgrey', color='black', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    if ref is not None:                      # MH medians, aligned with ba
        plt.plot(np.arange(1, len(ref) + 1), ref, "D--", color="tab:blue",
                 ms=4, lw=1, label="Mitchell & Hazell (median)")
        plt.legend(fontsize=8)
    plt.xlabel("aspect ratio  b/a")
    plt.ylabel(r"$\text{Err}_f / \%$")
    plt.title(title or f"{name} vs FEM, all modes")
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    if SAVE_FIGS and fname:
        plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()

def boxplot_overlay(labels, buckets_ref, buckets_ours, fname,
                    ref_name="Mitchell & Hazell", ours_name="f_rect",
                    title=None, offset=0.0):
    """Two overlaid boxplots per b/a bucket: reference model + our model."""
    pos  = np.arange(len(labels), dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))

    common = dict(widths=0.55, patch_artist=True, showfliers=False,
                  manage_ticks=False)

    # reference first -> the transparent one is drawn on top of it
    ax.boxplot(buckets_ref, positions=pos - offset, **common,
               boxprops   =dict(facecolor="tab:blue",   alpha=0.35, edgecolor="tab:blue"),
               medianprops=dict(color="tab:blue",   linewidth=2.0),
               whiskerprops=dict(color="tab:blue"),
               capprops    =dict(color="tab:blue"))

    ax.boxplot(buckets_ours, positions=pos + offset, **common,
               boxprops   =dict(facecolor="tab:orange", alpha=0.35, edgecolor="tab:orange"),
               medianprops=dict(color="tab:orange", linewidth=2.0),
               whiskerprops=dict(color="tab:orange"),
               capprops    =dict(color="tab:orange"))

    ax.set_xticks(pos, labels)
    ax.set_xlabel("aspect ratio  b/a")
    ax.set_ylabel(r"$\mathrm{Err}_f\,/\,\%$")
    ax.set_title(title or f"{ours_name} vs {ref_name} vs FEM")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(handles=[
        Patch(facecolor="tab:blue",   alpha=0.35, edgecolor="tab:blue",   label=ref_name),
        Patch(facecolor="tab:orange", alpha=0.35, edgecolor="tab:orange", label=ours_name)],
        loc="upper left")
    fig.tight_layout()
    if SAVE_FIGS:
        fig.savefig(fname, dpi=200)
    plt.show()

lbl_mh, buck_mh, flat_mh = err_by_ba(
    lambda md, mask: f_mn_MH(a[mask], b[mask], h[mask],
                             float(md[0]), float(md[1]), E, nu, rho_p))
lbl_us, buck_us, flat_us = err_by_ba(
    lambda md, mask: f_rect(a[mask], b[mask], h[mask],
                            float(md[0]), float(md[1]), E, nu, rho_p))

print(f"{'f_rect':>16}  mean {flat_us.mean():6.2f} %  median {np.median(flat_us):6.2f} %  max {flat_us.max():6.2f} %")
print(f"{'f_MH':>16}  mean {flat_mh.mean():6.2f} %  median {np.median(flat_mh):6.2f} %  max {flat_mh.max():6.2f} %")

boxplot_overlay(lbl_mh, buck_mh, buck_us,
                "figs/ef_ours_vs_mh_vs_fem.png", ours_name="f_rect")

# every HoF model: the b slot gets the b/a RATIO
for name in (list(hof_models.HOF_MODELS) if SUBSET is None else SUBSET):
    f = hof_models.HOF_MODELS[name]
    labels, buckets, flat = err_by_ba(
        lambda md, mask, f=f: f(a[mask], b[mask], h[mask],
                                float(md[0]), float(md[1]), E, nu, rho_p))
    # boxplot_by_ba(name, labels, buckets, f"figs/hof_{name}.png")
    print(f"{name:>16}  mean {flat.mean():6.2f} %  median {np.median(flat):6.2f} %  max {flat.max():6.2f} %")

    boxplot_overlay(lbl_mh, buck_mh, buckets,
                f"figs/boxplot_{name}.png", ours_name=name)