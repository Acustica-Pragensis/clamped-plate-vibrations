#!/usr/bin/env python3
"""Test the present eigenfrequency approximation against the material sweep.

The reference data are the COMSOL eigenfrequencies in
data/ef/ef_materials.csv, computed by ef_other_materials_comsol.m for

    a = b = 900 um,  h = 10 um

and four materials.  The mode order is taken from the b/a = 1.03 row of
data/comsol_modes.csv, so the k-th frequency column f{k} is labelled by the
k-th mode in that row.
"""

import csv
import math
import re

# ------------------------- configuration -----------------------------
MATERIALS_CSV = "data/ef/ef_materials.csv"
MODES_CSV     = "data/comsol_modes.csv"
MODE_ROW_BA   = 1.03
MATERIALS     = ["silicon", "aluminium", "iron", "copper"]

# Geometry used in ef_other_materials_comsol.m
A = 900e-6          # m
BA = 1.0            # b/a
B = BA * A          # m
H = 10e-6           # m

# ------------------------- present approximation ----------------------
def f_present(a, b, h, m, n, E, nu, rho):
    """Present closed-form approximation, as used in ef_6_test.py."""
    return (
        math.pi / 8 * math.sqrt(E * h**2 / (12 * (1 - nu**2) * rho))
        * (((m + ((m - 0.224) / (a * n / b + 1.923 * m - 0.717))) / a)**2
           + ((n + ((n - 0.224) / (1.923 * n - 0.717 + b * m / a))) / b)**2)
    )


def pct_err(pred, ref):
    return abs((pred - ref) / ref) * 100.0


# ------------------------- input --------------------------------------
def read_mode_labels(path, ba):
    """Mode labels from the row of comsol_modes.csv whose b/a equals ba."""
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))

    for row in rows[1:]:
        if not row:
            continue
        try:
            value = float(row[0])
        except ValueError:
            continue
        if abs(value - ba) < 1e-9:
            return [
                label.strip()
                for label in row[1:]
                if label.strip() and label.strip().lower() != "nan"
            ]

    raise ValueError(f"no row with b/a = {ba} in {path}")


def parse_mode(label):
    """f12 -> (1, 2)."""
    match = re.fullmatch(r"f(\d)(\d)", label)
    if match is None:
        raise ValueError(f"cannot parse mode label: {label!r}")
    return int(match.group(1)), int(match.group(2))


def read_materials(path):
    with open(path, newline="") as fh:
        return {
            row["material"].strip().lower(): row
            for row in csv.DictReader(fh)
        }


# ------------------------- comparison ---------------------------------
labels = read_mode_labels(MODES_CSV, MODE_ROW_BA)
modes = [parse_mode(label) for label in labels]
materials = read_materials(MATERIALS_CSV)

# The number of frequency columns in ef_materials.csv.
n_material_modes = sum(
    1 for key in materials[MATERIALS[0]] if re.fullmatch(r"f\d+", key)
)

results = []
for name in MATERIALS:
    if name not in materials:
        raise ValueError(f"material {name!r} is missing from {MATERIALS_CSV}")

    row = materials[name]
    E = float(row["E_GPa"]) * 1e9
    nu = float(row["nu"])
    rho = float(row["rho_kgm3"])

    f_num = []
    f_pred = []
    err = []

    for k, (m, n) in enumerate(modes, start=1):
        ref = float(row[f"f{k}"])
        pred = f_present(A, B, H, m, n, E, nu, rho)

        f_num.append(ref)
        f_pred.append(pred)
        err.append(pct_err(pred, ref))

    results.append((name, f_num, f_pred, err))

# ------------------------- table --------------------------------------
COL_W = 13
ROW_W = 16

print()
print(f"Geometry: a = b = {A * 1e6:.0f} um, h = {H * 1e6:.0f} um")
print(f"Mode order: b/a = {MODE_ROW_BA:g} row of {MODES_CSV}")
if len(labels) < n_material_modes:
    print(
        f"Note: {MATERIALS_CSV} has {n_material_modes} frequency columns, "
        f"but only {len(labels)} modes are labelled for b/a = {MODE_ROW_BA:g}."
    )
print()

header = f"{'material':<{ROW_W}}" + "".join(
    f"{f'({m},{n})':>{COL_W}}" for m, n in modes
)
print(header)
print("-" * len(header))

for name, f_num, f_pred, err in results:
    print(f"{name}")
    print(
        f"{'  f_num [Hz]':<{ROW_W}}"
        + "".join(f"{value:>{COL_W}.2f}" for value in f_num)
    )
    print(
        f"{'  f_pred [Hz]':<{ROW_W}}"
        + "".join(f"{value:>{COL_W}.2f}" for value in f_pred)
    )
    print(
        f"{'  Err_f [%]':<{ROW_W}}"
        + "".join(f"{value:>{COL_W}.2f}" for value in err)
    )

# ---------------------------------------------------------------------------
# LaTeX version of the table -- only the modes used in the manuscript
# ---------------------------------------------------------------------------
LATEX_MODES = [(1, 1), (1, 2), (2, 2), (1, 3), (2, 3), (3, 3)]
COL_W       = 11        # width of a numeric cell in the .tex source (cosmetic)
THIN        = r'\,'     # thousands separator

def fmt_num(x, nd=2):
    return f"{x:,.{nd}f}".replace(",", THIN)

sel  = [(m, n) for m, n in LATEX_MODES if (m, n) in modes]
idx  = {mn: modes.index(mn) for mn in sel}
ncol = len(sel)

missing = [mn for mn in LATEX_MODES if mn not in modes]
if missing:
    print(f"% WARNING: {missing} not present in the mode list, skipped")

cells = lambda values: " & ".join(f"{fmt_num(v):>{COL_W}}" for v in values)

print()
print(r"\begin{tabular}{|l|" + "r|" * ncol + r"} \hline")
print(r"mode ($m,n$)" + " " * 10 + "& "
      + " & ".join(f"({m},{n})".rjust(COL_W) for m, n in sel) + r" \\ \hline")

for name, f_num, f_pred, err in results:

    print(f"& \\multicolumn{{{ncol}}}{{c|}}{{{name}}} \\\\ \\hline")

    num = [f_num[idx[(m, n)]] for m, n in sel]
    prd = [f_pred[idx[(m, n)]] for m, n in sel]
    e   = [err[idx[(m, n)]]   for m, n in sel]

    print(r"$f_{\mathrm{num}}$ (Hz)     & " + cells(num) + r" \\")
    print(r"$f$ (Hz)                    & " + cells(prd) + r" \\")
    print(r"$\mathrm{Err}_f$ (\%)       & " + cells(e)   + r" \\ \hline")

print(r"\end{tabular}")