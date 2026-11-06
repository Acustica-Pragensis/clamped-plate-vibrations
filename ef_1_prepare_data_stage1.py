"""Stage 1: extract the 9 selected modes (f11..f33) from the COMSOL exports.

Reads  data/comsol_modes.csv                  rank -> (m,n) table per b/a
       data/ef/ef_comsol_train.csv    train (a,h) geometries, all ba
       data/ef/ef_comsol_test.csv     test  (a,h) geometries, all ba
Writes data/ef/ef_selected_train.csv             train (a,h) geometries, selected ba
       data/ef/ef_selected_test.csv              test  (a,h) geometries, selected ba
"""
import re
import numpy as np

MODES_CSV = "data/comsol_modes.csv"
OUT_COLS = ["f11", "f12", "f13", "f21", "f22", "f23", "f31", "f32", "f33"]

SETS = [
    ("train", "data/ef/ef_comsol_train.csv", "data/ef/ef_selected_train.csv"),
    ("test",  "data/ef/ef_comsol_test.csv",  "data/ef/ef_selected_test.csv"),
]

def load_modes(path):
    """b/a -> list of mode labels in COMSOL rank order."""
    table = {}
    with open(path) as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    for ln in lines[1:]:                                   # skip header
        parts = [p.strip() for p in ln.split(",")]
        if parts[0]:
            table[round(float(parts[0]), 6)] = [p for p in parts[1:] if p]
    return table


def load_comsol(path):
    """Return ba, a, h and the f1..fN matrix; f-columns are detected."""
    data = np.genfromtxt(path, delimiter=",", names=True)
    names = [n.strip().lstrip("\ufeff").lower() for n in data.dtype.names]

    fnums = sorted(int(m.group(1)) for n in names
                   for m in [re.fullmatch(r"f(\d+)", n)] if m)
    if not fnums:
        raise SystemExit(f"{path}: no f* columns found")

    ba = np.atleast_1d(data["ba"]).astype(float)
    a  = np.atleast_1d(data["a"]).astype(float)
    h  = np.atleast_1d(data["h"]).astype(float)
    F  = np.column_stack([np.atleast_1d(data[f"f{i}"]).astype(float)
                          for i in fnums])
    return ba, a, h, F


def extract(name, path, out_csv, modes):
    ba, a, h, F = load_comsol(path)
    kb = np.round(ba, 6)

    unknown = sorted(set(kb) - set(modes))
    if unknown:
        raise SystemExit(f"[{name}] b/a missing from {MODES_CSV}: "
                         f"{', '.join(f'{v:g}' for v in unknown)}")

    print(f"[{name}] {len(ba):4d} rows loaded   f1..f{F.shape[1]}")
    print(f"[{name}] ba available in file: "
          f"{', '.join(f'{v:g}' for v in np.unique(kb))}")

    key = np.column_stack([kb, np.round(a, 12), np.round(h, 12)])
    if len(np.unique(key, axis=0)) != len(key):
        print(f"[{name}] WARNING: duplicated (ba, a, h) rows present")

    order = np.lexsort((np.round(h, 12), np.round(a, 12), np.round(ba, 6)))
    ba, a, h, F = ba[order], a[order], h[order], F[order]

    # rank -> label rename
    n = len(ba)
    out = np.full((n, len(OUT_COLS)), np.nan)
    for i in range(n):
        labels = modes[round(ba[i], 6)][:F.shape[1]]
        for j, lab in enumerate(labels):
            if lab in OUT_COLS:
                out[i, OUT_COLS.index(lab)] = F[i, j]

    nan_cols = np.isnan(out).sum(axis=0)
    if nan_cols.any():
        detail = ", ".join(f"{c}:{k}" for c, k in
                           zip(OUT_COLS, nan_cols) if k)
        print(f"[{name}] missing modes: {detail}")

    rows = np.column_stack([ba, a, h, out])
    header = "ba,a,h," + ",".join(OUT_COLS)
    np.savetxt(out_csv, rows, delimiter=",", header=header,
               comments="", fmt="%.10g")
    print(f"[{name}] wrote {out_csv}: {n} geometries, "
          f"ba {ba.min():g}..{ba.max():g}\n")


def main():
    modes = load_modes(MODES_CSV)
    for name, path, out_csv in SETS:
        extract(name, path, out_csv, modes)


if __name__ == "__main__":
    main()