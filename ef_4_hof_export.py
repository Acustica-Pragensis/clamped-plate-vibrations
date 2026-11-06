#!/usr/bin/env python3
"""Export the PySR Hall of Fame to Python and LaTeX from a single parse.

    hall_of_fame.csv  ->  hof_equations.py         (full model)
                          hof_equations_delta.py   (slot only)
                          hof_equations.tex        (compilable with pdflatex)

Every slot is canonicalised in the ratio x = a/b, never in a and b
separately. That gives two things at once:

  * duplicate entries collapse (f5 and f7 are the same function of x),
  * a/b can never be split into b / (a + ...), so the printed equation
    still states the physics claim: the mode shape depends on the ratio
    only, not on a and b individually.

Duplicates are found at three levels of tolerance; the entry with the
lowest complexity always wins:

  1. identical canonical expression
  2. identical after blanking every fitted constant   ("same form")
  3. identical numerical fingerprint                  (same values, folded
                                                       differently by sympy)
"""

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sympy as sp


# ------------------------------------------------------------- settings
HOF_CSV       = "outputs/20261021_115947_0ebgKa/hall_of_fame.csv"
OUT_PY        = "hof_equations.py"
OUT_DELTA_PY  = "hof_equations_delta.py"
OUT_TEX       = "hof_equations.tex"
OUT_TEX_DELTA = "hof_equations_delta.tex"

MIN_COMPLEXITY  = 3      # complexity 1 and 2 are useless
SIG_DIGITS_PY   = 4      # digits in hof_equations*.py
SIG_DIGITS_TEX  = 5      # digits in hof_equations.tex

DEDUPLICATE     = True
BLIND_CONSTANTS = True   # also drop entries that differ only by their numbers
FINGERPRINT_SIG = 10     # digits used by the numerical duplicate check

# Python name of one model = FUNC_PREFIX + tag, e.g. "f_" + "cx4" = "f_cx4".
# Use "f_hof_" for the older naming, or "" for bare "cx4".
FUNC_PREFIX = "f_"

# (pi/8) * h * sqrt(E / (12 (1-nu^2) rho))
PREFACTOR_PY  = "np.pi/8 * np.sqrt(E*h**2/(12*(1-nu**2)*rho))"
PREFACTOR_TEX = r"\frac{\pi h}{8}\sqrt{\frac{E}{12(1-\nu^{2})\rho}}"


# -------------------------------------------------------------- symbols
# x = a/b is the slot argument. positive=True so that simplify() may cancel
# sqrt(x**2), Abs(...) etc. -- lengths and mode numbers are positive here.
X, M, N = sp.symbols("x m n", positive=True)
A, B, H = sp.symbols("a b h", positive=True)

CONST = sp.Symbol("const")          # placeholder for one fitted number

SLOT_ARGS  = ["x", "m", "n"]                    # slot signature f(x, m, n)
SLOT_NAMES = {"x": "a/b", "m": "m", "n": "n"}   # for the comment lines only

SLOT_FUNCTIONS = {
    "square": lambda t: t**2,
    "cube":   lambda t: t**3,
    "inv":    lambda t: 1 / t,
    "sqrt":   sp.sqrt,
    "log":    sp.log,
    "exp":    sp.exp,
    "abs":    sp.Abs,
}


# ------------------------------------------------------------- parsing
def read_hall_of_fame(path):
    """CSV with columns: complexity, loss, equation (quoted)."""
    try:
        return np.loadtxt(path, delimiter=",", dtype=str,
                          skiprows=1, quotechar='"')
    except TypeError:                       # older numpy: read it by hand
        with open(path) as handle:
            next(handle)
            rows = [line.rstrip("\n").split(",", 2)
                    for line in handle if line.strip()]
        return np.atleast_2d(np.array(rows, dtype=str))


def parse_slot(rhs):
    """'inv(#1 + 1.8223)' -> sympy expression in (x, m, n)."""
    with_names = re.sub(r"#(\d+)",
                        lambda hit: SLOT_ARGS[int(hit.group(1)) - 1], rhs)
    local = {name: sp.Symbol(name, positive=True) for name in SLOT_ARGS}
    local.update(SLOT_FUNCTIONS)
    return sp.sympify(with_names.replace("^", "**"), locals=local)


def canonical_form(expr):
    """Normal form, always in the ratio variable x.

    cancel/together first (cheap and keeps the fraction structure tidy),
    then simplify (this is the step that merges f5 and f7), then cancel
    again because simplify likes to re-nest fractions.
    """
    expr = sp.cancel(sp.together(expr))
    expr = sp.simplify(expr)
    return sp.cancel(sp.together(expr))


def parse_loss(text):
    """Loss column -> float, nan if it is not a number."""
    try:
        return float(text)
    except ValueError:
        return float("nan")


# --------------------------------------------------------- expression ops
def round_constants(expr, digits):
    """Round every fitted constant to `digits` significant digits."""
    replacements = {value: sp.Float(value, digits)
                    for value in expr.atoms(sp.Float) if value.is_finite}
    return expr.xreplace(replacements)


def with_ratio(expr):
    """x -> a/b. Used only when printing, never before canonicalising."""
    return expr.xreplace({X: A / B})


def mirrored(expr):
    """The second slot occurrence: x -> 1/x and m <-> n."""
    return expr.xreplace({X: 1 / X, M: N, N: M})


def blank_constants(expr):
    """Structure key: every fitted number replaced by one placeholder.

    Exponents and integer factors are left alone on purpose -- (m+n)**2 and
    (m+n)**3 are different models, while 0.2819*x and 0.5170*x are not.
    """
    floats = {value: CONST for value in expr.atoms(sp.Float)
              if value.is_finite}
    return expr.xreplace(floats)


FINGERPRINT_X  = (0.6180339887, 0.7548776662, 1.0,
                  1.3247179572, 1.6180339887, 2.0581710272)
FINGERPRINT_MN = ((1, 1), (1, 2), (2, 1), (1, 3),
                  (3, 1), (2, 3), (3, 2), (3, 3))


def fingerprint(expr):
    """Numerical signature over a grid of generic (x, m, n) points.

    Catches expressions that are mathematically equal but that sympy folded
    into different shapes, so the structural keys disagree.
    """
    try:
        function = sp.lambdify((X, M, N), expr, "numpy")
    except Exception:
        return None

    values, n_finite = [], 0
    for x_value in FINGERPRINT_X:
        for m_value, n_value in FINGERPRINT_MN:
            try:
                value = float(function(x_value, float(m_value), float(n_value)))
            except Exception:
                return None
            n_finite += np.isfinite(value)
            values.append(f"{value:.{FINGERPRINT_SIG}g}")
    # a slot that is mostly nan/inf is not a usable signature
    return tuple(values) if n_finite >= 4 else None


def to_python(expr):
    try:
        from sympy.printing.numpy import NumPyPrinter
        return NumPyPrinter().doprint(expr).replace("numpy.", "np.")
    except Exception:
        return sp.printing.pycode(expr).replace("math.", "np.")


def to_latex(expr):
    return r"\left(" + sp.latex(expr) + r"\right)"


def describe_slot(rhs):
    """'#1 + #2' -> '(a/b) + m', for the comment lines only."""
    def replace(hit):
        name = SLOT_NAMES[SLOT_ARGS[int(hit.group(1)) - 1]]
        return f"({name})" if "/" in name else name
    return re.sub(r"#(\d+)", replace, rhs)


# --------------------------------------------------------------- models
@dataclass
class Model:
    complexity: int
    loss: float
    raw_slot: str
    canonical: sp.Expr
    form_key: str                 # canonical with constants blanked
    value_key: Optional[tuple]    # numerical fingerprint
    ops_raw: int
    ops_canonical: int
    tag: str = ""                 # "cx4"
    name: str = ""                # "f_cx4"


class DuplicateIndex:
    """Remembers the models kept so far and explains why a new one is a copy."""

    def __init__(self):
        self._expressions = {}
        self._forms = {}
        self._values = {}

    def duplicate_of(self, candidate):
        """Short explanation if `candidate` duplicates a kept model, else None."""
        expression_key = sp.srepr(candidate.canonical)
        if expression_key in self._expressions:
            return f"identical expression (cx {self._expressions[expression_key]})"

        if BLIND_CONSTANTS and candidate.form_key in self._forms:
            return (f"same form, only constants differ "
                    f"(cx {self._forms[candidate.form_key]})")

        if candidate.value_key is not None and candidate.value_key in self._values:
            return (f"same numbers, folded differently "
                    f"(cx {self._values[candidate.value_key]})")

        return None

    def keep(self, model):
        self._expressions[sp.srepr(model.canonical)] = model.complexity
        self._forms[model.form_key] = model.complexity
        if model.value_key is not None:
            self._values[model.value_key] = model.complexity


def collect_models(rows):
    """Rows -> list of distinct Models, plus counters of what was skipped."""
    index = DuplicateIndex()
    models = []
    per_complexity = {}
    n_low = n_bad = n_dup = 0

    for row in rows:
        complexity = int(float(str(row[0]).strip().lstrip("\ufeff")))
        loss_text  = str(row[1]).strip()
        equation   = str(row[2]).strip().strip('"').strip()

        if complexity < MIN_COMPLEXITY:
            n_low += 1
            continue

        raw_slot = equation.partition("=")[2].strip().strip('"').strip()

        used = [int(index_text) for index_text in re.findall(r"#(\d+)", raw_slot)]
        if used and max(used) > len(SLOT_ARGS):
            n_bad += 1
            print(f"[skip] cx {complexity}: needs #{max(used)}, "
                  f"slot has {len(SLOT_ARGS)} arguments")
            continue

        try:
            slot = parse_slot(raw_slot)
            canonical = canonical_form(slot)
        except Exception as error:
            n_bad += 1
            print(f"[skip] cx {complexity}: {type(error).__name__}: {error}")
            continue

        model = Model(
            complexity=complexity,
            loss=parse_loss(loss_text),
            raw_slot=raw_slot,
            canonical=canonical,
            form_key=sp.srepr(blank_constants(canonical)),
            value_key=fingerprint(canonical),
            ops_raw=sp.count_ops(slot),
            ops_canonical=sp.count_ops(canonical),
        )

        reason = index.duplicate_of(model)
        if DEDUPLICATE and reason is not None:
            n_dup += 1
            print(f"[dup ] cx {complexity}: {reason}")
            continue

        per_complexity[complexity] = per_complexity.get(complexity, 0) + 1
        occurrence = per_complexity[complexity]
        suffix = "" if occurrence == 1 else chr(ord("a") + occurrence - 1)
        model.tag  = f"cx{complexity}{suffix}"
        model.name = FUNC_PREFIX + model.tag

        index.keep(model)
        models.append(model)
        print(f"[keep] cx {complexity}: ops {model.ops_raw} -> "
              f"{model.ops_canonical}   {sp.sstr(canonical)[:58]}")

    return models, (n_low, n_bad, n_dup)


def summary_line(model):
    """'complexity 4, loss 0.0132865' -- one shared format for py and tex."""
    if np.isnan(model.loss):
        return f"complexity {model.complexity}"
    return f"complexity {model.complexity}, loss {model.loss:.7g}"


# -------------------------------------------------------- python writers
PY_DOCSTRING = f'''"""Auto-generated from {HOF_CSV} -- do not edit by hand.

The distinct Hall-of-Fame entries with complexity >= {MIN_COMPLEXITY}, as a
drop-in replacement for f_rect (same signature and units):

    f(a, b, h, m, n, E, nu, rho)

CALL CONVENTION (matters!): b is the TRUE side length, same unit as a. The
slot is fed the ratio a/b internally, so you pass the width, not the ratio.

Constants rounded to {SIG_DIGITS_PY} significant digits.
E in Pa, rho in kg/m^3, frequencies in the unit the search was fitted in.
"""
import numpy as np
'''

DELTA_DOCSTRING = f'''"""Auto-generated from {HOF_CSV} -- do not edit by hand.

The same models as hof_equations.py, but each function returns ONLY the slot
part, in ratio form:

    f(a, b, m, n)

Aspect ratio and mode numbers only -- no h, E, nu, rho.
b is the TRUE side length, same unit as a.
Constants rounded to {SIG_DIGITS_PY} significant digits.
"""
import numpy as np
'''


def python_definition(model, with_prefactor):
    """The lambda lines: full frequency model, or just the slot."""
    rounded = round_constants(model.canonical, SIG_DIGITS_PY)
    first   = to_python(with_ratio(rounded))

    if not with_prefactor:
        return [f"{model.name} = lambda a, b, m, n: (",
                f"    ({first})",
                ")"]

    # mirror the already rounded expression, so both halves carry exactly
    # the same constants -- no asymmetry from rounding twice
    second = to_python(with_ratio(mirrored(rounded)))

    return [f"{model.name} = lambda a, b, h, m, n, E, nu, rho: (",
            f"    {PREFACTOR_PY}",
            f"    * (((m + ({first}))/a)**2",
            f"       + ((n + ({second}))/b)**2)",
            ")"]


def write_python(models, path, docstring, dict_name, with_prefactor):
    lines = [docstring, "import numpy as np", ""]

    for model in models:
        lines += ["",
                  f"# {summary_line(model)}",
                  f"# HoF slot: f = {describe_slot(model.raw_slot)}"]
        lines += python_definition(model, with_prefactor)

    lines += ["", f"{dict_name} = {{"]
    lines += [f'    "{model.name}": {model.name},' for model in models]
    lines += ["}", ""]

    with open(path, "w") as handle:
        handle.write("\n".join(lines))
    print(f"wrote {path}: {len(models)} models")

# ----------------------------------------------------------- tex writer
TEX_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=2.5cm]{geometry}
\usepackage{amsmath}

\begin{document}

"""

TEX_FULL_TITLE  = r"\section*{Hall-of-Fame equations: full frequency model}"
TEX_DELTA_TITLE = r"\section*{Hall-of-Fame equations: $\Delta_m$}"


def tex_equation(model, with_prefactor, label_prefix):
    """One display equation: full model, or only the slot delta(a/b, m, n)."""
    rounded = round_constants(model.canonical, SIG_DIGITS_TEX)
    first   = to_latex(with_ratio(rounded))

    if with_prefactor:
        # mirror the already rounded expression: identical constants in
        # both halves, no asymmetry from rounding twice
        second = to_latex(with_ratio(mirrored(rounded)))
        head   = rf"f_{{{model.tag}}} = {PREFACTOR_TEX}"
        body   = (r"\left[\left(\frac{m + " + first + r"}{a}\right)^{2}"
                  r" + \left(\frac{n + " + second + r"}{b}\right)^{2}\right]")
    else:
        # much shorter line: no prefactor, only the slot in ratio form
        head   = (rf"\Delta_{{{model.tag}}}"
                  r"\!\left(\tfrac{a}{b},\; m,\; n\right)")
        body   = first

    return [
        "",
        f"% {summary_line(model)}",
        f"% HoF slot (raw): f = {model.raw_slot}",
        f"% canonical (x = a/b): {sp.sstr(model.canonical)}",
        r"\begin{equation}",
        rf"\label{{{label_prefix}:{model.tag}}}",
        f"{head} = {body}",
        r"\end{equation}",
    ]


def write_tex(models, path, with_prefactor):
    """One LaTeX document, either the full model or the slot only."""
    if with_prefactor:
        title   = TEX_FULL_TITLE
        content = "full model f(a,b,h,m,n)"
        prefix  = "eq"
    else:
        title   = TEX_DELTA_TITLE
        content = "slot only, delta(a/b, m, n); prefactor NOT included"
        prefix  = "eqd"

    lines = [
        "% Hall-of-Fame equations, generated by 4_ef_hof_export.py",
        f"% source: {HOF_CSV}",
        f"% content: {content}",
        f"% canonicalised in the ratio x = a/b, constants to "
        f"{SIG_DIGITS_TEX} significant digits",
        f"% {len(models)} distinct equations",
        "",
    ] + TEX_PREAMBLE.splitlines() + ["", title, ""]

    for model in models:
        lines += tex_equation(model, with_prefactor, prefix)
    lines += ["", r"\end{document}", ""]

    with open(path, "w") as handle:
        handle.write("\n".join(lines))
    print(f"wrote {path}: {len(models)} equations "
          f"({'full model' if with_prefactor else 'slot only'})")

# ----------------------------------------------------------------- main
def main():
    rows = read_hall_of_fame(HOF_CSV)
    print(f"{HOF_CSV}: {len(rows)} rows\n")

    models, (n_low, n_bad, n_dup) = collect_models(rows)

    write_python(models, OUT_PY, PY_DOCSTRING,
                 dict_name="HOF_MODELS", with_prefactor=True)
    write_python(models, OUT_DELTA_PY, DELTA_DOCSTRING,
                 dict_name="HOF_DELTA_MODELS", with_prefactor=False)
    write_tex(models, OUT_TEX, with_prefactor=True)
    write_tex(models, OUT_TEX_DELTA, with_prefactor=False)

    print(f"\nskipped {n_low} below complexity {MIN_COMPLEXITY}, "
          f"{n_bad} unparseable, {n_dup} duplicates")


if __name__ == "__main__":
    main()