"""Emit Table III: fp_100 vs fp_10 over the foot-plant-dependent rows.

The claim is that the DEFINITION of an event is part of the measurement. The same foot
plant, taken at the 100 %% threshold rather than the 10 %% one, changes which cells of the
map exist at all -- two rows are measurable only at fp_100.

Sources, all canonical:
  fp_target_rows.csv    per-row cell counts under each anchor  (rows holding >= 1 cell)
  fp_target_cells.csv   the cell-level dump -- its length IS the 629 denominator
  paper_registry.csv    which rows are fp-dependent at all (anchor_type carries `fp`)

⚠ The fp-dependent row count is 27 and comes from the REGISTRY, not from this table's
own length. A window observable carries the event key of its FAR end (`rel`) while
reading the whole [fp, rel] window, so six rows were once filed as release-anchored;
the registry's `anchor_type` already corrects for that. `fp_target_rows.csv` lists only
the rows that hold a graded cell under one anchor or the other, so it is shorter.

Run:  conda activate diamond; cd src\\analysis; python table3_fp_anchor.py
"""
import os, sys
_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
import pandas as pd
import config
from table1_estimators import OWN, pretty

V = config.OBP_VALIDATION_DIR


# Generic phrase collapses -- NOT a metric list. Every row here is anchor-related, so
# the long-form anchor phrases `pretty()` emits repeat down the whole column.
ABBREV = [("@ foot plant", "@ fp"), ("@ release", "@ rel"), ("@ MER", "@ MER"),
          ("extension from @ fp to @ rel", "extension, fp$\\to$rel"),
          ("angular velocity", "ang. velocity"),
          ("horizontal abduction", "hz. abduction")]


def display(m):
    """Paper name for the adopted rows, prettified column name otherwise. Both are
    already free of underscores, so no escaping is needed."""
    s = OWN[m][0] if m in OWN else pretty(m)
    for a, b in ABBREV:
        s = s.replace(a, b)
    return s


def signed(v):
    """`{:+d}` renders a zero delta as `+0`, which reads like a rounded positive."""
    return f"{v:+d}" if v else "0"


def main():
    rows = pd.read_csv(os.path.join(V, "fp_target_rows.csv"))
    cells = pd.read_csv(os.path.join(V, "fp_target_cells.csv"))
    reg = pd.read_csv(os.path.join(V, "paper_registry.csv"))

    n_fp_rows = int(reg.anchor_type.fillna("").str.contains("fp").sum())
    n_cells = len(cells)
    g100, g10 = int(rows.graded_fp100.sum()), int(rows.graded_fp10.sum())
    p100, p10 = int(rows.prefers_fp100.sum()), int(rows.prefers_fp10.sum())
    ties = int(rows.ties.sum())
    only100 = rows[rows.graded_fp10 == 0]

    # ---- assertions against the frozen anchor numbers -----------------------
    assert n_fp_rows == 27, n_fp_rows
    assert n_cells == 629, n_cells
    assert int(rows.cells.sum()) == n_cells, (int(rows.cells.sum()), n_cells)
    assert (p100, p10, ties) == (474, 155, 0), (p100, p10, ties)
    assert (g100, g10) == (577, 471), (g100, g10)
    assert g100 - g10 == 106, g100 - g10
    assert len(only100) == 2, len(only100)
    assert p100 + p10 + ties == n_cells, (p100 + p10 + ties, n_cells)

    print(f"{n_fp_rows} fp-dependent rows in the registry; "
          f"{len(rows)} hold a graded cell under one anchor or the other, so "
          f"{n_fp_rows - len(rows)} hold none")
    print(f"{n_cells} cells graded under either anchor: "
          f"{p100} prefer fp_100, {p10} prefer fp_10, {ties} exact ties")
    print(f"graded cells {g100} (fp_100) vs {g10} (fp_10) = {g100 - g10:+d}")
    print("measurable only at fp_100: "
          + ", ".join(f"{r.metric_id} (+{r.delta})" for r in only100.itertuples()))

    d = rows.sort_values("delta", ascending=False)
    # table* : seven columns plus row names this long do not fit one IEEE column.
    # No \multirow -- the two-line header is built from \multicolumn and \cline only,
    # so the table needs no package the IEEE class does not already load.
    out = ["\\begin{table*}[htbp]",
           "\\renewcommand{\\arraystretch}{1.15}",
           "\\caption{Foot-plant threshold fp\\_100 against fp\\_10 over the "
           f"{n_fp_rows}" " foot-plant-dependent rows.}",
           "\\label{tab:fp-anchor}",
           "\\centering\\footnotesize",
           "\\begin{tabular}{@{}p{0.42\\textwidth}rrrrrr@{}}",
           "\\hline",
           "Row & Cells & \\multicolumn{2}{c}{Graded} & "
           "\\multicolumn{2}{c}{Prefers} & $\\Delta$\\\\",
           "\\cline{3-4}\\cline{5-6}",
           " & & fp\\_100 & fp\\_10 & fp\\_100 & fp\\_10 & \\\\",
           "\\hline"]
    for r in d.itertuples(index=False):
        out.append(f"{display(r.metric_id)} & {r.cells} & {r.graded_fp100} & "
                   f"{r.graded_fp10} & {r.prefers_fp100} & {r.prefers_fp10} & "
                   f"{signed(r.delta)}\\\\")
    out.append("\\hline")
    out.append(f"total & {n_cells} & {g100} & {g10} & {p100} & {p10} & "
               f"{signed(g100 - g10)}\\\\")
    out.append("\\hline")
    out.append("\\end{tabular}")
    out.append("\\end{table*}")

    tex = "\n".join(out)
    p = os.path.join(V, "table3_fp_anchor.tex")
    open(p, "w", encoding="utf-8").write(tex + "\n")
    print("\n" + tex)
    print(f"\n% {len(d)} rows + total -> {p}", file=sys.stderr)


if __name__ == "__main__":
    main()
