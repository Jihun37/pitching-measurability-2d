"""Emit Table II: the disposition of all 81 OBP poi columns.

This is the table that makes the census checkable -- no column is silently dropped.
It is also the ONE place the accounting appears, because the paper's most dangerous
numeric coincidences live here:

  * two 35s   `n_retained_rows` (a RESULT: rows holding >= 1 graded cell) and the 35
              screened rows named after their own OBP column (a NAMING fact)
  * two 12s   the 12 rows adopted in this study (5 direct-3D + 7 OBP-column, all of
              them retained) and the 12 non-retained rows (all of them screened)

Presenting the accounting once and referencing it afterwards is the structural defence
against substituting one for the other.

Two files are written from the same CSV, so the author can choose without hand-typing:

  table2_coverage.tex        the class x disposition accounting -- 5 body rows
  table2_coverage_full.tex   all 81 columns, grouped by class, for an appendix

Every count is asserted against the frozen accounting before either file is written.

Run:  conda activate diamond; cd src\\analysis; python table2_coverage.py
"""
import os, sys
_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
import pandas as pd
import config
from table1_estimators import tex_escape

CSV = os.path.join(config.OBP_VALIDATION_DIR, "coverage_accounting.csv")
CLASSES = ["metadata", "kinetic", "kinematic"]
DASH = "--"


def accounting(d):
    """The five body rows, all derived. Returns (rows, totals) for assertion."""
    n_total = len(d)
    per = {c: int((d.column_class == c).sum()) for c in CLASSES}
    kin = d[d.column_class == "kinematic"]
    n_kin_eval = int(kin.evaluated.sum())
    n_kin_ret = int(kin.retained.sum())
    n_kin_not = n_kin_eval - n_kin_ret
    # The direct-3D rows are NOT columns: they are evaluated pathways whose truth is the
    # 3D joint set itself, so they appear in the row count and not in the column count.
    n_eval_rows = 47
    n_direct = n_eval_rows - n_kin_eval
    rows = [
        ("metadata", per["metadata"], DASH, DASH, DASH),
        ("kinetic", per["kinetic"], "inference only", DASH, DASH),
        ("kinematic", per["kinematic"], f"{n_kin_eval}", f"{n_kin_ret}",
         f"{n_kin_not}"),
        ("direct 3D truth", DASH, f"{n_direct}", f"{n_direct}", DASH),
    ]
    totals = ("total", n_total, n_kin_eval + n_direct, n_kin_ret + n_direct, n_kin_not)
    return rows, totals, per, (n_kin_eval, n_kin_ret, n_kin_not, n_direct)


def main():
    d = pd.read_csv(CSV)
    rows, totals, per, (n_kin_eval, n_kin_ret, n_kin_not, n_direct) = accounting(d)

    # ---- assertions against the frozen accounting ---------------------------
    assert len(d) == 81, len(d)
    assert per == {"metadata": 5, "kinetic": 34, "kinematic": 42}, per
    assert (n_kin_eval, n_direct) == (42, 5), (n_kin_eval, n_direct)
    assert (n_kin_ret, n_kin_not) == (30, 12), (n_kin_ret, n_kin_not)
    assert totals[1:] == (81, 47, 35, 12), totals
    # ⚠ This CSV's `evaluated` and `retained` flags count COLUMNS, so they read 42 and
    # 30 -- NOT the registry's 47 evaluated rows and 35 retained rows. The 5 direct-3D
    # rows have no OBP column to flag. Quoting 42 as "evaluated rows" off this file
    # would contradict the registry; the row counts come from the totals line below.
    assert int(d.evaluated.sum()) == 42, int(d.evaluated.sum())
    assert int(d.retained.sum()) == 30, int(d.retained.sum())
    off = d[d.evaluated & ~d.retained]
    assert len(off) == 12, len(off)
    assert round(off.best_ccc.max(), 3) == 0.680, off.best_ccc.max()
    assert round(off.best_r2.max(), 3) == 0.471, off.best_r2.max()

    print(f"81 columns = " + " + ".join(f"{v} {k}" for k, v in per.items()))
    print(f"47 evaluated rows = {n_kin_eval} kinematic-column pathways "
          f"+ {n_direct} direct-3D-truth rows")
    print(f"   of the {n_kin_eval} kinematic pathways: {n_kin_ret} retained, "
          f"{n_kin_not} not retained")
    print(f"   the {n_direct} direct-3D rows are all retained -> 35 retained, 12 not")
    print(f"non-retained ceiling: best CCC {off.best_ccc.max():.4f}, "
          f"best r2 {off.best_r2.max():.4f}  (no borderline case)")
    # ASCII only: the console here is cp949 and a warning glyph aborts the run
    print("NOTE retained==True in this CSV counts COLUMNS (30), not paper rows (35): "
          "the 5 direct-3D rows have no column to mark.")

    # ---- body table ---------------------------------------------------------
    out = ["\\begin{table}[htbp]",
           "\\renewcommand{\\arraystretch}{1.15}",
           "\\caption{Disposition of all "
           f"{len(d)}" " OBP performance-of-interest columns.}",
           "\\label{tab:coverage}",
           "\\centering\\footnotesize",
           "\\begin{tabular}{@{}lrrrr@{}}",
           "\\hline",
           "Class & Columns & Evaluated & Retained & Not ret.\\\\",
           "\\hline"]
    for name, cols, ev, ret, nr in rows:
        out.append(f"{name} & {cols} & {ev} & {ret} & {nr}\\\\")
    out.append("\\hline")
    t = totals
    out.append(f"{t[0]} & {t[1]} & {t[2]} & {t[3]} & {t[4]}\\\\")
    out.append("\\hline")
    out.append("\\end{tabular}")
    out.append("\\end{table}")
    body = "\n".join(out)

    p1 = os.path.join(config.OBP_VALIDATION_DIR, "table2_coverage.tex")
    open(p1, "w", encoding="utf-8").write(body + "\n")
    print("\n" + body)

    # ---- full listing, for an appendix --------------------------------------
    full = ["\\begin{table*}[htbp]",
            "\\renewcommand{\\arraystretch}{1.10}",
            "\\caption{Every OBP column, its class, its pathway and its outcome.}",
            "\\label{tab:coverage-full}",
            "\\centering\\scriptsize",
            "\\begin{tabular}{@{}p{0.30\\textwidth}p{0.28\\textwidth}"
            "p{0.10\\textwidth}rrl@{}}",
            "\\hline",
            "Column & Pathway & Outcome & CCC & $r^{2}$ & View\\\\",
            "\\hline"]
    for c in CLASSES:
        sub = d[d.column_class == c]
        full.append("\\multicolumn{6}{@{}l}{\\textit{" + c
                    + f" ({len(sub)} columns)" + "}}\\\\")
        for r in sub.itertuples(index=False):
            if not r.evaluated:
                path = (tex_escape(str(r.pathway)) if isinstance(r.pathway, str)
                        and r.pathway else DASH)
                full.append(f"{tex_escape(r.column)} & {path} & {DASH} & {DASH} & "
                            f"{DASH} & {DASH}\\\\")
                continue
            outcome = "retained" if r.retained else "not retained"
            view = str(r.best_view).replace("/", "\\,/\\,")
            full.append(f"{tex_escape(r.column)} & {tex_escape(str(r.pathway))} & "
                        f"{outcome} & {r.best_ccc:.3f} & {r.best_r2:.3f} & "
                        f"{view}\\\\")
        full.append("\\hline")
    full.append("\\end{tabular}")
    full.append("\\end{table*}")
    p2 = os.path.join(config.OBP_VALIDATION_DIR, "table2_coverage_full.tex")
    open(p2, "w", encoding="utf-8").write("\n".join(full) + "\n")

    print(f"\n% body    {len(rows) + 1} rows -> {p1}", file=sys.stderr)
    print(f"% full    {len(d)} columns -> {p2}", file=sys.stderr)


if __name__ == "__main__":
    main()
