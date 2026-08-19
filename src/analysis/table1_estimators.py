"""Emit Table I of the paper: the 47 evaluated estimator rows.

Columns: quantity, the temporal anchor or reading rule it is read at, and its
ground truth. Generated from layer_summary.csv so the row set can never drift
from the map. Anchors are derived from the row definition, not typed by hand.

Run:  conda activate diamond; cd src\\analysis; python table1_estimators.py
Writes table1_estimators.tex next to the other outputs and prints it.
"""
import os, sys, re
_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
import pandas as pd
import config

# The 12 rows defined in this study: display name, anchor, truth.
OWN = {
    "Lead Knee Angle [O]":  ("Lead knee angle", "release", "3D joints"),
    "Trunk Tilt (ant) [O]": ("Trunk anterior tilt", "release",
                             "torso\\_anterior\\_tilt\\_br"),
    "Arm Slot [O]":         ("Arm slot", "release", "3D joints"),
    "Release Height [O]":   ("Release height", "release", "3D joints"),
    "Release Ext [O]":      ("Release extension", "release", "3D joints"),
    "Wrist Speed [O]":      ("Wrist speed", "window max", "3D joints"),
    "Stride (anchor) [O]":  ("Stride length", "settled read",
                             "stride\\_length"),
    "Stride Angle [O]":     ("Stride angle", "foot plant", "stride\\_angle"),
    "Hip-Shoulder Sep [O]": ("Hip--shoulder separation", "signature anchor $t^{*}$",
                             "max\\_rotation\\_hip\\_shoulder\\_separation"),
    "Pelvis Rot Velo [O]":  ("Pelvis rotational velocity", "window max",
                             "max\\_pelvis\\_rotational\\_velo"),
    "COG Fwd Velo [O]":     ("COM forward velocity", "window max",
                             "max\\_cog\\_velo\\_x"),
    "COG Velo @PKH [O]":    ("COM velocity at peak knee height",
                             "peak knee height", "cog\\_velo\\_pkh"),
}

# Column-named rows whose anchor is not given by a plain suffix.
SPECIAL = {
    "arm_slot": "release",
    "lead_knee_extension_from_fp_to_br": "foot plant to release",
    "lead_knee_extension_angular_velo_max": "window max, [fp, rel$+$12\\,f]",
    "timing_peak_torso_to_peak_pelvis_rot_velo": "interval, two window maxima",
    "torso_rotation_min": "window min",
}


def anchor_of(col):
    if col in SPECIAL:
        return SPECIAL[col]
    if col.endswith("_fp"):
        return "foot plant"
    if col.endswith("_br"):
        return "release"
    if col.endswith("_mer"):
        return "MER"
    if col.startswith("max_"):
        return "window max"
    return "--"


def pretty(col):
    s = col.replace("_", " ")
    s = re.sub(r"\bfp\b", "@ foot plant", s)
    s = re.sub(r"\bbr\b", "@ release", s)
    s = re.sub(r"\bmer\b", "@ MER", s)
    s = re.sub(r"^max ", "peak ", s)
    s = s.replace("velo", "velocity").replace("rot ", "rotational ")
    s = s.replace("cog", "COM").replace("sh ", "shoulder ")
    return s[0].upper() + s[1:]


def tex_escape(s):
    return s.replace("_", "\\_")


def main():
    d = pd.read_csv(os.path.join(config.OBP_VALIDATION_DIR, "layer_summary.csv"))
    rows = []
    for r in d.itertuples(index=False):
        m = str(r.metric).strip()
        if m in OWN:
            q, a, t = OWN[m]
        else:
            q, a, t = pretty(m), anchor_of(m), tex_escape(m)
        rows.append((q, a, t, m in OWN))
    rows.sort(key=lambda x: (not x[3], x[0]))

    out = []
    out.append("\\begin{table*}[htbp]")
    out.append("\\renewcommand{\\arraystretch}{1.15}")
    # Caption is ONE LINE, says what the table is and nothing else (user rule).
    # The count is INTERPOLATED: it read "the 47 evaluated" as a literal, which a
    # re-run could not have refreshed if the registry ever moved.
    out.append(f"\\caption{{Quantity, temporal anchor and ground truth of the "
               f"{len(rows)} evaluated estimator rows.}}")
    out.append("\\label{tab:estimators}")
    out.append("\\centering\\footnotesize")
    out.append("\\begin{tabular}{@{}p{0.30\\textwidth}p{0.20\\textwidth}"
               "p{0.42\\textwidth}@{}}")
    out.append("\\hline")
    out.append("Quantity & Anchor or reading rule & Ground truth\\\\")
    out.append("\\hline")
    for q, a, t, _own in rows:
        out.append(f"{q} & {a} & {t}\\\\")
    out.append("\\hline")
    out.append("\\end{tabular}")
    out.append("\\end{table*}")

    tex = "\n".join(out)
    p = os.path.join(config.OBP_VALIDATION_DIR, "table1_estimators.tex")
    open(p, "w", encoding="utf-8").write(tex + "\n")
    print(tex)
    print(f"\n% {len(rows)} rows written -> {p}", file=sys.stderr)


if __name__ == "__main__":
    main()
