"""Assemble exactly what Overleaf needs, and nothing else.

`paper/figures/` accumulated 29 images across several generations of the manuscript and
the current draft cites 11 of them. Uploading the folder wholesale carries 18 retired
figures into the project, where the only thing distinguishing them from the live ones is
that nothing includes them. This derives the list from the .tex on every run instead of
keeping a second copy of it.

Output: paper/overleaf/  containing first_draft.tex, references.bib and figures/
Run:  conda activate diamond; cd src\\analysis; python build_overleaf_bundle.py
"""
import os
import re
import shutil
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config

PAPER = os.path.join(config.ROOT, "paper")
OUT = os.path.join(PAPER, "overleaf")
TEX = "first_draft.tex"
SEARCH = [os.path.join(PAPER, "figures"),
          os.path.join(config.ROOT, "data", "outputs", "viz"),
          config.OBP_VALIDATION_DIR]


def main():
    t = open(os.path.join(PAPER, TEX), encoding="utf-8").read()
    figs = [os.path.basename(f) for f in
            re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]*)\}", t)]
    bibs = re.findall(r"\\bibliography\{([^}]*)\}", t)
    assert len(figs) == len(set(figs)), "a figure is included twice"

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "figures"))

    shutil.copy2(os.path.join(PAPER, TEX), os.path.join(OUT, TEX))
    for b in bibs:
        src = os.path.join(PAPER, b + ".bib")
        assert os.path.exists(src), f"missing {b}.bib"
        shutil.copy2(src, os.path.join(OUT, b + ".bib"))

    total = 0
    for f in figs:
        hit = None
        for d in SEARCH:
            for ext in (".png", ".pdf", ".jpg", ""):
                c = os.path.join(d, f + ext)
                if os.path.exists(c):
                    hit = c
                    break
            if hit:
                break
        assert hit, f"figure not found anywhere: {f}"
        dst = os.path.join(OUT, "figures", os.path.basename(hit))
        shutil.copy2(hit, dst)
        total += os.path.getsize(dst)
        print(f"   {os.path.basename(hit):<32} {os.path.getsize(dst) / 1e6:6.2f} MB"
              f"   <- {os.path.dirname(hit)}")

    print(f"\n{len(figs)} figures, {total / 1e6:.1f} MB")
    print(f"bundle -> {OUT}")
    # the console here is cp949, so keep stdout ASCII
    print("\n!! ieeeaccess.cls is NOT included and is not in this repository. Start the "
          "Overleaf project from the IEEE Access template, which supplies the class and "
          "IEEEtran.bst, then replace its main .tex with first_draft.tex.")
    big = [f for f in os.listdir(os.path.join(OUT, "figures"))
           if os.path.getsize(os.path.join(OUT, "figures", f)) > 2e6]
    if big:
        print(f"!! over 2 MB, worth re-rendering smaller before submission: {big}")


if __name__ == "__main__":
    main()
