"""Remove the working notes from references.bib.

IEEEtran prints the `note` field, so entries carrying "[VERIFIED 2026-08-02]" or
"[DOI from database; page range TO BE CONFIRMED]" would ship those into the reference
list. They are internal provenance marks from the citation audit, not part of any
reference. Notes that belong to the reference itself, such as an access date on a web
resource, are kept.

⚠ Two of the notes say a page range is UNCONFIRMED. Stripping the note hides the warning
without answering it, so those keys are printed for the author to settle before
submission.

Output: references.bib rewritten; references_with_notes.bib keeps the annotated copy.
Run:  conda activate diamond; cd src\\analysis; python strip_bib_notes.py
"""
import os
import re
import shutil

PAPER = r"D:\project\diamond\paper"
SRC = os.path.join(PAPER, "references.bib")
KEEP = os.path.join(PAPER, "references_with_notes.bib")

t = open(SRC, encoding="utf-8").read()
if not os.path.exists(KEEP):
    shutil.copy2(SRC, KEEP)

# which entries carry an unresolved warning, so the author can settle them
unresolved = []
for m in re.finditer(r"@\w+\s*\{\s*([^,\s]+)(.*?)\n\}", t, re.S):
    if re.search(r"CONFIRM|NOT RE-VERIFIED|TODO|CHECK", m.group(2), re.I):
        note = re.search(r"note\s*=\s*\{([^}]*)\}", m.group(2))
        unresolved.append((m.group(1), note.group(1) if note else "?"))

WORKING = re.compile(
    r",?\s*note\s*=\s*\{\s*\[(?:VERIFIED|DOI from database|EDITION NOT RE-VERIFIED)"
    r"[^}]*\}", re.I)
out, n = WORKING.subn("", t)
# a stripped note can leave a dangling comma before the closing brace
out = re.sub(r",(\s*\n\})", r"\1", out)

open(SRC, "w", encoding="utf-8").write(out)
kept = len(re.findall(r"note\s*=\s*\{", out))
print(f"stripped {n} working notes; {kept} legitimate notes kept")
print(f"annotated copy preserved at {os.path.basename(KEEP)}")
n_entries = len(re.findall(r"@\w+\s*\{", out))
print(f"entries still {n_entries}")
if unresolved:
    print("\n!! settle these before submission, the warning is now no longer printed:")
    for k, v in unresolved:
        print(f"   {k:<24} {v}")
