"""Post-flight step 1: copy the Labelme JSONs you made in the flat label_session/ folder
back into the per-clip subdirs (labeling/frames/<seq>/) where labelme_to_gt.py expects them.

    python labeling/collect_session.py      # then:
    python labeling/labelme_to_gt.py        # -> gt.csv + 4-class masks

Frames you left unlabeled (blink/unusable) simply have no JSON -> the converter marks them
'reject', which is correct. Idempotent: re-running just re-copies.
"""
import glob, os, shutil

SESSION = "labeling/label_session"
n, skipped = 0, 0
for jp in sorted(glob.glob(os.path.join(SESSION, "*.json"))):
    name = os.path.basename(jp)                 # subject_e_1_L_f000307.json
    seq = name.rsplit("_f", 1)[0]               # subject_e_1_L
    dst = os.path.join("labeling", "frames", seq)
    if not os.path.isdir(dst):
        print("  skip (no matching subdir):", name); skipped += 1; continue
    shutil.copy(jp, os.path.join(dst, name)); n += 1
print(f"collected {n} JSON(s) into labeling/frames/<seq>/ ({skipped} skipped).")
print("next: python labeling/labelme_to_gt.py")
