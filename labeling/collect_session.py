"""Copy JSONs from an optional flat Labelme session into canonical frames_v2 directories.

    python labeling/collect_session.py      # then:
    python labeling/labelme_to_gt.py        # -> gt.csv + 4-class masks

No JSON means pending. Mark unusable frames explicitly with a rectangle named ``reject``.
Idempotent: re-running copies only JSON files whose image basename exists in frames_v2.
"""
import glob, os, shutil

SESSION = "labeling/label_session"
destinations = {}
for png in glob.glob("labeling/frames_v2/*/*.png"):
    destinations[os.path.basename(png)[:-4] + ".json"] = os.path.dirname(png)
n, skipped = 0, 0
for jp in sorted(glob.glob(os.path.join(SESSION, "*.json"))):
    name = os.path.basename(jp)
    dst = destinations.get(name)
    if dst is None:
        print("  skip (no matching frames_v2 image):", name); skipped += 1; continue
    shutil.copy(jp, os.path.join(dst, name)); n += 1
print(f"collected {n} JSON(s) into labeling/frames_v2/<seq>/ ({skipped} skipped).")
print("next: python labeling/labelme_to_gt.py")
