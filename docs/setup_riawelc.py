"""
Run this from the repo root on the VM:
    python3 setup_riawelc.py
"""
import os, shutil

src_root = "/tmp/riawelc_extracted/DB - Copy"
dst_root = "datasets/riawelc"

class_map = {
    "Difetto1": "lack_of_penetration",
    "Difetto2": "porosity",
    "Difetto4": "cracks",
    "NoDifetto": "no_defect",
}

total = 0
for split in ["training", "testing", "validation"]:
    for orig, eng in class_map.items():
        src = os.path.join(src_root, split, orig)
        dest = os.path.join(dst_root, split, eng)
        os.makedirs(dest, exist_ok=True)
        files = [f for f in os.listdir(src) if f.endswith(".png")]
        for f in files:
            shutil.move(os.path.join(src, f), os.path.join(dest, f))
        total += len(files)
        print(f"  {split}/{eng}: {len(files)}")

print(f"\nTotal: {total} images moved to {dst_root}/")

# Cleanup
shutil.rmtree("/tmp/riawelc_raw", ignore_errors=True)
shutil.rmtree("/tmp/riawelc_extracted", ignore_errors=True)
print("Temp files cleaned up. Done!")
