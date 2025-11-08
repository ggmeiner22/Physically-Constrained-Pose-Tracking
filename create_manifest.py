import os, csv
root = "datasets/marble/train"
rows = []
for f in sorted(os.listdir(root)):
    if f.endswith(".mp4"):
        rows.append({
            "path": os.path.join(root, f),
            "scenario": "sliding",
            "anchor_x": "",
            "anchor_y": "",
            "cable_length": "",
            "slide_y": "128",      # or whatever y-position your track center is
            "mu_slide": "0.5",
            "pendulum_damping": ""
        })

os.makedirs("data", exist_ok=True)
out = "data/manifest.csv"
with open(out, "w", newline="") as fp:
    w = csv.DictWriter(fp, fieldnames=["path","scenario","anchor_x","anchor_y","cable_length","slide_y","mu_slide","pendulum_damping"])
    w.writeheader()
    w.writerows(rows)
print(f"Wrote {out} with {len(rows)} videos.")
