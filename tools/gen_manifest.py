#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, re
from pathlib import Path

SCEN_BY_NAME = [
    (re.compile(r"(pendulum|hang|rope)", re.I), "hanging"),
    (re.compile(r"(marble|track|slide)", re.I), "sliding"),
    (re.compile(r"(drop|fall)", re.I), "dropping"),
]

def guess_scenario(path: str) -> str:
    p = path.lower()
    for pat, scen in SCEN_BY_NAME:
        if pat.search(p):
            return scen
    # also check immediate parent folder name as a fallback
    parts = Path(path).parts
    for s in parts[::-1]:
        for pat, scen in SCEN_BY_NAME:
            if pat.search(s):
                return scen
    return "sliding"

def write_manifest(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "path","scenario","anchor_x","anchor_y","cable_length","slide_y","mu_slide","pendulum_damping"
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_path} with {len(rows)} items")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="datasets")
    ap.add_argument("--outdir", type=str, default="data")
    ap.add_argument("--exts", type=str, default=".mp4,.mov,.avi,.mkv")
    ap.add_argument("--default-slide-y", type=str, default="128")
    ap.add_argument("--default-mu-slide", type=str, default="0.5")
    ap.add_argument("--default-cable-length", type=str, default="")
    ap.add_argument("--default-anchor", type=str, default="128,0")
    ap.add_argument("--default-pendulum-damping", type=str, default="0.05")
    args = ap.parse_args()

    root = Path(args.root)
    exts = {e.strip().lower() for e in args.exts.split(",") if e.strip()}

    train_rows, val_rows, test_rows = [], [], []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        scen = guess_scenario(str(p))
        # defaults per scenario
        anchor_x = anchor_y = ""
        cable_length = ""
        slide_y = ""
        mu_slide = ""
        pendulum_damping = ""
        if scen == "sliding":
            slide_y = args.default_slide-y if hasattr(args, "default_slide-y") else args.default_slide_y  # robustness
            slide_y = args.default_slide_y
            mu_slide = args.default_mu_slide
        elif scen == "hanging":
            cable_length = args.default_cable_length
            try:
                ax, ay = (s.strip() for s in args.default_anchor.split(","))
                anchor_x, anchor_y = ax, ay
            except Exception:
                pass
            pendulum_damping = args.default_pendulum_damping

        row = {
            "path": p.as_posix(),
            "scenario": scen,
            "anchor_x": anchor_x,
            "anchor_y": anchor_y,
            "cable_length": cable_length,
            "slide_y": slide_y,
            "mu_slide": mu_slide,
            "pendulum_damping": pendulum_damping,
        }

        pl = p.as_posix().lower()
        if "/train/" in pl:
            train_rows.append(row)
        elif "/val/" in pl:
            val_rows.append(row)
        elif "/test/" in pl:
            test_rows.append(row)
        else:
            # if no split in path, default to train
            train_rows.append(row)

    # Stable ordering
    train_rows.sort(key=lambda r: r["path"])
    val_rows.sort(key=lambda r: r["path"])
    test_rows.sort(key=lambda r: r["path"])

    outdir = Path(args.outdir)
    write_manifest(train_rows, outdir / "manifest_train.csv")
    write_manifest(val_rows,   outdir / "manifest_val.csv")
    write_manifest(test_rows,  outdir / "manifest_test.csv")

if __name__ == "__main__":
    main()
