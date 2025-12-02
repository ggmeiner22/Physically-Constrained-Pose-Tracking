from __future__ import annotations

from pathlib import Path
import json
import re

from pcpose.data.preprocess import build_manifest


SCENARIOS = ["drop", "marble", "pendulum"]
SPLITS = ["train", "val", "test"]


def find_matching_npy(mp4_path: Path) -> Path:
    """
    Given an mp4 like 'marble-on-track-sim-15.mp4',
    find the corresponding .npy in the same directory.

    We look for the 'sim-<number>' pattern and match any .npy
    that also contains that tag.
    """
    stem = mp4_path.stem  # e.g. 'marble-on-track-sim-15'
    m = re.search(r"sim-\d+", stem)
    if not m:
        raise RuntimeError(f"Could not find 'sim-<id>' pattern in {stem}")

    tag = m.group(0)  # 'sim-15'

    candidates = list(mp4_path.parent.glob(f"*{tag}*.npy"))
    if len(candidates) == 0:
        raise FileNotFoundError(f"No .npy found in {mp4_path.parent} matching tag '{tag}'")
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple .npy files in {mp4_path.parent} matching tag '{tag}': {candidates}"
        )
    return candidates[0]


def build_split_manifest(root: Path, split: str, out_root: Path) -> None:
    """
    Traverse datasets/<scenario>/<split>/ for all scenarios and
    aggregate into a single JSON manifest for this split.

    Writes: out_root / f"manifest_{split}.json"
    """
    all_records = []

    for scenario in SCENARIOS:
        split_dir = root / scenario / split
        if not split_dir.exists():
            print(f"[WARN] Missing directory {split_dir}, skipping scenario '{scenario}' for split '{split}'")
            continue

        print(f"[INFO] Processing scenario='{scenario}', split='{split}' in {split_dir}")

        for mp4 in split_dir.glob("*.mp4"):
            try:
                npy = find_matching_npy(mp4)
            except Exception as e:
                print(f"  [WARN] Skipping {mp4.name}: {e}")
                continue

            # Where to put extracted frames and per-video manifest
            frames_dir = out_root / scenario / split / f"{mp4.stem}_frames"
            per_video_manifest = out_root / scenario / split / f"{mp4.stem}_manifest.json"

            frames_dir.parent.mkdir(parents=True, exist_ok=True)

            print(f"   - {mp4.name}  +  {npy.name}")
            build_manifest(mp4, npy, per_video_manifest, frames_dir)

            # Append per-video records to combined list
            with per_video_manifest.open("r") as f:
                recs = json.load(f)
                # (optional) annotate with scenario if you ever need it
                for r in recs:
                    r["scenario"] = scenario
                all_records.extend(recs)

    combined_path = out_root / f"manifest_{split}.json"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with combined_path.open("w") as f:
        json.dump(all_records, f, indent=2)

    print(f"[OK] Wrote combined {split} manifest with {len(all_records)} records → {combined_path}")


def main() -> None:
    datasets_root = Path("datasets")
    out_root = Path("data")

    for split in SPLITS:
        build_split_manifest(datasets_root, split, out_root)


if __name__ == "__main__":
    main()
