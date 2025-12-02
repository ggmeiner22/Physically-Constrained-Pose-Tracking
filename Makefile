.RECIPEPREFIX := >
PY ?= python3
VENV ?= .venv
ACT := . $(VENV)/bin/activate
VIDEO ?= datasets/marble/marble-on-track-sim-1.mp4
SCEN ?= sliding

.PHONY: venv dev train viz freeze clean

venv:
> $(PY) -m venv $(VENV)
> $(ACT) && python -m pip install --upgrade pip setuptools wheel
> $(ACT) && pip install -r requirements.txt

dev:
> $(ACT) && pip install -e .

manifest:
> $(ACT) && python tools/gen_manifest.py

train:
> $(ACT) && python train.py --model temporal --train_manifest data/manifest_train.csv --val_manifest data/manifest_val.csv --test_manifest data/manifest_test.csv --epochs 10 --batch_size 4 --clip_len 32

traindet:
> $(ACT) && python train_detector.py --manifest data/manifest_train.json

trainpos:
> $(ACT) && python train_position.py --manifest data/manifest_train.json

runclips:
> $(ACT) && python run_pose_model_on_clips.py \
  --manifest path/to/your/video_manifest.csv \
  --scenario hanging \
  --detector_ckpt outputs/detector/best_detector.pt \
  --position_ckpt outputs/position/best_position.pt

viz-one:
> $(ACT) && python tools/viz_one_sim.py \
  --video datasets/marble/test/marble-on-track-sim-15.mp4 \
  --info datasets/marble/test/marble-on-track-info-sim-15.npy \
  --draw_pos \
  --scale 0.7

viz:
> $(ACT) && python tools/visualize_ekf_vs_raw.py --scenario $(SCEN) --video_path $(VIDEO)

freeze:
> $(ACT) && pip freeze > requirements-lock.txt

clean:
> rm -rf $(VENV) build dist *.egg-info __pycache__ **/__pycache__ outputs/runs
