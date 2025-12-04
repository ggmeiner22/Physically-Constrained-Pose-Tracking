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
> $(ACT) && python train.py \
  --model two_stage \
  --detector_ckpt outputs/detector/best_detector.pt \
  --position_ckpt outputs/position/best_position.pt \
  --train_manifest data/manifest_train.json \
  --val_manifest   data/manifest_val.json \
  --scenario pendulum \
  --filter ekf

traindet:
> $(ACT) && python train_detector.py \
  --manifest data/manifest_train.json \
  --val_manifest data/manifest_val.json \
  --outdir outputs/detector \
  --batch_size 2 \
  --num_workers 4

trainpos:
> $(ACT) && python train_position.py \
  --manifest data/manifest_train.json \
  --val_manifest data/manifest_val.json \
  --outdir outputs/position \
  --batch_size 2 \
  --num_workers 4

runclips:
> $(ACT) && python run_pose_model_on_clips.py \
  --manifest data/manifest_test.json \
  --detector_ckpt outputs/detector/best_detector.pt \
  --position_ckpt outputs/position/best_position.pt

viz-one:
> $(ACT) && python tools/viz_one_sim.py \
  --video datasets/pendulum/test/pendulum-sim-15.mp4 \
  --info datasets/pendulum/test/pendulum-info-sim-15.npy \
  --draw_pos \
  --scale 0.7

viz:
> $(ACT) && python tools/visualize_ekf_vs_raw.py \
  --video_path datasets/pendulum/test/pendulum-sim-15.mp4 \
  --scenario pendulum \
  --model two_stage \
  --detector_ckpt outputs/detector/best_detector.pt \
  --position_ckpt outputs/position/best_position.pt

freeze:
> $(ACT) && pip freeze > requirements-lock.txt

clean:
> rm -rf $(VENV) build dist *.egg-info __pycache__ **/__pycache__ outputs/runs
