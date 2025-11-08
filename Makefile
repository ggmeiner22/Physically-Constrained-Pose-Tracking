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
> $(ACT) && python tools/gen_manifest.py --root datasets --outdir data

train:
> $(ACT) && python train.py --model temporal --train_manifest data/manifest_train.csv --val_manifest data/manifest_val.csv --test_manifest data/manifest_test.csv --epochs 10 --batch_size 4 --clip_len 32

viz:
> $(ACT) && python tools/visualize_ekf_vs_raw.py --scenario $(SCEN) --video_path $(VIDEO)

freeze:
> $(ACT) && pip freeze > requirements-lock.txt

clean:
> rm -rf $(VENV) build dist *.egg-info __pycache__ **/__pycache__ outputs/runs
