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

train:
> $(ACT) && python train.py --model temporal --scenario $(SCEN) --video_path $(VIDEO)

viz:
> $(ACT) && python tools/visualize_ekf_vs_raw.py --scenario $(SCEN) --video_path $(VIDEO)

freeze:
> $(ACT) && pip freeze > requirements-lock.txt

clean:
> rm -rf $(VENV) build dist *.egg-info __pycache__ **/__pycache__ outputs/runs
