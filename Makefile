PY ?= python3
VENV ?= .venv
ACT := source $(VENV)/bin/activate
VIDEO ?= datasets/sample.mp4
SCEN ?= hanging


.PHONY: venv dev clean train viz freeze


venv:
    $(PY) -m venv $(VENV)
    $(ACT) && python -m pip install --upgrade pip setuptools wheel
    $(ACT) && pip install -r requirements.txt


# Editable install of the package
dev:
    $(ACT) && pip install -e .


train:
    $(ACT) && python train.py --scenario $(SCEN) --video_path $(VIDEO)


viz:
    $(ACT) && python tools/visualize_ekf_vs_raw.py --scenario $(SCEN) --video_path $(VIDEO)


freeze:
    $(ACT) && pip freeze > requirements-lock.txt


clean:
    rm -rf $(VENV) build dist *.egg-info __pycache__ **/__pycache__ outputs/runs

