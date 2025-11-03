# Physically-Constrained-Pose-Tracking

## Enviornment Setup
  1) Use Python 3.10 or 3.11 (PyTorch wheels are smooth there).
```
python3 --version
```

  2) Create & activate the virtual environment
2a) macOS/Linux
```
python3 -m venv .venv
source .venv/bin/activate
```
2b) Windows (PowerShell)
```
py -m venv .venv
.venv\Scripts\Activate
```

  3) Upgrade packaging tools
```
python -m pip install --upgrade pip setuptools wheel
```

  4) Install dependencies
```
pip install -r requirements.txt
```
> ⚠️ If PyTorch fails from requirements.txt (common on GPUs), install it first using the official command for your CUDA/OS, then run pip install -r requirements.txt again. Example (CPU-only):
```
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

  5) Run the project
5a) Training
```
python train.py --scenario hanging --video_path datasets/sample.mp4 --filter ekf
```
> If your backbone's velocity outputs are reliable:
```
python train.py --scenario hanging --video_path datasets/sample.mp4 --filter ekf --use_vel_meas
```

5b) Visualization
```
python tools/visualize_ekf_vs_raw.py --video_path datasets/sample.mp4 --scenario hanging
```
> If you trust backbone velocities as measurements:
```
python tools/visualize_ekf_vs_raw.py --video_path datasets/sample.mp4 --scenario hanging --use_vel_meas
```

  6) Save your environment (optional)
```
pip freeze > requirements-lock.txt
```
  7) Deactivate when done
```
deactivate
```

---



