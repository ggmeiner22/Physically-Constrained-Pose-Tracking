# Physically-Constrained-Pose-Tracking

A modular PyTorch + OpenCV pipeline for physics-aware pose tracking. Predict poses from video, then enforce physical plausibility (gravity, cable length, friction limits) through soft constraints and EKF filtering.

---

## Features
- Perception backbone (swap-in ViT/OpenPose/DeepLabCut)
- Physics layer: Hanging (pendulum), Sliding (line + friction), Dropping
- Filtering: EMA or Extended Kalman Filter (CV + gravity)
- Unified loss and simple visualizations

---

## Configuration (common flags)
```bash
--scenario {hanging,sliding,dropping}
--filter {ekf,ema}
--use_vel_meas
--epochs 10 --lr 1e-3
--lambda_phys 5.0 --lambda_smooth 0.5
--video_path datasets/sample.mp4
--max_frames 300
```

---

## Running the Software
macOS / Linux (recommended)
  1. Install prerequisites
```
python3 --version    
```
>  Use 3.10+

  2. Open a terminal in your project folder
  3. Create the venv + install deps
```
make venv
```

  4. (Optional) Editable install for imports
```
make dev
```

  5. Run a quick training
```
make train
```
> By default --MODEL=temporal, VIDEO=datasets/sample.mp4 and SCEN=hanging

  6. Render EKF vs RAW comparison
```
make viz
```
> Want to point at a different model, video, or scenario?
```
make train MODEL=tiny VIDEO=/path/to/your.mp4 SCEN=sliding
make viz   VIDEO=/path/to/your.mp4 SCEN=sliding
```

  7. Freeze the environment (optional)
```
make freeze
```   

Windows options
  1. Install Git for Windows (includes Git Bash).
  2. Open Git Bash in the project folder.
  3. Run the same commands:
```
make venv
make dev
make train VIDEO=datasets/sample.mp4 SCEN=hanging
```

---

## Verification
After make train, you should see epoch logs and an overlay video at:
```
outputs/runs/viz/overlay.mp4
```

After make viz, you should see:
```
outputs/runs/viz/ekf_vs_raw.png
```
