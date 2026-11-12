# Violetta mini-game: current status

**This replaces the YOLO plan below.** `src/detection/violetta_solver.py` solves the game with
no ML. It uses the game's own sprites (`assets/violetta/`, taken from the MIT-licensed simulator
https://github.com/hikaricalyx/violetta-sim) and the way the shuffle actually works:

- During a shuffle each Violetta walks in its own depth row (the front row or one of the three
  deepest), and each row has a fixed size and height. The row a Violetta walks in identifies
  it, so crossings never need to be untangled: the solver scans each row's band for the
  walking sprite.
- Every walk steps into its row in place, moves sideways, and steps out in place. A row's first
  x is its start slot and its last x is its end slot. Both ends are solved as an assignment, and
  a check confirms exactly one Violetta stayed put (the game always keeps one in place).
- The stage lights (80% colour drawn over the characters) are handled by local contrast
  normalisation.
- The window (scale and position) is found once, by matching the stage backdrop during the reveal.
- The make-up Violetta is found at the reveal by comparing the special (eyes-open) sprite
  with the normal one in each slot.
- It carries the target across up to 4 rounds and clicks each answer when the buttons light up.

Validation:

- `violetta-sol.mp4`: started in slot 2, answered **3**, which matches the in-game reveal.
- `violetta-vid.mp4`: started in slot 4, answered **1** (the clip has no reveal; paths checked by eye).
- 36/36 synthetic games (random scale and position, shuffles following the game's rules,
  colour lights, 15 and 30 fps).
- `tests/test_violetta_solver.py` covers calibration, multi-round shuffles and the guess fallback.
- About 30 ms per frame, fast enough for live capture.

Usage (from the repo root):

```powershell
.\venv\Scripts\python.exe -m src.detection.violetta_solver --video "C:\Users\sumee\Videos\violetta-sol.mp4" --output annotated.mp4
.\venv\Scripts\python.exe -m src.detection.violetta_solver              # live preview: prints answers
.\venv\Scripts\python.exe -m src.detection.violetta_solver --select     # live: clicks answers
```

Start live mode before the opening reveal ends, because it needs to see which Violetta has make-up.

Not verified yet: a live run against the real game, including whether the click actually
registers, and rounds 2-4 (both recordings stop after round 1). The older experiments
(`violetta_challenge.py`, `violetta_poc.py`, `violetta_yolo_poc.py`) are no longer needed.

---

# Violetta YOLO-seg POC handoff (superseded)

## Goal and constraints

Train a small instance-segmentation detector to find every visible Violetta, then evaluate identity tracking separately. The opening reveal identifies the target as the second copy (slot 2). After initialization, association uses only mask-derived position, motion, scale, aspect ratio, and rotation-normalized contour similarity. Do not use face/eye appearance after the reveal. If detections or assignments are ambiguous, report unknown rather than switch IDs.

## Current status

- Added a local sampling, polygon-labeling, YOLO-seg export/training, and motion-only replay POC in `src/detection/violetta_yolo_poc.py`.
- Added unit tests for crossing identity association, missing detections, and temporal dataset export.
- No frames have been labeled, no model has been trained, and no tracking result has been validated yet.
- This Python environment currently has no PyTorch or Ultralytics installed. The POC intentionally does not alter the project's existing dependencies or install a large ML runtime.

## Local workflow

From the repository root in PowerShell:

```powershell
.\venv\Scripts\python.exe -m src.detection.violetta_yolo_poc prepare --video "C:\Users\sumee\Videos\violetta-sol.mp4" --dataset "assets\violetta_training" --interval 0.3 --end-seconds 10
.\venv\Scripts\python.exe -m src.detection.violetta_yolo_poc label --dataset "assets\violetta_training"
.\venv\Scripts\python.exe -m src.detection.violetta_yolo_poc export --dataset "assets\violetta_training"
```

In the labeler, outline **every visible copy** with a polygon. Press Enter to close each copy's polygon; repeat for the other visible copies; press D to save the frame. Partially occluded copies should be labeled only over their visible pixels. The sampler stops at 10 seconds to exclude the later answer reveal; do not extend into frames that show which face/eyes reveal the answer. Include examples from the clear reveal and shuffled motion, scale changes, rotations, transparency, effects, and overlaps. A 0.3-second sample interval is a starting point, not a required final dataset size. Review hard transitions and add extra samples if the regular sampling misses them.

Train after labeling and exporting:

```powershell
py -3.11 -m venv .venv-violetta
.\.venv-violetta\Scripts\python.exe -m pip install ultralytics
.\.venv-violetta\Scripts\python.exe -m src.detection.violetta_yolo_poc train --dataset "assets\violetta_training\dataset.yaml" --epochs 30
```

Use the isolated environment so the computer-vision dependency stack does not alter Auto Maple's runtime. Training is explicitly CPU-only for this first feasibility run; installing PyTorch/Ultralytics and training may take time and disk space. Neither has been done yet. If installation fails or CPU training is impractical, stop before changing system/GPU drivers and reassess the runtime.

Replay the trained model:

```powershell
.\.venv-violetta\Scripts\python.exe -m src.detection.violetta_yolo_poc track --model "runs\violetta\violetta\weights\best.pt" --video "C:\Users\sumee\Videos\violetta-sol.mp4" --target-slot 2 --output "violetta-yolo-poc.mp4" --metrics "violetta-yolo-poc.csv"
```

The tracker initializes IDs in left-to-right order, assigns target ID 2 from the opening lineup, and never uses post-reveal eye/face evidence. The first implementation requires four visible masks to make a new assignment; it displays no target box when detections are missing or an assignment is ambiguous. It does not claim to resolve a fully hidden identity.

## Validation and handoff

1. Run the existing Violetta POC tests and the new YOLO POC tests; verify the new command-line help works without optional model packages.
2. Check annotated sample frames and exported labels before training. Every visible copy should be represented; do not label occluded pixels as visible target.
3. Keep train/validation frames separated by alternating three-second temporal blocks. Do not randomly split neighboring frames.
4. Evaluate detection coverage (frames with four visible masks), target identity continuity, unknown intervals, and any ID swaps in the replay/CSV. A final revealed face is not valid tracking evidence.
5. If CPU model execution is too slow or the detector underperforms, the next handoff is a runtime/model decision—not more unverified tracker tuning. Candidate experiments are AMD-supported Linux/ROCm, a local lightweight point tracker, or a separately approved cloud run.
