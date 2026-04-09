# Session Log — 2026-04-09
**Project:** PPE Advance — Person Detection & Deduplication Pipeline  
**Working directory:** `/Users/paulrydrickpuri/Documents/code/script/ppe-advance`  
**Duration context:** Single session, 2 dataset pipelines run end-to-end

---

## Summary

The goal of this session was to build a reusable person-detection pipeline that processes CCTV footage from two PPE datasets, extracts frames at 1 FPS, detects persons using YOLOv8, crops each detection, and saves only unique crops (deduplication via perceptual hash). The pipeline was implemented, tested with 15 unit tests (all passing), and run successfully against both the Advanced-PPE-dataset and Basic-PPE-dataset, producing 1,956 and 1,600 unique person crops respectively.

---

## What Was Built

### Person Detection Pipeline (`detect_persons.py`)
- Main CLI script that ties together frame extraction, YOLOv8 inference, pHash deduplication, and file saving
- Accepts `--dataset`, `--output`, `--video`, `--conf`, `--hash-thresh`, `--model` flags — fully reusable across any dataset folder
- Samples exactly 1 FPS by computing `step = round(video_fps)` and using `cap.set(CAP_PROP_POS_FRAMES, frame_idx)` for direct seeks (faster than sequential decode)
- Filters detections to COCO class 0 (person) only, with configurable confidence threshold (default 0.40)
- Writes per-detection CSV log alongside the output folder: `output/<folder-name>_detections.csv`
- Key file: `detect_persons.py`

### Pure Helpers Module (`detect_utils.py`)
- Extracted `phash_of_array`, `is_duplicate`, `short_id` into a separate module with no torch/ultralytics dependency
- This allowed unit tests to run cleanly without needing a working torch installation
- Key file: `detect_utils.py`

### Unit Test Suite (`test_detect_persons.py`)
- 15 pytest tests covering all three helper functions
- Tests use gradient images and checkerboard patterns (not solid colours) because pHash is DCT-based — solid-colour images all produce near-identical hashes (no AC components), making naive colour-based tests unreliable
- Key file: `test_detect_persons.py`

### Output — Advanced-PPE-dataset
- **2 videos** processed: `202604081513441_192.168.1.217_001_...mp4` and `_002_...mp4`
- Video FPS: 20 → step: 20 frames → ~1,800 samples per video
- **1,956 unique person crops** saved to `output/persons-advance-ppe/`
- Detection log: `output/detections.csv`
- Avg confidence: 0.773

### Output — Basic-PPE-dataset
- **2 videos** processed: `202604091117012_192.168.1.220_001_...mp4` and `_002_...mp4`
- Video FPS: 10 → step: 10 frames → ~1,802 samples per video
- **1,600 unique person crops** saved to `output/persons-basic-ppe/`
- Detection log: `output/persons-basic-ppe_detections.csv`
- Avg confidence: 0.695

---

## Key Decisions & Rationale

| Decision | Choice Made | Why |
|---|---|---|
| Detection model | YOLOv8n (nano) | Fast enough for batch offline processing; can swap to yolov8m for higher accuracy via `--model` flag |
| Deduplication method | pHash (perceptual hash, 64-bit DCT) via `imagehash` library | Robust to minor JPEG compression artefacts and slight brightness shifts; hamming distance threshold configurable |
| Default hash threshold | 10 (hamming distance) | Empirically balances removing repeated static-camera crops vs keeping genuinely different poses |
| Frame sampling | 1 FPS via direct `CAP_PROP_POS_FRAMES` seek | Avoids decoding every frame; correct even for variable-fps sources |
| Helper module separation | `detect_utils.py` (no torch) vs `detect_persons.py` (torch) | Allows unit tests to run even when torch dylib is broken in the test environment |
| Output naming | `{8-char md5 of video name}_t{timestamp}_d{det_idx}.png` | Stable, collision-resistant, sortable by time |
| CSV log location | `output/<output-folder-name>_detections.csv` | Keeps log co-located with but outside the images folder; separate per dataset run |

---

## Findings & Observations

1. The miniconda base environment had a broken torch installation — `libtorch_cpu.dylib` was missing from `site-packages/torch/lib/`. This prevented both the pipeline and tests from running initially.
2. Reinstalling torch via `pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cpu` resolved the missing dylib and upgraded torch from a corrupted 2.7.0 to 2.11.0.
3. The `openmmlab` conda env had torch 2.4.1 but was broken due to a missing `typing_extensions` module — not used.
4. Basic-PPE-dataset videos are 10 FPS (vs 20 FPS for Advanced-PPE), indicating different camera sources.
5. Basic-PPE had detections from frame 0 onwards (persons visible immediately), whereas Advanced-PPE had no detections for the first ~1,000 frames — likely a different scene/camera angle.
6. Basic-PPE average confidence (0.695) was lower than Advanced-PPE (0.773), suggesting more challenging detection conditions (distance, occlusion, or lower resolution).
7. Solid-colour test images produce near-identical pHashes because pHash operates on DCT coefficients of the luminance channel — all-flat images have only a DC component, which is ignored. Gradient and checkerboard patterns are necessary for meaningful hash tests.
8. `imagehash.ImageHash.__sub__` returns an `int` (hamming distance), not a boolean or array — confirmed in tests.

---

## Problems & Fixes

| Problem | Root Cause | Fix Applied |
|---|---|---|
| `ImportError: dlopen libtorch_cpu.dylib` on running pipeline and tests | Incomplete torch wheel in miniconda base — `libtorch_cpu.dylib` missing from `torch/lib/` | `pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cpu` |
| Tests collecting 0 items / import error | `test_detect_persons.py` imported from `detect_persons.py` which imported torch at module level | Extracted pure helpers into `detect_utils.py` (no torch); tests import from there instead |
| 3 test failures on solid-colour images | pHash of any solid-colour image is near-identical (no AC components in DCT) | Replaced solid-colour test images with gradient and checkerboard patterns |
| `DATASET_DIR` and `OUTPUT_DIR` hardcoded as module-level `Path` constants | No way to point pipeline at a different dataset without editing source | Added `--dataset` and `--output` CLI args; resolved to local `Path` vars inside `main()` |
| `OUTPUT_DIR` global used inside `process_video()` | Function captured module-level constant | Added `output_dir: Path` parameter to `process_video()` with the global as default |

---

## ML / Training Config

**YOLOv8 inference settings used:**
```python
model = YOLO("yolov8n.pt")
results = model(frame, verbose=False, conf=0.40, classes=[0])  # class 0 = person
```

- `conf=0.40` — balanced threshold; reduces false positives while catching partially-occluded workers
- `classes=[0]` — restricts inference to person class only, skipping all other COCO detections
- `yolov8n.pt` — nano model, ~6.25 MB, auto-downloaded from ultralytics assets on first run
- To improve recall on distant/small persons: increase to `yolov8m.pt` and lower `conf` to 0.35

**Deduplication settings:**
```python
DEFAULT_HASH_THRESH = 10   # hamming bits; 0-64 range
```
- A threshold of 10 means two crops must differ by >10 bits in their 64-bit pHash to both be saved
- Lower (e.g. 5) = stricter, saves more crops; higher (e.g. 15) = looser, more aggressive dedup

---

## Files Created / Modified

| File | Location | Purpose |
|---|---|---|
| `detect_persons.py` | `ppe-advance/` | Main CLI pipeline: frame extraction → YOLOv8 detection → pHash dedup → save crops |
| `detect_utils.py` | `ppe-advance/` | Pure helpers: `phash_of_array`, `is_duplicate`, `short_id` (no torch dependency) |
| `test_detect_persons.py` | `ppe-advance/` | 15 pytest unit tests for all helper functions |
| `output/persons-advance-ppe/*.png` | `ppe-advance/output/` | 1,956 unique person crops from Advanced-PPE-dataset |
| `output/detections.csv` | `ppe-advance/output/` | Per-detection log for Advanced-PPE run |
| `output/persons-basic-ppe/*.png` | `ppe-advance/output/` | 1,600 unique person crops from Basic-PPE-dataset |
| `output/persons-basic-ppe_detections.csv` | `ppe-advance/output/` | Per-detection log for Basic-PPE run |
| `yolov8n.pt` | `ppe-advance/` | YOLOv8 nano weights (auto-downloaded) |

---

## Next Steps / Open Questions
- [ ] Run PPE attribute detection (helmet, vest, etc.) on the saved person crops
- [ ] Verify crop quality — review sample images from both datasets visually
- [ ] Consider lowering `--hash-thresh` to 5–8 if too many near-duplicate poses are present in the crops
- [ ] Consider upgrading to `yolov8m.pt` for the final pipeline run to improve recall on distant workers
- [ ] Build a labelling pipeline on top of the extracted crops for PPE training data

---

## Session Metadata
- **Date:** 2026-04-09
- **Model:** Claude (claude-sonnet-4-6)
- **Working directory:** `/Users/paulrydrickpuri/Documents/code/script/ppe-advance`
- **Key packages used:** `ultralytics 8.3.129`, `opencv-python 4.11.0.86`, `torch 2.11.0`, `imagehash` (installed this session), `pytest 8.3.5`, `numpy 2.2.5`
