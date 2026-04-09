# PPE Advance

> A person detection and PPE (Personal Protective Equipment) analysis pipeline for CCTV footage.

---

## Project Overview

This project processes CCTV video recordings from industrial/construction sites to extract person crops and analyse PPE compliance (helmets, vests, etc.). It uses YOLOv8 for person detection and perceptual hashing for deduplication, producing clean datasets of unique person crops for downstream PPE classification or model training.

---

## Progress

| Date | Session | What Was Done |
|---|---|---|
| 2026-04-09 | [person-detection-dedup-pipeline](sessions/2026-04-09_person-detection-dedup-pipeline.md) | Built full person detection + pHash dedup pipeline; processed 4 videos (2× Advanced, 2× Basic PPE); extracted 3,556 unique person crops total |

---

## Dataset

| Item | Detail |
|---|---|
| Type | Person Detection / Crop Extraction |
| Source videos | 4 MP4 files across 2 datasets (Advanced-PPE, Basic-PPE) |
| Crops extracted | 1,956 (Advanced) + 1,600 (Basic) = **3,556 unique person crops** |
| Sampling rate | 1 FPS |
| Deduplication | pHash hamming distance threshold = 10 |
| Format | PNG crops + CSV detection log per dataset |

---

## Model

| Item | Detail |
|---|---|
| Architecture | YOLOv8n (nano) |
| Task | Person Detection (COCO class 0) |
| Confidence threshold | 0.40 |
| Status | Inference complete — crops ready for PPE labelling |
| Weights | `yolov8n.pt` (6.25 MB, ultralytics) |

---

## Output Structure

| Folder / File | Contents |
|---|---|
| `output/persons-advance-ppe/` | 1,956 unique person crops from Advanced-PPE-dataset |
| `output/persons-basic-ppe/` | 1,600 unique person crops from Basic-PPE-dataset |
| `output/detections.csv` | Per-detection log for Advanced-PPE (video, frame, timestamp, bbox, conf) |
| `output/persons-basic-ppe_detections.csv` | Per-detection log for Basic-PPE |

---

## Key Files

| File | Purpose |
|---|---|
| `detect_persons.py` | Main CLI pipeline — frame extraction, detection, dedup, save |
| `detect_utils.py` | Pure helpers: `phash_of_array`, `is_duplicate`, `short_id` |
| `test_detect_persons.py` | 15 pytest unit tests (all passing) |
| `sessions/` | Per-session progress logs |
| `PROGRESS.md` | Session index |

---

## Usage

```bash
# Process all .mp4 in a dataset folder
python detect_persons.py --dataset Advanced-PPE-dataset --output output/persons-advance-ppe

# Basic PPE dataset
python detect_persons.py --dataset Basic-PPE-dataset --output output/persons-basic-ppe

# Tune detection sensitivity
python detect_persons.py --dataset Basic-PPE-dataset --output output/persons-basic-ppe \
  --conf 0.5 --hash-thresh 5 --model yolov8m.pt

# Run tests
pytest test_detect_persons.py -v
```

---

## Tech Stack
- Python 3.12
- `ultralytics 8.3.129` — YOLOv8 inference
- `opencv-python 4.11.0.86` — video decode, frame seek, image write
- `torch 2.11.0` — YOLOv8 backend
- `imagehash` — perceptual hash deduplication
- `pytest 8.3.5` — unit tests

---

## Next Steps
- [ ] Run PPE attribute detection (helmet, vest) on saved person crops
- [ ] Visual QA — review sample crops from both datasets
- [ ] Consider `yolov8m.pt` for improved recall on distant/small persons
- [ ] Build labelling pipeline for PPE training data

---

*Session logs generated with [Claude Code](https://claude.ai/code)*
