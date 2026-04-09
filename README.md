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
| 2026-04-09 | [full-body-filter-3way-sort](sessions/2026-04-09_full-body-filter-3way-sort.md) | YOLOv8-pose filter sorts all crops into kept/partial/remove; 1,925 full-body images ready for PPE labelling |

---

## Dataset

| Item | Detail |
|---|---|
| Type | Person Detection / Crop Extraction |
| Source videos | 4 MP4 files across 2 datasets (Advanced-PPE, Basic-PPE) |
| Total crops extracted | 1,956 (Advanced) + 1,600 (Basic) = **3,556 unique person crops** |
| Full-body crops (kept) | 968 (Advanced) + 957 (Basic) = **1,925 head-to-toe crops** |
| Partial crops | 950 (Advanced) + 496 (Basic) = 1,446 |
| No-person crops | 38 (Advanced) + 147 (Basic) = 185 |
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
| Status | Inference complete — 1,925 full-body crops ready for PPE labelling |
| Weights | `yolov8n.pt` (6.25 MB, ultralytics) |
| Pose filter | `yolov8n-pose.pt` (6.52 MB) — ankle keypoint visibility @ conf ≥ 0.20 |

---

## Output Structure

| Folder / File | Contents |
|---|---|
| `output/persons-advance-ppe/kept/` | **968** full-body crops — Advanced PPE |
| `output/persons-advance-ppe/partial/` | 950 partial crops — Advanced PPE |
| `output/persons-advance-ppe/remove/` | 38 no-person crops — Advanced PPE |
| `output/persons-basic-ppe/kept/` | **957** full-body crops — Basic PPE |
| `output/persons-basic-ppe/partial/` | 496 partial crops — Basic PPE |
| `output/persons-basic-ppe/remove/` | 147 no-person crops — Basic PPE |
| `output/detections.csv` | Per-detection log for Advanced-PPE |
| `output/persons-basic-ppe_detections.csv` | Per-detection log for Basic-PPE |

---

## Key Files

| File | Purpose |
|---|---|
| `detect_persons.py` | Main CLI pipeline — frame extraction, detection, dedup, save |
| `detect_utils.py` | Pure helpers: `phash_of_array`, `is_duplicate`, `short_id` |
| `filter_full_body.py` | YOLOv8-pose 3-way sorter — kept / partial / remove |
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

# Filter full-body crops (3-way sort: kept / partial / remove)
python filter_full_body.py --input output/persons-advance-ppe
python filter_full_body.py --input output/persons-basic-ppe

# Dry run first (preview counts, no files moved)
python filter_full_body.py --input output/persons-advance-ppe --dry-run

# Run tests
pytest test_detect_persons.py -v
```

---

## Tech Stack
- Python 3.12
- `ultralytics 8.3.129` — YOLOv8 detection + pose inference
- `opencv-python 4.11.0.86` — video decode, frame seek, image write
- `torch 2.11.0` — YOLOv8 backend
- `imagehash` — perceptual hash deduplication
- `pytest 8.3.5` — unit tests

---

## Next Steps
- [ ] Visual QA on `kept/` samples — spot-check pose model isn't misclassifying tight crops
- [ ] Run PPE attribute detection (helmet, vest, gloves) on the 1,925 full-body crops
- [ ] Investigate the 147 no-person crops in Basic PPE — may reveal upstream detection issues
- [ ] Consider `yolov8m.pt` for improved recall on distant/small persons
- [ ] Build labelling pipeline for PPE training data

---

*Session logs generated with [Claude Code](https://claude.ai/code)*
