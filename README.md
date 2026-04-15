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
| 2026-04-15 | [ppe-dataset-prep-augmentation-training-config](sessions/2026-04-15_ppe-dataset-prep-augmentation-training-config.md) | Full dataset prep: EDA, class trim to 7, coverall_advance augmentation (2,363 images), train/test/val redistribution, VisionSamurai 100-epoch training config |

---

## Dataset

| Item | Detail |
|---|---|
| Type | Object Detection (COCO format) |
| Source videos | 4 MP4 files across 2 datasets (Advanced-PPE, Basic-PPE) |
| Total crops extracted | 1,956 (Advanced) + 1,600 (Basic) = **3,556 unique person crops** |
| Full-body crops (kept) | 968 (Advanced) + 957 (Basic) = **1,925 head-to-toe crops** |
| Labelled dataset | snapshot(15-04-2026) — 4,275 images (train 3,895 / test 191 / val 189) |
| Active classes | 7 — `coverall`, `coverall_advance`, `helmet`, `missing_headgear`, `missing_gloves`, `gloves`, `boots` |
| Augmented images | 2,363 synthetic `coverall_advance` crops (spatial-only, no colour change) |
| Format | COCO JSON — `{split}/images/` + `{split}/annotations/annotations.json` |
| Sampling rate | 1 FPS |
| Deduplication | pHash hamming distance threshold = 10 |

---

## Model

| Item | Detail |
|---|---|
| Architecture | YOLOv8m (medium) |
| Task | Object Detection — 7-class PPE compliance |
| Status | Dataset ready — training not yet started |
| Platform | VisionSamurai (app.visionsamur.ai) |
| Person detector | `yolov8n.pt` — upstream crop extraction |
| Pose filter | `yolov8n-pose.pt` — ankle keypoint visibility @ conf ≥ 0.20 |

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

## Class Map

| Class | Train | Test | Val | Notes |
|---|---|---|---|---|
| `boots` | 2,375 | 278 | 284 | Safety boots worn |
| `coverall_advance` | 2,375 | 3 | 2 | Full blue coverall — colour-critical |
| `coverall` | 1,529 | 185 | 187 | Coverall worn (non-blue or missing headgear) |
| `missing_headgear` | 928 | 110 | 115 | Person not wearing headgear |
| `missing_gloves` | 742 | 109 | 91 | Person not wearing gloves |
| `helmet` | 284 | 45 | 33 | Safety helmet worn |
| `gloves` | 138 | 22 | 24 | Safety gloves worn — lowest class |

---

## Next Steps
- [ ] Upload snapshot dataset to VisionSamurai and verify 7-class list
- [ ] Run training — 100 epochs, yolov8m, WarmupCosineLR, RepeatFactorTrainingSampler
- [ ] Monitor val mAP for `gloves` (138 train) — may need further augmentation
- [ ] Source `missing_boots` labelled data — absence detection not yet possible
- [ ] Consider `vest` / `missing_vest` labelling in a future round
- [ ] After training, review `coverall_advance` vs `coverall` confusion — colour is the key signal

---

*Session logs generated with [Claude Code](https://claude.ai/code)*
