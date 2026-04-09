# Session Log — 2026-04-09
**Project:** PPE Advance — Person Crop Quality Filter
**Working directory:** `/Users/paulrydrickpuri/Documents/code/script/ppe-advance`
**Duration context:** Single session, followed on from person-detection-dedup-pipeline session

---

## Summary

The goal of this session was to improve dataset quality by filtering the 3,556 person crops
(extracted in the previous session) down to only head-to-toe (full-body) images suitable for
PPE attribute training. A YOLOv8-pose based script (`filter_full_body.py`) was written and run
on both output folders. Images are now sorted into three sub-folders per dataset: `kept/`
(full body, ankles visible), `partial/` (person visible but feet cut off), and `remove/`
(no person detected by pose model). The final usable dataset contains **968 + 957 = 1,925
full-body person crops** across the two datasets.

---

## What Was Built

### `filter_full_body.py` — 3-Way Person Crop Sorter

- **Location:** `ppe-advance/filter_full_body.py`
- Loads `yolov8n-pose.pt` and runs pose inference on each person crop PNG.
- Classifies each image into one of three categories:
  - **full body** → `kept/` — at least one ankle keypoint (COCO kp 15 = left ankle, kp 16 = right ankle) detected above confidence threshold (default `0.20`)
  - **partial** → `partial/` — person detected by the pose model but neither ankle visible above threshold
  - **no person** → `remove/` — pose model detects no person bounding box at all
- Scans the root folder **and** all three sub-folders on each run, so re-running re-sorts correctly without duplication.
- Supports `--dry-run` to preview counts before moving any files.
- CLI arguments: `--input`, `--model`, `--kp-conf`, `--dry-run`.

**How ankle detection works:**
```python
# COCO keypoint indices
LEFT_ANKLE  = 15
RIGHT_ANKLE = 16

for person_kp_conf in kps.conf:   # iterate detected persons in crop
    if max(left_vis, right_vis) >= kp_conf_thresh:
        return True  # full body
```

---

## Key Decisions & Rationale

| Decision | Choice Made | Why |
|---|---|---|
| Detection method | YOLOv8-pose ankle keypoints | More reliable than aspect-ratio heuristics; directly checks if feet are visible |
| Keypoint confidence threshold | `0.20` (default) | Low enough to catch partially occluded ankles; user can tune with `--kp-conf` |
| Ankle keypoints only (not knees/hips) | Checked kp 15 + 16 only | Ankles are the strictest proxy for a complete head-to-toe crop |
| Re-scan all sub-folders on re-run | Collect from root + kept/ + partial/ + remove/ | Allows re-sorting after threshold changes without data loss |
| 3-way split (kept/partial/remove) | Added `remove/` for no-person crops | Cleaner than leaving un-classifiable crops mixed in with partials |
| Dry run first | Ran `--dry-run` before committing | Previewed split ratios to validate threshold before moving 3,556 files |
| Parallel execution | Both datasets processed with bash `&` + `wait` | Halved total runtime |

---

## Findings & Observations

1. **Advance dataset split (1,956 images):** 968 full body (49.5%), 950 partial (48.6%), 38 no-person (1.9%).
2. **Basic dataset split (1,600 images):** 957 full body (59.8%), 496 partial (31.0%), 147 no-person (9.2%).
3. The Basic PPE dataset has a much higher no-person rate (147 vs 38), suggesting more crops from that dataset were degenerate or too blurry for the pose model to detect a person.
4. The Advance dataset has a higher partial rate (~49%) — likely because more of those videos contain close-up or waist-up camera angles.
5. A dry run on the Advance folder perfectly predicted the final counts, confirming the script is deterministic.
6. `yolov8n-pose.pt` (6.52 MB) was auto-downloaded from Ultralytics assets on first use.
7. Final usable (full-body) pool: **1,925 images** — a 45.9% yield from the original 3,556 crops.
8. Both parallel runs completed without errors; no file conflicts since each process owned its own folder.

---

## Problems & Fixes

| Problem | Root Cause | Fix Applied |
|---|---|---|
| Initial script only had 2-way sort (kept / partial) | First version didn't handle no-person crops | Rewrote script with 3-way sort: kept / partial / remove |
| `partial/` sub-folder already existed from first run | First version of script moved partial images before the 3-way version existed | Updated script scans all sub-folders on entry so existing partials are re-classified correctly |
| No-person crops were left in root on first pass | First version skipped no-person images | Added `remove/` destination; no-person crops now explicitly moved |

---

## ML / Training Config (if applicable)

**Pose model used for filtering (not training):**
```
Model     : yolov8n-pose.pt
KP conf   : 0.20  (ankle keypoint visibility threshold)
Task      : Pose estimation (inference only, no training)
```

**Recommended next threshold tuning:**
- `--kp-conf 0.15` — looser, keeps more images in `kept/` (may include some with barely-visible feet)
- `--kp-conf 0.30` — stricter, smaller but cleaner `kept/` set

---

## Files Created / Modified

| File | Location | Purpose |
|---|---|---|
| `filter_full_body.py` | `ppe-advance/filter_full_body.py` | YOLOv8-pose 3-way image sorter (created this session) |
| `output/persons-advance-ppe/kept/` | 968 PNG files | Full-body crops — Advance PPE dataset |
| `output/persons-advance-ppe/partial/` | 950 PNG files | Partial crops (no ankles) — Advance PPE |
| `output/persons-advance-ppe/remove/` | 38 PNG files | No-person crops — Advance PPE |
| `output/persons-basic-ppe/kept/` | 957 PNG files | Full-body crops — Basic PPE dataset |
| `output/persons-basic-ppe/partial/` | 496 PNG files | Partial crops — Basic PPE |
| `output/persons-basic-ppe/remove/` | 147 PNG files | No-person crops — Basic PPE |

---

## Next Steps / Open Questions

- [ ] Visual QA on `kept/` samples — spot-check that pose model isn't misclassifying tight crops as full body
- [ ] Consider re-running with `--kp-conf 0.25` to tighten the full-body definition
- [ ] Move to PPE attribute labelling (helmet, vest, gloves) on the 1,925 `kept/` images
- [ ] Investigate the 147 no-person crops in Basic PPE — may reveal upstream detection issues worth filtering at the `detect_persons.py` stage
- [ ] Optionally add a `--move-no-person` flag to `detect_persons.py` to skip saving crops where YOLO confidence is below a secondary threshold

---

## Session Metadata
- **Date:** 2026-04-09
- **Model:** Claude (claude-sonnet-4-6)
- **Working directory:** `/Users/paulrydrickpuri/Documents/code/script/ppe-advance`
- **Key packages used:** `ultralytics` (YOLOv8-pose), `shutil`, `pathlib`, `opencv-python`
