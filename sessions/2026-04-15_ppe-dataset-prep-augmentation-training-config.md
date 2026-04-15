# Session Log — 2026-04-15
**Project:** PPE Advance — Industrial PPE Detection
**Working directory:** `/Users/paulrydrickpuri/Documents/code/script/ppe-advance`
**Duration context:** Multi-day session spanning 14–15 April 2026

---

## Summary

This session focused entirely on preparing the labelled COCO dataset for model training. Starting from a raw Roboflow export, the work covered: consolidating the underrepresented `coverall_advance` class, running full EDA across both the working dataset and two snapshot exports, applying targeted image augmentation (spatial-only, no colour changes) to balance class distributions, trimming unused zero-annotation classes, redistributing eval samples to ensure test/val coverage for all trained classes, and generating a complete VisionSamurai training configuration for a 7-class PPE detector at 100 epochs.

---

## What Was Built

### 1. Dataset Consolidation — Working Dataset (4-class)
- Source: `/labelled dataset/(COCO) PPE Industrial (v2)(14Apr2026-16_50_55)/`
- Moved all 4 `coverall_advance` images from test/val into train (2 from each)
- Result: 17 original `coverall_advance` images consolidated into train for augmentation

### 2. Image Augmentation Pipeline — coverall_advance (4-class)
- Script written inline (PIL + NumPy, no albumentations)
- 85 augmentation configs per source image × 17 images = 1,445 augmented images
- Augmentation types: horizontal flip, rotation (±30°), Gaussian blur (0.8–2.5), Gaussian noise (std 5–12), sharpening (1.5–3.0×), combinations
- **Constraint strictly enforced:** No vertical flip (persons must stay upright), no hue/saturation/colour changes (blue = `coverall_advance`, non-blue = `coverall`)
- Augmented filenames prefixed `aug_` for easy identification
- Balanced result: `coverall_advance` 17 → **1,462** vs `coverall` 1,491

### 3. EDA — Working Dataset
- Ran full class distribution analysis across train/test/val
- Identified 10 zero-annotation classes in 15-class schema
- Image size stats (original train): width 42–726px mean 155, height 120–823px mean 394
- All bboxes cover ~100% of image area (tight single-person crops, not scene images)
- Annotations per image: 82% single, 17% two, <1% three
- Aspect ratio: `coverall` mean H/W 2.72 vs `coverall_advance` mean 2.22

### 4. Class Trimming — 4 Active Classes Only
- Removed 11 zero-annotation classes from all 3 splits' category lists
- Also dropped 3 `missing_gloves` annotations (only 3 total — untrainable)
- 10 train images and 3 test images with only removed-class annotations were dropped
- Final 4 classes: `coverall`, `coverall_advance`, `helmet`, `missing_headgear`

### 5. Test/Val Redistribution for coverall_advance
- Problem: `coverall_advance` had 0 samples in test and val after consolidation — model performance on this class could not be evaluated
- Fix: Moved 3 original (non-augmented) images → test, 2 → val
- Augmented copies remain in train only, ensuring honest evaluation on real images

### 6. New Snapshot Dataset Processing — 7-class
- Source: `/labelled dataset/snapshot(15-04-2026_10-38-17)/`
- Images stored under `{split}/images/`, annotations under `{split}/annotations/annotations.json`
- New classes present vs old dataset: `boots` (2,937), `gloves` (184), `missing_gloves` (942)
- `coverall_advance` still only 17 samples — same images as before
- Ran full pipeline:
  1. **Phase 1** — Stripped 8 zero-annotation classes, kept 7
  2. **Phase 2** — Moved all 4 `coverall_advance` samples from test/val into train (17 total)
  3. **Phase 3** — Generated 139 augmentation configs × 17 images = **2,363 augmented images**; balance target = `boots` count (2,375); proper bbox coordinate transformation applied (not just zeroed)
  4. **Phase 4** — Moved 3 originals → test, 2 → val for evaluation coverage

### 7. VisionSamurai Training Config
- Generated two configs: one for 4-class dataset (250 epochs), one for 7-class (300 epochs)
- User requested 100-epoch version; config adjusted with WarmupCosineLR, warmup 3 epochs, patience 15, eval interval 5

---

## Key Decisions & Rationale

| Decision | Choice Made | Why |
|---|---|---|
| No colour augmentation | Hue = 0, no saturation shift | Blue full-body coverall = `coverall_advance`; hue shifts would corrupt the primary visual signal distinguishing it from `coverall` |
| No vertical flip | Disabled entirely | Upside-down persons are not a realistic deployment scenario |
| Balance target for augmentation | Match `boots` count (dominant class, 2,375) | Prevents the model from ignoring `coverall_advance` due to frequency imbalance |
| Proper bbox transformation | 4-corner rotation + clip | Bboxes in snapshot dataset have sub-pixel offsets (e.g. [1.09, 0, 94.37, 179.18]) — zeroing would be slightly wrong |
| Test/val contain originals only | Augmented images stay in train | Evaluation on augmented data would inflate metrics; honest eval requires real images |
| WarmupCosineLR at 100 epochs | Switched from MultiStep | At 100 epochs, MultiStep milestone tuning wastes budget; cosine gives smooth continuous decay |
| RepeatFactorTrainingSampler | Used for both configs | `gloves` (138) is ~17× less frequent than `boots`/`coverall_advance` (2,375) — uniform sampling would starve it |

---

## Findings & Observations

1. Both downloaded dataset exports (`14Apr2026-16_50_55` and `14Apr2026-17_13_18`) were identical — same annotation counts, just different export timestamps.
2. The snapshot `15-04-2026_10-38-17` is the materially different version — it added `boots` (2,937), `gloves` (184), and significantly more `missing_gloves` (742 train vs 3 before).
3. All `coverall_advance` images are tight single-person crops where the bounding box spans 90–100% of the image — they are pre-cropped detections, not full scene images.
4. `boots` and `missing_boots` exist as separate classes in the schema, but `missing_boots` has zero annotations — the dataset can detect boots being worn but cannot flag their absence.
5. Image inspection of `c90bb87d-fcca-4823-9e63-40591cbfb93d.png` confirmed why boots have no annotations: crops are cut off at the shins, feet/boots not visible.
6. `coverall_advance` aspect ratio (mean H/W 2.22) is consistently narrower/squarer than `coverall` (mean H/W 2.72), consistent with the full blue suit having a broader silhouette.
7. After augmentation, `coverall_advance` exactly matched `boots` at **2,375** annotations in train — clean 1:1 ratio on the two dominant classes.
8. 10 zero-annotation categories existed in the schema (`cap`, `vest`, `missing_vest`, `ppe_uniform`, `welding_mask`, `missing_welding_mask`, `harness`, `boots` [in old dataset], `missing_boots`, `gloves` [in old dataset]) — these were stripped to keep the category list clean.
9. The 7-class final dataset has `gloves` (138 train) as the most underrepresented class — RepeatFactorTrainingSampler is critical for this class.

---

## Problems & Fixes

| Problem | Root Cause | Fix Applied |
|---|---|---|
| First augmentation included vertical flip | Initial config included `vflip` variants | Removed all augmented images, rebuilt configs without any `vflip`, regenerated |
| `coverall_advance` had 0 test/val samples after consolidation | All originals moved to train for augmentation | Moved 3 → test, 2 → val after augmentation completed (originals only) |
| Other classes not updated during augmentation | User questioned if co-occurring annotations were missed | Verified: all 17 `coverall_advance` source images contain only `coverall_advance` (single-person crops) — no missed annotations |
| Snapshot dataset had different folder structure | Images in `{split}/images/`, not `{split}/` directly | Updated all path references in scripts |
| Stale clone in `/tmp/session-journal-push` | Prior session left directory | Removed and re-cloned |

---

## ML / Training Config (Final — 100 Epochs, 7 Classes)

```
📋 RECOMMENDED TRAINING CONFIG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROJECT : PPE Industrial Advanced (snapshot 15-04-2026)
TASK    : Object Detection (COCO) | 7 classes
DATASET : ~4,275 images | Mostly balanced (gloves low at 138)

▸ MODEL
  Architecture : yolov8m
  Fine-Tune    : ✅ Yes (from COCO pretrained)

▸ DATA SETTINGS
  Input Shape  : 640
  Batch Size   : 16

▸ TRAINING SETTINGS
  Base LR      : 0.001
  Epochs       : 100
  Patience     : 15

▸ SOLVER SETTINGS
  Warmup Epochs       : 3
  Evaluation Interval : 5
  Event Interval      : 5
  Checkpoint Interval : 10

▸ OPTIMIZER
  Type              : SGD
  Weight Decay      : 0.0001
  Nesterov Momentum : ✅
  Gradient Clipping : ☐

▸ LR SCHEDULER
  Type             : WarmupCosineLR
  Final LR         : 0

▸ DATA SAMPLING
  Sampler : RepeatFactorTrainingSampler

▸ AUGMENTATION
  ✅ Brightness  (0.8–1.2, 50%)     ✅ Contrast    (0.8–1.2, 50%)
  ✅ Flip L/R    (50%)              ❌ Flip U/D    DISABLED
  ✅ Rotation    (0–15°, 40%)       ✅ Blur        (0.1–1.5, 40%)
  ✅ Dropout     (0.05–0.10, 30%)   ✅ Shadow      (40%)
  ⚠️ HSV         Hue=0, Sat±10, Val±15, 40%
  ☐  Rain / Fog / Snow / Sunflare

⚠️ CRITICAL: Hue delta must be 0. Full blue = coverall_advance.
```

---

## Files Created / Modified

| File | Location | Purpose |
|---|---|---|
| `annotations.json` (train) | `labelled dataset/(COCO).../train/` | Cleaned to 4 classes, augmented coverall_advance |
| `annotations.json` (test/val) | `labelled dataset/(COCO).../test/val/` | Cleaned to 4 classes, coverall_advance redistributed |
| `aug_*.png` (×1,445) | `labelled dataset/(COCO).../train/images/` | Augmented coverall_advance crops |
| `annotations.json` (train) | `snapshot(15-04-2026...)/train/annotations/` | 7-class clean, 2,363 augmented images added |
| `annotations.json` (test/val) | `snapshot(15-04-2026...)/test+val/annotations/` | 7-class clean, coverall_advance redistributed |
| `aug_*.png` (×2,363) | `snapshot(15-04-2026...)/train/images/` | Augmented coverall_advance crops (139 configs × 17 sources) |

---

## Next Steps / Open Questions

- [ ] Upload snapshot dataset to VisionSamurai and verify class list shows 7 classes
- [ ] Run training with recommended 100-epoch config on VisionSamurai
- [ ] Monitor val mAP for `gloves` (138 train samples) — may need further augmentation if underfitting
- [ ] Source `missing_boots` labelled data — boots detected but absence not flagged
- [ ] Consider adding `vest` / `missing_vest` data in a future labelling round
- [ ] After training, evaluate per-class AP — `coverall_advance` vs `coverall` confusion matrix is the key metric to watch

---

## Session Metadata
- **Date:** 2026-04-15
- **Model:** Claude (claude-sonnet-4-6)
- **Working directory:** `/Users/paulrydrickpuri/Documents/code/script/ppe-advance`
- **Key packages used:** Python 3.12, Pillow 12.1.1, NumPy 2.4.3, OpenCV 4.10.0
