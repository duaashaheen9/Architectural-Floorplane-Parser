# Architectural Floorplan Parser (Phase 1: YOLO + Segmentation)

Phase 1 of a two-part team graduation project. It converts a floor-plan image into structured room data (JSON).
Phase 2, [Transformer-Driven-Network-Architect](https://github.com/duaashaheen9/Transformer-Driven-Network-Architect), uses that JSON to design a wired network with VLAN zoning.

## What it does

- **Input:** a floor-plan image.
- **Output:** `rooms_data.json` with, for each room: center, four corner points, area in pixels, a transparent-background crop, and a Transformer feature vector. Also a labeled overlay image.
- **Validity check:** if YOLO finds no room (class 1) at confidence 0.3, the image is rejected (`is_valid_blueprint: false`).

## Results

| Model | Task | Data | Metric | Result |
|---|---|---|---|---|
| ResNet50 + Transformer segmentation | Binary room mask | CubiCasa5K `high_quality_architectural` subset: 3,217 train / 245 validation plans | Pixel-level IoU, threshold 0.5, 512x512 | **about 85%** |
| YOLO (trained by a teammate) | Room detection | CubiCasa5K | mAP@0.5 | TODO |

- IoU is computed over all pixels of the validation set (total intersection / total union), not averaged per image.
- Source: the training-log screenshots in `results/`. Validation IoU was 0.853-0.855 over epochs 33-39, after which early stopping (patience 10) ended training and restored the best weights. The full log was lost, so the exact value of the restored weights is not recorded.
- The validation split also chose the best weights, so this is a validation score, not a test score.

## Method

1. **Masks.** The COCO annotations used (`train_coco_pt.json`, `val_coco_pt.json`) contain a single category, `room`. Each plan's mask is the union of its room annotations. A room without a polygon is filled from its bounding box.
2. **Negative samples.** 2,000 natural images from COCO 2017 (`val2017`) with all-zero masks, so the segmentation model learns to predict nothing on images that are not floor plans. Each batch has 4 images: 2 floor plans and 2 negatives. An epoch is 1,000 steps.
3. **Segmentation model.** ResNet50 encoder (ImageNet-initialized) -> 1x1 conv -> 16x16 token sequence -> 4 Transformer blocks -> decoder with three skip connections -> sigmoid mask. TODO: state whether the backbone was frozen throughout.
4. **Training.** Kaggle notebook, Tesla P100 16 GB. Loss: 0.5 x binary cross-entropy + 0.5 x Dice. Adam, flips, 90-degree rotations, brightness/contrast jitter. Stage 1 (`training/01_train_segmentation.py`): up to 30 epochs at lr 1e-4, then up to epoch 100 at lr 1e-5. Stage 2 (`training/02_finetune_segmentation.py`): continued from the stage-1 weights, 20 epochs at lr 1e-4, then lr 1e-5 up to epoch 50 with ReduceLROnPlateau. Early stopping on validation loss stopped stage 2 at epoch 39.
5. **YOLO.** A separate room detector trained on CubiCasa5K by a teammate. TODO: label names (class 1 = ?), epochs, image size.
6. **Inference (`predict_floorplan_combined`).** YOLO boxes -> segmentation probability map -> per box: crop with 5 px padding, Gaussian blur, Otsu threshold, morphological close -> largest contour -> corners (4-point polygon, or minimum-area rectangle) -> center, area, crop, and the Transformer feature at the room center -> JSON and overlay.

## Limitations

- The IoU measures the binary room-vs-background mask. It is not a per-room detection accuracy.
- Each epoch uses 2,000 of the 3,217 training plans, because the training stream stops when the 2,000 negative images run out.
- Ground-truth masks mix polygons and bounding boxes. Stage-1 and stage-2 masks were built differently, so their IoU values are not comparable.
- Corner points are a 4-point approximation, so non-rectangular rooms are simplified (the full contour is used for the overlay, area and crop).
- Trained and evaluated on CubiCasa5K only; other drawing styles are untested.
- Model weights are not uploaded yet (slow connection). Available on request.

## Run

Requirements: Python 3.8+, TensorFlow 2.x, Ultralytics, OpenCV, NumPy. `pycocotools` is needed for training.

```
python main.py
```

TODO: if there is a Flask API (`/predict`, `/health`), document it here. Add the YOLO training script and the two log screenshots (`results/`).
