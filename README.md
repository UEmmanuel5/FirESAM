[![Code License: Apache-2.0](https://img.shields.io/badge/Code%20License-Apache--2.0-blue.svg)](LICENSE)
[![Code DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18598906.svg)](https://doi.org/10.5281/zenodo.18598906)
[![Dataset License: CC BY 4.0](https://img.shields.io/badge/Dataset%20License-CC%20BY%204.0-orange.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Dataset DOI](https://img.shields.io/badge/Dataset%20DOI-10.6084%2Fm9.figshare.31311118-brightgreen.svg)](https://doi.org/10.6084/m9.figshare.31311118)


# FirESAM: An Ultra-Lightweight Prompt-in-the-Loop Distillation Model for Real-Time Fire Segmentation on Edge Devices and the FirESAM Semantic Segmentation Dataset (FSSSD)

This repository contains the **FirESAM codebase**:

- **EdgeSAM-Fire** (teacher): a fire-domain adapted promptable segmenter built by fine-tuning EdgeSAM’s **prompt encoder + mask decoder** while freezing the RepViT image encoder.
- **ProLimFUNet** (student baseline): an ultra-lightweight promptable U-Net variant trained **only with ground truth**.
- **FirESAM** (student, KD): the **same ProLimFUNet architecture** trained with **prompt-in-the-loop knowledge distillation** from the EdgeSAM-Fire teacher.

> Note on naming in this repo:
> - Some scripts still use historical filenames containing `limfunet` (e.g., `train_student_limfunet_baseline.py`). In the paper and throughout this README, these correspond to **ProLimFUNet** (baseline) and **FirESAM** (KD).

---

## Repository layout

- `firesam/`
  - `data/` dataset loaders and utilities
  - `train/` training scripts (teacher + students)
  - `export/` ONNX export and INT8 PTQ scripts
  - `eval/` utilities for model evaluation and mask generation / dataset consolidation
- `tools/`
  - `benchmark_video_firesam_onnx.py` video throughput benchmarking
- `interactive_annotator.py` web UI for prompt-based re-annotation
- `FSSSD/` documenting the dataset curation pipeline

---

## Installation

### 1) Create an environment

Recommended: **Python 3.10**.

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Clone EdgeSAM

Clone the official EdgeSAM repo next to this FirESAM folder, for example:

```bash
cd ..
git clone https://github.com/chongzhou96/EdgeSAM
```

You should now have:

```text
.../
  FirESAM/
  edge_sam/
```

### 3) Torch + ONNX Runtime

Install a CUDA or CPU build of [PyTorch](https://pytorch.org/get-started/locally/) that matches your system.
For ONNX inference/benchmarking, install either:

- `onnxruntime` (CPU)
- `onnxruntime-gpu` (CUDA)

---

## Dataset preparation (Khan + Roboflow fire)

The code supports training and evaluation on **Khan**, **Roboflow**, or **Combined** (Khan ∪ Roboflow). 

### 3.1. Datasets

Please obtain datasets from their official sources and respect licenses:

* **Khan et al.** DOI: 10.1109/TITS.2022.3203868
  Link: [https://drive.google.com/drive/folders/1Xfq7zLwIwJ4vPx50G-k7j2-ofh1bj3fx](https://drive.google.com/drive/folders/1Xfq7zLwIwJ4vPx50G-k7j2-ofh1bj3fx)

* **Roboflow Fire Segmentation**
  [https://universe.roboflow.com/firesegpart1/fire-seg-part1/dataset/21](https://universe.roboflow.com/firesegpart1/fire-seg-part1/dataset/21)

* **Foggia (MIVIA) Fire dataset** DOI: 10.1109/TCSVT.2015.2392531
  [https://mivia.unisa.it/datasets/video-analysis-datasets/fire-detection-dataset/](https://mivia.unisa.it/datasets/video-analysis-datasets/fire-detection-dataset/)

* **BurnedAreaUAV (BAUAV)** DOI: [https://doi.org/10.1016/j.isprsjprs.2023.07.002](https://doi.org/10.1016/j.isprsjprs.2023.07.002)
  [https://zenodo.org/records/7944963](https://zenodo.org/records/7944963)

* **FiSmo** paper and sources
  Paper: [https://www.researchgate.net/publication/322365857](https://www.researchgate.net/publication/322365857)
  GitHub: [https://github.com/mtcazzolato/dsw2017](https://github.com/mtcazzolato/dsw2017)
  Example video (fireVid_017): [https://drive.google.com/drive/folders/1SoYViOABT_Pt-rwrU7vPrgM7ts09D9tu?usp=sharing](https://drive.google.com/drive/folders/1SoYViOABT_Pt-rwrU7vPrgM7ts09D9tu?usp=sharing)




First unify your fire datasets into a simple **image/mask + split file** format.

### 3.2. Directory layout

Create a `data/fire/` folder inside `FirESAM`:

```text
FirESAM/
  data/
    fire/
      images/
        khan_0001.jpg
        khan_0002.jpg
        ...
        rf_0001.jpg
        ...
      masks/
        khan_0001.png
        khan_0002.png
        ...
        rf_0001.png
        ...
      splits/
        train.txt
        val.txt
        test.txt
```

### 3.3. Split text files

Each split file is a plain text file with **one sample per line**:

```text
relative/path/to/image.jpg relative/path/to/mask.png
```

Relative paths are relative to `FirESAM/data/fire/`.

Example `data/fire/splits/train.txt`:

```text
images/khan_0001.jpg masks/khan_0001.png
images/khan_0002.jpg masks/khan_0002.png
images/rf_0001.jpg   masks/rf_0001.png
...
```

Similarly create `val.txt` and `test.txt` for validation and final testing.

---

## Method overview

### ProLimFUNet and FirESAM input interface

Both the baseline student and KD student use the same promptable interface: a **6-channel** input.

- Channels 1–3: RGB image
- Channels 4–6: rasterized prompt map
  - box channel (filled rectangle)
  - positive points channel
  - negative points channel

### Distillation (prompt-in-the-loop)

FirESAM training runs a first pass with GT-derived prompts (box + points), then samples additional “hard” points from student error regions and runs a second pass. The final loss combines segmentation supervision + KD + loop loss.

---

## Training

All main training runs in the paper use **50 epochs** (teacher + students). Some exploratory runs may use more epochs; treat 50 as the default.

### 1) Train the teacher (EdgeSAM-Fire)

Script: `firesam/train/train_teacher_edgesam_fire.py`

Key paper-aligned settings:
- **Epochs:** 50
- **Batch size:** 1
- **Learning rate:** `1e-4`
- **Effective batch size:** 1 (teacher training loop processes one image per step)
- **Trainable:** prompt encoder + mask decoder (image encoder frozen)

Example:

```bash
python -m firesam.train.train_teacher_edgesam_fire \
  --cfg /path/to/edgesam_config.yaml \
  --checkpoint /path/to/edgesam_pretrained.pth \
  --train_split /path/to/DATASET_ROOT/splits/train.txt \
  --val_split /path/to/DATASET_ROOT/splits/val.txt \
  --output checkpoints/teacher_edgesam_fire \
  --epochs 50 \
  --lr 1e-4
```

> The `--cfg` file is an EdgeSAM YAML config from the EdgeSAM repo (RepViT-based). See EdgeSAM docs for available configs.

The best teacher checkpoint will be saved as (for example):

```text
checkpoints/teacher_edgesam_fire/best_teacher.pth
```

This checkpoint is then used for distillation.

---

To evaluate:

```bash
python -m firesam.eval.eval_teacher_vs_edgesam \
  --cfg /path/to/edgesam_config.yaml \
  --teacher_ckpt checkpoints/teacher_edgesam_fire/best_teacher.pth \
  --edgesam_ckpt /path/to/edgesam_pretrained.pth \
  --test_split /path/to/DATASET_ROOT/splits/train.txt \
  --output eval/teacher_vs_edgesam
  --threshold 0.5
```
Optionally include roc points (we used 200000 points due to memory constraint):

```bash
  --max_roc_points <roc points>
```

---

### 2) Train the student baseline (ProLimFUNet)

Script: `firesam/train/train_student_limfunet_baseline.py`

This script:
- Uses the same dataset (`FireSegmentationDataset`).
- Generates box + point prompts from GT masks.
- Trains the student purely with supervised loss:

  - `L_seg = Dice + BCEWithLogits` on the fire mask.
  - Optional boundary loss (enabled via arguments).


Paper-aligned defaults:
- **Epochs:** 50
- **Batch size:** 8 (baseline)
- **Learning rate:** `1e-4`

```bash
python -m firesam.train.train_student_limfunet_baseline \
  --train_split /path/to/DATASET_ROOT/splits/train.txt \
  --val_split /path/to/DATASET_ROOT/splits/val.txt \
  --output checkpoints/student_prolimfunet_baseline \
  --epochs 50 \
  --batch_size 8 \
  --lr 1e-4
```

This gives the **“before KD”** ProLimFUNet checkpoint, e.g.:

```text
checkpoints/student_baseline/best_student_baseline.pth
```

To evaluate:
Script: `firesam/eval/eval_student.py`

Evaluates a ProLimFUNet checkpoint on a given split.

```bash
python -m firesam.eval.eval_student \
  --checkpoint checkpoints/student_baseline/best_student_baseline.pth \
  --split /path/to/DATASET_ROOT/splits/test.txt \
  --batch_size 8
```
---
### 3) Train the KD student (FirESAM)

Script: `firesam/train/train_student_firesam_limfunet_kd.py`

This script:

- Loads the **fixed** EdgeSAM-Fire teacher checkpoint.
- Freezes all teacher parameters.
- Runs both teacher and student with the same image and prompts.
- Computes a combined loss:

  - `L_seg` (same as baseline).
  - `L_KD` (KL divergence or MSE between teacher and student mask logits).
  - `L_bdry` (boundary-aware Dice on edge maps).
  - `L_loop` (optional; from a second pass with prompts sampled from disagreement regions).

Total loss:

```text
L = λ_seg * L_seg + λ_KD * L_KD + λ_bdry * L_bdry + λ_loop * L_loop
```

Paper-aligned defaults:
- **Epochs:** 50
- **Batch size:** 4 (KD)
- **Learning rate:** `1e-4`
- Loss weights (default used in the paper unless noted otherwise):
  - `lambda_seg = 1.0`
  - `lambda_kd  = 0.5`
  - `lambda_bdry = 0.1`
  - `lambda_loop = 0.5`


```bash
python -m firesam.train.train_student_firesam_limfunet_kd \
  --teacher_cfg /path/to/edgesam_config.yaml \
  --train_split /path/to/DATASET_ROOT/splits/train.txt \
  --val_split /path/to/DATASET_ROOT/splits/val.txt \
  --teacher_checkpoint checkpoints/teacher_edgesam_fire/best_model \
  --output checkpoints/student_firesam_kd.pth \
  --epochs 50 \
  --batch_size 4 \
  --lr 1e-4 \
  --lambda_seg 1.0 \
  --lambda_kd 0.5 \
  --lambda_bdry 0.1 \
  --lambda_loop 0.5
```
This produces the **“after KD”** FirESAM checkpoint, e.g.:

```text
checkpoints/student_kd/best_student_kd.pth
```


To evaluate:
Script: `firesam/eval/eval_student.py`

Evaluates a FirESAM checkpoint on a given split.

```bash
python -m firesam.eval.eval_student \
  --checkpoint checkpoints/student_kd/best_student_kd.pth \
  --split /path/to/DATASET_ROOT/splits/test.txt \
  --batch_size 8
```
---

## Evaluate a prompt-rasterized student under detector prompts (YOLO)

Script: `firesam/eval/eval_yolo_prompted_student.py`

Evaluates **ProLimFUNet** (baseline) or **FirESAM** (KD) when prompts come from a **detector**, matching deployment behavior.

**Important:** This evaluates the **segmenter under detector prompts**, not detector quality. If YOLO misses a fire region entirely, the prompt map will be empty and the segmenter cannot recover it (deployment-faithful).

### Usage

Baseline (YOLO + ProLimFUNet):
```bash
python -m firesam.eval.eval_yolo_prompted_student \
  --split_file /path/to/DATASET_ROOT/splits/test.txt \
  --student_ckpt checkpoints/to/students/student_baseline.pth \
  --yolo_weights yolo/Fire_best.pt \
  --yolo_class 0 \
  --conf 0.3 \
  --img_h 416 --img_w 608
````

KD student (YOLO + FirESAM):

```bash
python -m firesam.eval.eval_yolo_prompted_student \
  --split_file /path/to/DATASET_ROOT/splits/test.txt \
  --student_ckpt checkpoints/to/students/student_kd.pth \
  --yolo_weights yolo/Fire_best.pt \
  --yolo_class 0 \
  --conf 0.3 \
  --img_h 416 --img_w 608
```

---


## Stress-test prompt robustness (loose boxes + injected false-positive boxes)

Script: `firesam/eval/eval_prompt_stress.py`

Creates detector-like prompt errors and evaluates robustness across:
- **EdgeSAM-Fire** (teacher)
- **ProLimFUNet** (baseline student)
- **FirESAM** (KD student)

It sweeps two noise sources:
1. **Box looseness** `ℓ`: expands (or shrinks) the GT-derived prompt box by a margin proportional to box size.
2. **Injected false-positive (FP) boxes** `k`: adds extra prompt boxes with low IoU to the GT box to emulate spurious detections.

### Usage examples

1) Loosen-only sweep (box-only prompts):
```bash
python -m firesam.eval.eval_prompt_stress \
  --split /path/to/DATASET_ROOT/splits/test.txt \
  --student_baseline_ckpt checkpoints/.../best_student_baseline.pth \
  --student_kd_ckpt checkpoints/.../best_student_kd.pth \
  --teacher_cfg ../edge_sam/...yaml \
  --teacher_ckpt checkpoints/.../best_teacher.pth \
  --loosen_levels 0 0.25 0.50 1.00 \
  --out_csv runs/prompt_stress_loosen.csv
```

2. FP-only sweep (no loosening):

```bash
python -m firesam.eval.eval_prompt_stress \
  --split /path/to/DATASET_ROOT/splits/test.txt \
  --student_baseline_ckpt checkpoints/.../best_student_baseline.pth \
  --student_kd_ckpt checkpoints/.../best_student_kd.pth \
  --teacher_cfg ../edge_sam/...yaml \
  --teacher_ckpt checkpoints/.../best_teacher.pth \
  --fp_boxes_per_image 1 \
  --fp_iou_max 0.05 \
  --fp_trials 200 \
  --out_csv runs/prompt_stress_fp.csv
```

3. Full grid sweep (loosen + FP):

```bash
python -m firesam.eval.eval_prompt_stress \
  --split /path/to/DATASET_ROOT/splits/test.txt \
  --student_baseline_ckpt checkpoints/.../best_student_baseline.pth \
  --student_kd_ckpt checkpoints/.../best_student_kd.pth \
  --teacher_cfg ../edge_sam/...yaml \
  --teacher_ckpt checkpoints/.../best_teacher.pth \
  --loosen_levels 0 0.25 0.50 \
  --fp_boxes_per_image 2 \
  --fp_iou_max 0.05 \
  --fp_trials 300 \
  --out_csv runs/prompt_stress_both.csv
```

4. Add point prompts + point perturbation:

```bash
python -m firesam.eval.eval_prompt_stress \
  --split /path/to/DATASET_ROOT/splits/test.txt \
  --student_baseline_ckpt checkpoints/.../best_student_baseline.pth \
  --student_kd_ckpt checkpoints/.../best_student_kd.pth \
  --teacher_cfg ../edge_sam/...yaml \
  --teacher_ckpt checkpoints/.../best_teacher.pth \
  --loosen_levels 0 0.50 \
  --fp_boxes_per_image 1 \
  --use_points --num_pos 2 --num_neg 2 --point_noise_px 5
```

### Outputs
* Saves a **CSV** via `--out_csv`.

---

## ONNX export and benchmarking

### Export

- FP32: `firesam/export/export_student_onnx32.py`
- FP16: `firesam/export/export_student_onnx16.py`
- INT8 PTQ: `firesam/export/export_student_onnx_int8.py`

Example (FP32):

```bash
#For onnx32
python -m firesam.export.export_student_onnx32 \
  --checkpoint /path/to/student/checkpoint \
  --output /path/to/output/student_firesam_fp32.onnx \
  --height 416 \
  --width 608


#For onnx16
python -m firesam.export.export_student_onnx16 \
  --checkpoint /path/to/student/checkpoint \
  --output /path/to/output/student_firesam_fp16.onnx \
  --height 416 \
  --width 608


#For int8
python -m firesam.export.quantize_student_int8 \
  --input  /path/to/onnx32/input \
  --output /path/to/output/student_firesam_int8.onnx \
  --calib_split /path/to/DATASET_ROOT/splits/val.txt \
  --num_calib 200 \
  --height 416 \
  --width 608
```

### Benchmark on video

Script: `tools/benchmark_video_firesam_onnx.py`

This runs ONNX inference on a video and reports timing/FPS on CPU and GPU.

```bash
#For onnx-only
python -m tools.benchmark_video_firesam_onnx \
  --onnx /path/to/onnx \
  --video /path/to/video.mp4 \
  --mode onnx \
  --provider cuda \ #or cpu
  --max_frames 500


#For complete pipline
python -m tools.benchmark_video_firesam_onnx \
  --onnx /path/to/onnx \
  --video /path/to/video.mp4 \
  --mode pipeline \
  --yolo path/to/yolo/Fire_best.pt \
  --provider cuda \ #or cpu
  --max_frames 500

```
The pretrained YOLO model is gotten from [ProFSAM](https://github.com/UEmmanuel5/ProFSAM).

---

## FSSSD dataset creation (pipeline + code pointers)
See the `README.md` file in the `/FSSSD` folder 

## Citation

If you use this repo, please cite the FirESAM paper.

**Manuscript:**

```
@article{Ugwu2026FirESAM,
  title={FirESAM: An Ultra-Lightweight Prompt-in-the-Loop Distillation Model for Real-Time Fire Segmentation on Edge Devices and the FirESAM Semantic Segmentation Dataset (FSSSD)},
  author={Ugwu, Emmanuel U. and Zhang, Xinming and Ouedraogo, Ezekiel B. and Aprilica Liemong, Caezar Al Fajr N. and Sukianto, Aurelia and Huang, Sicheng},
  journal={},
  year={2026}
}

```

**Code (for this repo):**

```
@software{Ugwu2026FirESAM_Code,
  author       = {Ugwu, Emmanuel U. and Zhang, Xinming and Ouedraogo, Ezekiel B. and Aprilica Liemong, Caezar Al Fajr N. and Sukianto, Aurelia and Huang, Sicheng},
  title        = {FirESAM},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18598906},
  url          = {https://doi.org/10.5281/zenodo.18598906}
}
```

**Dataset:**

```
@dataset{Ugwu2026FSSSD,
  author       = {Ugwu, Emmanuel U. and Zhang, Xinming and Ouedraogo, Ezekiel B. and Aprilica Liemong, Caezar Al Fajr N. and Sukianto, Aurelia and Huang, Sicheng},
  title        = {FSSSD (F3SD): FirESAM Semantic Segmentation Dataset},
  year         = {2026},
  publisher    = {figshare},
  doi          = {10.6084/m9.figshare.31311118},
  url          = {https://doi.org/10.6084/m9.figshare.31311118},
  note         = {Dataset archive (ZIP) and README describing folder structure.}
}
```

---

## License

This work is released under **Apache-2.0** [LICENSE](LICENSE).

