[![Code License: Apache-2.0](https://img.shields.io/badge/Code%20License-Apache--2.0-blue.svg)](LICENSE)
[![Code DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18598906.svg)](https://doi.org/10.5281/zenodo.18598906)
[![Dataset License: CC BY 4.0](https://img.shields.io/badge/Dataset%20License-CC%20BY%204.0-orange.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Dataset DOI](https://img.shields.io/badge/Dataset%20DOI-10.6084%2Fm9.figshare.31311118-brightgreen.svg)](https://doi.org/10.6084/m9.figshare.31311118)

# FirESAM: An Ultra-Lightweight Prompt-in-the-Loop Distillation Model for Real-Time Fire Segmentation on Edge Devices and the FirESAM Semantic Segmentation Dataset (FSSSD)

This repository provides code for **FirESAM**, an ultra-lightweight prompt-in-the-loop distillation model for fire segmentation, along with the **FirESAM Semantic Segmentation Dataset (FSSSD)**.

It contains:

- **EdgeSAM-Fire** (teacher): promptable segmenter fine-tuned on fire data.
- **ProLimFUNet** (student baseline): lightweight U-Net variant trained with ground truth only.
- **FirESAM** (student KD): ProLimFUNet trained with prompt-in-the-loop knowledge distillation from EdgeSAM-Fire.

> Script names may reference `train_student_limfunet_baseline` and `train_student_firesam_limfunet_kd`; these correspond to **ProLimFUNet** and **FirESAM** in this README.

---

## Repository Layout

```

FirESAM/
firesam/
data/           # dataset loaders and utilities
train/          # teacher and student training scripts
eval/           # evaluation scripts and mask generation
export/         # ONNX export and INT8 PTQ scripts
tools/
benchmark_video_firesam_onnx.py
interactive_annotator.py  # web-based prompt annotator
FSSSD/                    # dataset curation pipeline

````

---

## Installation

### 1) Environment

Python 3.10 recommended:

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
````

### 2) EdgeSAM

Clone EdgeSAM alongside FirESAM:

```bash
cd ..
git clone https://github.com/chongzhou96/EdgeSAM
```

Ensure directory structure:

```
.../
  FirESAM/
  edge_sam/
```

### 3) Torch + ONNX Runtime

Install a CUDA or CPU build of PyTorch compatible with your system. For ONNX inference:

```bash
pip install onnxruntime       # CPU
pip install onnxruntime-gpu   # CUDA
```

---

## Dataset Preparation

Supports **Khan**, **Roboflow**, **Foggia**, **BurnedAreaUAV**, and **FiSmo** datasets.

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



### 1) Directory Layout

```
FirESAM/data/fire/
  images/
  masks/
  splits/
    train.txt
    val.txt
    test.txt
```

Split files are plain text:

```
relative/path/to/image.jpg relative/path/to/mask.png
```
---

## Training

### 1) Teacher (EdgeSAM-Fire)

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

Evaluate:

```bash
python -m firesam.eval.eval_teacher_vs_edgesam \
  --cfg /path/to/edgesam_config.yaml \
  --teacher_ckpt checkpoints/teacher_edgesam_fire/best_teacher.pth \
  --edgesam_ckpt /path/to/edgesam_pretrained.pth \
  --test_split /path/to/DATASET_ROOT/splits/test.txt \
  --output eval/teacher_vs_edgesam \
  --threshold 0.5 \
  --max_roc_points 200000
```

### 2) Student Baseline (ProLimFUNet)

Train:

```bash
python -m firesam.train.train_student_limfunet_baseline \
  --train_split /path/to/DATASET_ROOT/splits/train.txt \
  --val_split /path/to/DATASET_ROOT/splits/val.txt \
  --output checkpoints/student_prolimfunet_baseline \
  --epochs 50 \
  --batch_size 8 \
  --lr 1e-4
```

Evaluate:

```bash
python -m firesam.eval.eval_student \
  --checkpoint checkpoints/student_baseline/best_student_baseline.pth \
  --split /path/to/DATASET_ROOT/splits/test.txt \
  --batch_size 8
```

### 3) KD Student (FirESAM)

Train:

```bash
python -m firesam.train.train_student_firesam_limfunet_kd \
  --teacher_cfg /path/to/edgesam_config.yaml \
  --train_split /path/to/DATASET_ROOT/splits/train.txt \
  --val_split /path/to/DATASET_ROOT/splits/val.txt \
  --teacher_checkpoint checkpoints/teacher_edgesam_fire/best_teacher.pth \
  --output checkpoints/student_firesam_kd.pth \
  --epochs 50 \
  --batch_size 4 \
  --lr 1e-4 \
  --lambda_seg 1.0 \
  --lambda_kd 0.5 \
  --lambda_bdry 0.1 \
  --lambda_loop 0.5
```

Evaluate:

```bash
python -m firesam.eval.eval_student \
  --checkpoint checkpoints/student_kd/best_student_kd.pth \
  --split /path/to/DATASET_ROOT/splits/test.txt \
  --batch_size 8
```

---

## YOLO-Prompted Evaluation

```bash
python -m firesam.eval.eval_yolo_prompted_student \
  --split_file /path/to/DATASET_ROOT/splits/test.txt \
  --student_ckpt checkpoints/student_kd.pth \
  --yolo_weights yolo/Fire_best.pt \
  --yolo_class 0 \
  --conf 0.3 \
  --img_h 416 --img_w 608
```

---

## Prompt Stress Testing

```bash
python -m firesam.eval.eval_prompt_stress \
  --split /path/to/DATASET_ROOT/splits/test.txt \
  --student_baseline_ckpt checkpoints/student_baseline.pth \
  --student_kd_ckpt checkpoints/student_kd.pth \
  --teacher_cfg ../edge_sam/config.yaml \
  --teacher_ckpt checkpoints/teacher_edgesam_fire/best_teacher.pth \
  --loosen_levels 0 0.25 0.50 \
  --fp_boxes_per_image 1 \
  --fp_iou_max 0.05 \
  --fp_trials 200 \
  --use_points --num_pos 2 --num_neg 2 --point_noise_px 5 \
  --out_csv runs/prompt_stress.csv
```

---

## ONNX Export and Benchmarking

### Export FP32 / FP16 / INT8

```bash
python -m firesam.export.export_student_onnx32 --checkpoint checkpoints/student_kd.pth --output student_fp32.onnx --height 416 --width 608
python -m firesam.export/export_student_onnx16 --checkpoint checkpoints/student_kd.pth --output student_fp16.onnx --height 416 --width 608
python -m firesam.export.quantize_student_int8 --input student_fp32.onnx --output student_int8.onnx --calib_split splits/val.txt --num_calib 200 --height 416 --width 608
```

### Benchmark Video

```bash
python -m tools.benchmark_video_firesam_onnx --onnx student_int8.onnx --video video.mp4 --mode onnx --provider cuda --max_frames 500
```

---

## Dataset Pipeline

See `FSSSD/README.md` for dataset creation instructions.

---

## Citation

**Paper:**

```
@article{Ugwu2026FirESAM,
  title={FirESAM: An Ultra-Lightweight Prompt-in-the-Loop Distillation Model for Real-Time Fire Segmentation on Edge Devices and the FirESAM Semantic Segmentation Dataset (FSSSD)},
  author={Ugwu, Emmanuel U. and Zhang, Xinming and Ouedraogo, Ezekiel B. and Aprilica Liemong, Caezar Al Fajr N. and Sukianto, Aurelia and Huang, Sicheng},
  journal={},
  year={2026}
}
```

**Code:**

```
@software{Ugwu2026FirESAM_Code,
  author={Ugwu, Emmanuel U. and Zhang, Xinming and Ouedraogo, Ezekiel B. and Aprilica Liemong, Caezar Al Fajr N. and Sukianto, Aurelia and Huang, Sicheng},
  title={FirESAM},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.18598906},
  url={https://doi.org/10.5281/zenodo.18598906}
}
```

**Dataset:**

```
@dataset{Ugwu2026FSSSD,
  author={Ugwu, Emmanuel U. and Zhang, Xinming and Ouedraogo, Ezekiel B. and Aprilica Liemong, Caezar Al Fajr N. and Sukianto, Aurelia and Huang, Sicheng},
  title={FSSSD (F3SD): FirESAM Semantic Segmentation Dataset},
  year={2026},
  publisher={figshare},
  doi={10.6084/m9.figshare.31311118},
  url={https://doi.org/10.6084/m9.figshare.31311118},
  note={Dataset archive (ZIP) and README describing folder structure.}
}
```

---

## License

Released under **Apache-2.0** [LICENSE](LICENSE).

