# FSSSD: FirESAM Semantic Segmentation Dataset Curation Pipeline

This document records the full pipeline used to derive the **FSSSD** from **FASDD_CV**, including deduplication, automatic mask generation, quality control, and interactive re-annotation.

The pipeline uses three main codebases:

- `FASDD_CV/` – original dataset (images + annotations).
- `EVFR` – embedding extraction and deduplication utilities.
- `FirESAM/` – FirESAM training / ONNX inference and interactive annotation tools.

---

## Step 1 — Start from FASDD_CV and audit scene labels

First verify image counts, in the `FASDD_CV/` (95,314 images in `images/`, YOLO labels in `annotations/YOLO_CV/labels`), and scene categories, then extract only fire scenes.
The FASSD dataset is gotten from [Scienc Data Bank](https://www.scidb.cn/en/detail?dataSetId=ce9c9400b44148e1b0a749f5c3eb0bda)

Use the `FSSSD/fasdd_cv_filename_audit.py` script:

```bash
cd FSSSD
python fasdd_cv_filename_audit.py \
    --images-dir /path/to/FASDD_CV/images \
    --expect-images 95314 \
    --output-csv /path/to/output/fasdd_cv_filename_audit.csv \
    --preview
````
---

## Step 2 — Copy only fire scenes into `fire2/`

Next, build a “fire-scene only” subset: images whose filenames indicate fire is present.

Use the `FSSSD/copy_fire_scenes.py` script:

```bash
python copy_fire_scenes.py \
        --images-dir /path/to/FASDD_CV/images \
        --csv-path fasdd_cv_filename_audit.csv \
        --output-dir fire2  # path containing **32,701** fire-scene images
```
folder `fire2/` should contain **32,701** fire-scene images.

---

## Step 3 — Compute DINOv2 embeddings for `fire2/` (EVFR)

Next, represent each fire-scene image by a 768-D embedding for deduplication.

Inside the `FSSSD/` folder, clone the **[EVFR GitHub Repository](https://github.com/jihun-yoon/EVFR)**:

```bash
git clone https://github.com/jihun-yoon/EVFR.git
cd EVFR
pip install -e .
```
Then:

```bash
cd ..

python embeddings_single_inference.py \
    --input_dir /path/to/fire2 \
    --output_dir /path/to/fire_embeddings \  # where the embeddings.npz would be saved
    --batch_size 8 \  # you decide  the batch size
    --device cuda   # or 'cpu'
```
---

## Step 4 — Inspect cosine similarity distribution

Now, to understand neighbor cosine similarity distribution to choose a dedup threshold, use the below command.

```bash
python inspect_cosine_hist.py \
    --emb-path /path/to/fire_embeddings \  # where the embeddings.npz was saved
    --topk 50 \
    --subsample 5000

#To save plots instead of just showing them
python inspect_cosine_hist.py \
    --emb-path /path/to/fire_embeddings \  # where the embeddings.npz was saved
    --save-prefix /path/to/cosine_hist
```
---

## Step 5 — Deduplicate fire scenes with FAISS

Now, to remove near-duplicate images (dedup), retaining one representative per highly similar cluster.

```bash
python dedup_with_faiss.py \
    --image-root /path/to/fire2 \  # path containing **32,701** fire-scene images
    --emb-path /path/to/fire_embeddings \  # where the embeddings.npz was saved
    --out-dir /path/to/fire2_dedu96_150 \  # where the deduped images will be saved
    --sim-thresh 0.96 \  # set this as you like
    --topk 150  # set this as you like
```
You can sweep through:

* `SIM_THRESH ∈ {0.93, 0.96}`
* `TOPK ∈ {20, 50, 70, 100, 150, 200, 500, 1000}`

From 32,701 fire images, for example:

* `SIM_THRESH = 0.96, TOPK = 150` → **21,403** images (chosen setting).
* `SIM_THRESH = 0.93, TOPK = 150` → **17,713** images (more aggressive).
---

## Step 6 — Split deduplicated set for QA (folders 1–7)

Next, divide the deduplicated images in `/path/to/fire2_dedu96_150` into manageable chunks for human QA (this is just a suggestion. We did this to better manage the annotation).

Source: `/path/to/fire2_dedu96_150` (21,403 images or your own images).
Target: `FSSSD/folder_1` … `FSSSD/folder_7`.

---

## Step 7 — Build `fire2_dedu96_boxes.csv` from YOLO labels

Next, convert YOLO bounding boxes for fire into a CSV suitable for FireSAM ONNX. The YOLO labels comes from `FASDD_CV/annotations/YOLO_CV/labels/*.txt`

```bash
python build_firesam_bbox_csv.py \
    --dedup-root /path/to/fire2_dedu96_150 \  # where the deduped images are saved
    --yolo-labels-root "/path/to/FASDD_CV/annotations/YOLO_CV/labels/*.txt \  # where the YOLO annotations are"
    --csv-out "FSSSD/fire2_dedu96_boxes.csv" \  # where the deduped csv will be saved
    --fire-class-ids 0
```
---

## Step 8 — Generate FireSAM ONNX masks (+ overlays + box images)

To do this, for each deduplicated image, generate a multi-instance segmentation mask using FireSAM ONNX, and also produce overlays and box-visualization images.

```bash
python firesam/eval/generate_masks_from_firesam_multi.py \
  --image-root /path/to/FSSSD/folder_1 \  # where the deduped images of folder_1 are saved
  --bbox-csv  "FSSSD/fire2_dedu96_boxes.csv" \  # where the deduped csv is saved
  --onnx-path "/path/to/onnx" \  # where the onnx32/16 is saved
  --output-dir "FSSSD/outputonnx/folder_1"  # where the annotated masks and overlays for folder_1 will be saved
```

(Then repeated for `folder_2`…`folder_7`.)

---

## Step 9 — Mask quality rules and manual QA

Next, manually go through all 7 folders and only accept high-quality automatic masks and mark the rest for re-annotation. See the paper for the guide we used in creating this dataset.

After QA:

* **18,584** images were accepted (from 21,403 deduped).

At the end, place all the images and masks of the accetped annotations into `FSSSD/annotated` folder.
```text
FSSSD/
annotated/
    images/       # 18,584 accepted RGB images
    masks/        # 18,584 accepted B&W masks
    overlays/     # 18,584 accepted overlays
```
---

## Step 10 — Consolidate accepted images + masks into FSSSD

Next, after building clean `FSSSD/annotated/images`, `FSSSD/annotated/masks`, and `FSSSD/annotated/overlays` folders, we need to build the “re-annotate” set using the accepted `FSSSD/annotated/overlays` folder, the `FSSSD/outputonnx` folder containing all the masks gotten before manually going through the annotation, and the images in the dedup folder (`/path/to/fire2_dedu96_150`).


```bash
python consolidate_fsssd.py \
    --good-overlay-dir FSSSD/FSSSD \
    --masks-root FSSSD/outputonnx \ #containing all folders (folder_1, folder_2 … folder_7)
    --source-images-dir /path/to/fire2_dedu96_150
```
You should now have
```text
FSSSD/
annotated/
    images/       # 18,584 accepted RGB images
    masks/        # 18,584 accepted B&W masks
    overlays/     # 18,584 accepted overlays
re_annotate/
    images/     # rejected / difficult cases
    masks/      # original auto masks (optional)
    overlay/    # original auto overlays (optional)
```
---

## Step 11 — Interactive re-annotation app (boxes + points)

Finally, manually refine or create new masks for the “re_annotate” images using box + point prompts in the web app:

```bash
python interactive_annotator.py
```
---
