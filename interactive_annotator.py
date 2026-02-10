"""
interactive_annotator_app.py
----------------------------
Improved interactive web app (Flask) for re-annotation with:
- Live box drawing while dragging.
- Click-to-add positive / negative points.
- Zoom with Ctrl + wheel (coordinates stay in original image space).
- Real-time visual prompts (boxes + points) on top of image.
- Mask + overlay generation via FirESAM ONNX.
- Saved / Working / Not started lists.

Run:
    pip install flask opencv-python onnxruntime numpy
    python interactive_annotator_app.py

Then open http://127.0.0.1:5000
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from flask import (
    Flask,
    jsonify,
    request,
    send_file,
    render_template_string,
)

from firesam.utils.prompts import rasterize_prompts

# =====================
# CONFIG
# =====================

REANNOTATE_DIR = Path("FSSSD/FSSSD/re_annotate")

SAVED_ROOT = REANNOTATE_DIR / "saved"
SAVED_IMAGES_DIR = SAVED_ROOT / "images"
SAVED_MASKS_DIR = SAVED_ROOT / "masks"
SAVED_OVERLAY_DIR = SAVED_ROOT / "overlay"
SAVED_PROMPTS_DIR = SAVED_ROOT / "prompts"

PREVIEW_DIR = REANNOTATE_DIR / "preview"

ONNX_PATH = "onnx/combined/firesam_limfunet_kd32.onnx"
ONNX_HEIGHT = 416
ONNX_WIDTH = 608
POINT_RADIUS = 1
SIGMOID_THRESHOLD = 0.5

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# =====================
# APP INIT
# =====================

app = Flask(__name__)

REANNOTATE_DIR.mkdir(parents=True, exist_ok=True)
SAVED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
SAVED_MASKS_DIR.mkdir(parents=True, exist_ok=True)
SAVED_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
SAVED_PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


def list_images(root: Path) -> List[Path]:
    return sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


image_paths: List[Path] = list_images(REANNOTATE_DIR)
image_names: List[str] = [p.name for p in image_paths]
name_to_path: Dict[str, Path] = {p.name: p for p in image_paths}

image_sizes: Dict[str, Tuple[int, int]] = {}  # name -> (w, h)
prompts_state: Dict[str, Dict[str, List[List[float]]]] = {}

ort_session = ort.InferenceSession(
    ONNX_PATH,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)


# =====================
# HELPERS
# =====================

def get_image_size(name: str) -> Tuple[int, int]:
    if name in image_sizes:
        return image_sizes[name]
    p = name_to_path.get(name)
    if p is None or not p.is_file():
        raise FileNotFoundError(f"Image not found: {name}")
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {p}")
    h, w = img.shape[:2]
    image_sizes[name] = (w, h)
    return w, h


def resize_and_normalize(image_bgr: np.ndarray) -> np.ndarray:
    img_resized = cv2.resize(image_bgr, (ONNX_WIDTH, ONNX_HEIGHT), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    img_chw = np.transpose(img_rgb, (2, 0, 1))
    return img_chw


def scale_box(
    box: Tuple[float, float, float, float],
    orig_w: int,
    orig_h: int,
) -> np.ndarray:
    x1, y1, x2, y2 = box
    sx = ONNX_WIDTH / float(orig_w)
    sy = ONNX_HEIGHT / float(orig_h)
    x1_r = np.clip(x1 * sx, 0, ONNX_WIDTH - 1)
    y1_r = np.clip(y1 * sy, 0, ONNX_HEIGHT - 1)
    x2_r = np.clip(x2 * sx, 0, ONNX_WIDTH - 1)
    y2_r = np.clip(y2 * sy, 0, ONNX_HEIGHT - 1)
    return np.array([x1_r, y1_r, x2_r, y2_r], dtype=np.float32)


def scale_points(
    points: List[List[float]],
    orig_w: int,
    orig_h: int,
) -> np.ndarray:
    if not points:
        return np.zeros((0, 2), dtype=np.float32)
    sx = ONNX_WIDTH / float(orig_w)
    sy = ONNX_HEIGHT / float(orig_h)
    arr = np.array(points, dtype=np.float32)
    arr[:, 0] = np.clip(arr[:, 0] * sx, 0, ONNX_WIDTH - 1)
    arr[:, 1] = np.clip(arr[:, 1] * sy, 0, ONNX_HEIGHT - 1)
    return arr


def make_overlay(
    original_bgr: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.3,
) -> np.ndarray:
    h_img, w_img = original_bgr.shape[:2]
    h_m, w_m = mask.shape[:2]
    if (h_m, w_m) != (h_img, w_img):
        mask_resized = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
    else:
        mask_resized = mask

    overlay = original_bgr.copy().astype(np.float32)
    green = np.array([0, 255, 0], dtype=np.float32)

    m = mask_resized > 0
    if np.any(m):
        overlay[m] = (1.0 - alpha) * overlay[m] + alpha * green

    return overlay.astype(np.uint8)


def run_inference_for_image(
    name: str,
    boxes: List[List[float]],
    pos_points: List[List[float]],
    neg_points: List[List[float]],
    save_paths: Tuple[Path, Path] | None = None,
) -> Path:
    img_path = name_to_path.get(name)
    if img_path is None:
        raise FileNotFoundError(f"Image not found: {name}")

    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")

    orig_h, orig_w = img_bgr.shape[:2]
    img_chw = resize_and_normalize(img_bgr)

    prompt_acc = np.zeros((3, ONNX_HEIGHT, ONNX_WIDTH), dtype=np.float32)

    use_boxes = boxes if boxes else [[0.0, 0.0, float(orig_w - 1), float(orig_h - 1)]]

    pos_scaled_all = scale_points(pos_points, orig_w, orig_h)
    neg_scaled_all = scale_points(neg_points, orig_w, orig_h)

    for box in use_boxes:
        box_resized = scale_box(tuple(box), orig_w, orig_h)
        prompt_ch = rasterize_prompts(
            height=ONNX_HEIGHT,
            width=ONNX_WIDTH,
            box=box_resized,
            pos_points=pos_scaled_all,
            neg_points=neg_scaled_all,
            point_radius=POINT_RADIUS,
        )
        if hasattr(prompt_ch, "detach"):
            prompt_ch = prompt_ch.detach().cpu().numpy()
        prompt_ch = prompt_ch.astype(np.float32)
        prompt_acc = np.maximum(prompt_acc, prompt_ch)

    input_6ch = np.concatenate([img_chw, prompt_acc], axis=0)
    input_6ch = np.expand_dims(input_6ch, axis=0)

    input_name = ort_session.get_inputs()[0].name
    outputs = ort_session.run(None, {input_name: input_6ch})
    logits = outputs[0]
    probs = 1.0 / (1.0 + np.exp(-logits))
    mask_small = (probs[0, 0] >= SIGMOID_THRESHOLD).astype(np.uint8) * 255

    mask_full = cv2.resize(mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    overlay = make_overlay(img_bgr, mask_full, alpha=0.3)

    if save_paths is None:
        out_overlay = PREVIEW_DIR / f"{Path(name).stem}_preview.png"
        cv2.imwrite(str(out_overlay), overlay)
        return out_overlay
    else:
        mask_path, overlay_path = save_paths
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_path), mask_full)
        cv2.imwrite(str(overlay_path), overlay)
        return overlay_path


def is_saved(name: str) -> bool:
    mask_path = SAVED_MASKS_DIR / f"{Path(name).stem}.png"
    overlay_path = SAVED_OVERLAY_DIR / f"{Path(name).stem}.png"
    return mask_path.is_file() and overlay_path.is_file()


def load_saved_prompts(name: str) -> Dict[str, List[List[float]]]:
    jpath = SAVED_PROMPTS_DIR / f"{Path(name).stem}.json"
    if not jpath.is_file():
        return {"boxes": [], "pos": [], "neg": []}
    with open(jpath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_prompts(name: str, prompts: Dict[str, List[List[float]]]) -> None:
    jpath = SAVED_PROMPTS_DIR / f"{Path(name).stem}.json"
    jpath.parent.mkdir(parents=True, exist_ok=True)
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(prompts, f)


def get_status_lists() -> Dict[str, List[str]]:
    saved = []
    working = []
    not_started = []

    for name in image_names:
        if is_saved(name):
            saved.append(name)
        else:
            st = prompts_state.get(name)
            if st and (st.get("boxes") or st.get("pos") or st.get("neg")):
                working.append(name)
            else:
                not_started.append(name)

    return {
        "saved": saved,
        "working": working,
        "not_started": not_started,
        "all": image_names,
    }


# =====================
# ROUTES
# =====================

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/api/list_status")
def api_list_status():
    return jsonify(get_status_lists())


@app.route("/api/get_state")
def api_get_state():
    name = request.args.get("name")
    if not name:
        status = get_status_lists()
        candidates = status["working"] + status["not_started"]
        if candidates:
            name = candidates[0]
        elif status["saved"]:
            name = status["saved"][0]
        else:
            return jsonify({"error": "No images"}), 404

    if name not in name_to_path:
        return jsonify({"error": f"Unknown image: {name}"}), 404

    w, h = get_image_size(name)

    if name in prompts_state:
        prompts = prompts_state[name]
    else:
        prompts = load_saved_prompts(name)
        prompts_state[name] = prompts

    return jsonify(
        {
            "name": name,
            "width": w,
            "height": h,
            "prompts": prompts,
            "saved": is_saved(name),
            "status_lists": get_status_lists(),
        }
    )


@app.route("/api/original_image")
def api_original_image():
    name = request.args.get("name")
    if not name or name not in name_to_path:
        return "Not found", 404
    return send_file(str(name_to_path[name]), mimetype="image/jpeg")


@app.route("/api/preview_overlay")
def api_preview_overlay():
    name = request.args.get("name")
    if not name:
        return "Missing name", 400
    path = PREVIEW_DIR / f"{Path(name).stem}_preview.png"
    if not path.is_file():
        if name not in name_to_path:
            return "Not found", 404
        return send_file(str(name_to_path[name]), mimetype="image/jpeg")
    return send_file(str(path), mimetype="image/png")


@app.route("/api/update_prompts", methods=["POST"])
def api_update_prompts():
    data = request.get_json(silent=True) or {}
    name = data.get("name")
    if not name or name not in name_to_path:
        return jsonify({"error": "Unknown image"}), 400

    boxes = data.get("boxes") or []
    pos = data.get("pos_points") or []
    neg = data.get("neg_points") or []

    prompts_state[name] = {
        "boxes": boxes,
        "pos": pos,
        "neg": neg,
    }

    if boxes or pos or neg:
        run_inference_for_image(name, boxes, pos, neg, save_paths=None)
    else:
        p = PREVIEW_DIR / f"{Path(name).stem}_preview.png"
        if p.is_file():
            p.unlink()

    return jsonify({"ok": True, "status_lists": get_status_lists()})


@app.route("/api/reset_prompts", methods=["POST"])
def api_reset_prompts():
    data = request.get_json(silent=True) or {}
    name = data.get("name")
    if not name or name not in name_to_path:
        return jsonify({"error": "Unknown image"}), 400
    prompts_state[name] = {"boxes": [], "pos": [], "neg": []}
    p = PREVIEW_DIR / f"{Path(name).stem}_preview.png"
    if p.is_file():
        p.unlink()
    return jsonify({"ok": True, "status_lists": get_status_lists()})


@app.route("/api/save", methods=["POST"])
def api_save():
    data = request.get_json(silent=True) or {}
    name = data.get("name")
    if not name or name not in name_to_path:
        return jsonify({"error": "Unknown image"}), 400

    prompts = prompts_state.get(name) or {"boxes": [], "pos": [], "neg": []}
    boxes = prompts.get("boxes") or []
    pos = prompts.get("pos") or []
    neg = prompts.get("neg") or []

    if not boxes and not pos and not neg:
        return jsonify({"error": "No prompts defined"}), 400

    mask_path = SAVED_MASKS_DIR / f"{Path(name).stem}.png"
    overlay_path = SAVED_OVERLAY_DIR / f"{Path(name).stem}.png"

    run_inference_for_image(name, boxes, pos, neg, save_paths=(mask_path, overlay_path))

    src_img = name_to_path[name]
    dst_img = SAVED_IMAGES_DIR / src_img.name
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    if str(src_img.resolve()) != str(dst_img.resolve()):
        import shutil
        shutil.copy2(str(src_img), str(dst_img))

    save_prompts(name, prompts)

    prev = PREVIEW_DIR / f"{Path(name).stem}_preview.png"
    if prev.is_file():
        prev.unlink()

    return jsonify({"ok": True, "status_lists": get_status_lists()})


@app.route("/api/next_image", methods=["POST"])
def api_next_image():
    data = request.get_json(silent=True) or {}
    current = data.get("name")
    status = get_status_lists()
    all_names = status["all"]
    saved = set(status["saved"])

    if not all_names:
        return jsonify({"error": "No images"}), 400

    if current not in all_names:
        candidates = [n for n in all_names if n not in saved]
        if not candidates:
            return jsonify({"name": all_names[0]})
        return jsonify({"name": candidates[0]})

    idx = all_names.index(current)
    n = len(all_names)
    for step in range(1, n + 1):
        j = (idx + step) % n
        name = all_names[j]
        if name not in saved:
            return jsonify({"name": name})
    return jsonify({"name": current})


@app.route("/api/prev_image", methods=["POST"])
def api_prev_image():
    data = request.get_json(silent=True) or {}
    current = data.get("name")
    status = get_status_lists()
    all_names = status["all"]
    saved = set(status["saved"])

    if not all_names:
        return jsonify({"error": "No images"}), 400

    if current not in all_names:
        candidates = [n for n in all_names if n not in saved]
        if not candidates:
            return jsonify({"name": all_names[0]})
        return jsonify({"name": candidates[0]})

    idx = all_names.index(current)
    n = len(all_names)
    for step in range(1, n + 1):
        j = (idx - step) % n
        name = all_names[j]
        if name not in saved:
            return jsonify({"name": name})
    return jsonify({"name": current})


# =====================
# FRONTEND
# =====================

INDEX_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Fire Re-Annotation Tool</title>
  <style>
    * {
      box-sizing: border-box;
      user-select: none;
    }
    body {
      font-family: sans-serif;
      margin: 0;
      padding: 0;
      display: flex;
      height: 100vh;
      overflow: hidden;
    }
    #left-panel {
      flex: 3;
      display: flex;
      flex-direction: column;
      padding: 8px;
      border-right: 1px solid #ccc;
    }
    #right-panel {
      flex: 1;
      display: flex;
      flex-direction: column;
      padding: 8px;
      font-size: 12px;
    }
    #toolbar {
      margin-bottom: 6px;
    }
    #toolbar button {
      margin-right: 4px;
      margin-bottom: 4px;
    }
    #status-bar {
      margin-top: 4px;
      font-size: 12px;
    }
    #image-container {
      position: relative;
      flex: 1;
      border: 1px solid #ccc;
      background: #111;
      overflow: auto;
    }
    #base-image {
      position: absolute;
      left: 0;
      top: 0;
      transform-origin: top left;
      image-rendering: auto;
      pointer-events: none;
    }
    #draw-canvas {
      position: absolute;
      left: 0;
      top: 0;
      transform-origin: top left;
      cursor: crosshair;
    }
    #lists {
      display: flex;
      flex: 1;
      overflow: auto;
      gap: 8px;
    }
    .list-box {
      flex: 1;
      border: 1px solid #ccc;
      padding: 4px;
      overflow: auto;
    }
    .list-box h3 {
      margin: 0 0 4px 0;
      font-size: 12px;
      text-align: center;
    }
    .list-box ul {
      list-style: none;
      padding: 0;
      margin: 0;
    }
    .list-box li {
      padding: 2px 4px;
      white-space: nowrap;
    }
    .current {
      background-color: #d0ebff;
    }
    .working {
      color: #b36b00;
    }
    .saved {
      color: #008000;
    }
    .not-started {
      color: #aaa;
    }
  </style>
</head>
<body>
  <div id="left-panel">
    <div id="toolbar">
      <button id="mode-box">Box mode</button>
      <button id="mode-pos">Positive point</button>
      <button id="mode-neg">Negative point</button>
      <button id="reset-btn">Reset</button>
      <button id="save-btn">Save</button>
      <button id="prev-btn">&laquo; Prev</button>
      <button id="next-btn">Next &raquo;</button>
      <span id="mode-label"></span>
    </div>
    <div id="image-container">
      <img id="base-image" src="" alt="image" draggable="false">
      <canvas id="draw-canvas"></canvas>
    </div>
    <div id="status-bar">
      <span id="current-image-label"></span>
      &nbsp;|&nbsp;
      Zoom: <span id="zoom-label">100%</span>
      &nbsp;|&nbsp;
      Ctrl + wheel = zoom, Left drag = box (in box mode), Left click = point (in pos/neg modes)
    </div>
  </div>

  <div id="right-panel">
    <div id="lists">
      <div class="list-box">
        <h3>Saved</h3>
        <ul id="saved-list"></ul>
      </div>
      <div class="list-box">
        <h3>Working</h3>
        <ul id="working-list"></ul>
      </div>
      <div class="list-box">
        <h3>Not started</h3>
        <ul id="not-started-list"></ul>
      </div>
    </div>
  </div>

<script>
  const maxDisplayWidth = 1024;
  const maxDisplayHeight = 768;

  let currentName = null;
  let imgWidth = 0;
  let imgHeight = 0;
  let scale = 1.0;

  let boxes = [];
  let posPoints = [];
  let negPoints = [];

  let drawingBox = false;
  let boxStart = null;   // {x,y} in original coords
  let currentBox = null; // temp box while dragging {x1,y1,x2,y2} original coords

  let mode = "box"; // "box", "pos", "neg"

  const imgEl = document.getElementById("base-image");
  const canvas = document.getElementById("draw-canvas");
  const ctx = canvas.getContext("2d");

  const modeLabel = document.getElementById("mode-label");
  const zoomLabel = document.getElementById("zoom-label");
  const currentImageLabel = document.getElementById("current-image-label");

  const savedListEl = document.getElementById("saved-list");
  const workingListEl = document.getElementById("working-list");
  const notStartedListEl = document.getElementById("not-started-list");

  function setMode(m) {
    mode = m;
    if (mode === "box") {
      modeLabel.textContent = "Mode: Box (click-drag to draw box)";
    } else if (mode === "pos") {
      modeLabel.textContent = "Mode: Positive point (click to add +)";
    } else {
      modeLabel.textContent = "Mode: Negative point (click to add -)";
    }
  }

  function setZoom(z) {
    scale = Math.max(0.1, Math.min(5.0, z));
    zoomLabel.textContent = Math.round(scale * 100) + "%";
    updateCanvasSize();
    drawAnnotations();
  }

  function updateCanvasSize() {
    const w = imgWidth * scale;
    const h = imgHeight * scale;
    imgEl.style.width = w + "px";
    imgEl.style.height = h + "px";
    canvas.width = w;
    canvas.height = h;
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";
  }

  function imgToCanvas(x, y) {
    return [x * scale, y * scale];
  }

  function canvasToImg(x, y) {
    return [x / scale, y / scale];
  }

  function drawAnnotations() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // permanent boxes
    ctx.lineWidth = 2;
    ctx.setLineDash([]);
    for (const b of boxes) {
      const [cx1, cy1] = imgToCanvas(b.x1, b.y1);
      const [cx2, cy2] = imgToCanvas(b.x2, b.y2);
      ctx.strokeStyle = "yellow";
      ctx.strokeRect(
        Math.min(cx1, cx2),
        Math.min(cy1, cy2),
        Math.abs(cx2 - cx1),
        Math.abs(cy2 - cy1)
      );
    }

    // current box while dragging
    if (drawingBox && currentBox) {
      const [cx1, cy1] = imgToCanvas(currentBox.x1, currentBox.y1);
      const [cx2, cy2] = imgToCanvas(currentBox.x2, currentBox.y2);
      ctx.setLineDash([5, 3]);
      ctx.strokeStyle = "cyan";
      ctx.strokeRect(
        Math.min(cx1, cx2),
        Math.min(cy1, cy2),
        Math.abs(cx2 - cx1),
        Math.abs(cy2 - cy1)
      );
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(0,255,255,0.1)";
      ctx.fillRect(
        Math.min(cx1, cx2),
        Math.min(cy1, cy2),
        Math.abs(cx2 - cx1),
        Math.abs(cy2 - cy1)
      );
    }

    // positive points
    for (const p of posPoints) {
      const [cx, cy] = imgToCanvas(p.x, p.y);
      ctx.fillStyle = "lime";
      ctx.beginPath();
      ctx.arc(cx, cy, 5, 0, 2 * Math.PI);
      ctx.fill();
    }

    // negative points
    for (const p of negPoints) {
      const [cx, cy] = imgToCanvas(p.x, p.y);
      ctx.fillStyle = "red";
      ctx.beginPath();
      ctx.arc(cx, cy, 5, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  // Mouse events
  canvas.addEventListener("mousedown", (e) => {
    if (e.button !== 0) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const [ix, iy] = canvasToImg(x, y);

    if (mode === "box") {
      drawingBox = true;
      boxStart = {x: ix, y: iy};
      currentBox = {x1: ix, y1: iy, x2: ix, y2: iy};
      drawAnnotations();
    } else if (mode === "pos") {
      posPoints.push({x: ix, y: iy});
      drawAnnotations();
      sendPromptsUpdate();
    } else if (mode === "neg") {
      negPoints.push({x: ix, y: iy});
      drawAnnotations();
      sendPromptsUpdate();
    }
  });

  canvas.addEventListener("mousemove", (e) => {
    if (!drawingBox || mode !== "box" || !boxStart) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const [ix, iy] = canvasToImg(x, y);

    const x1 = Math.min(boxStart.x, ix);
    const y1 = Math.min(boxStart.y, iy);
    const x2 = Math.max(boxStart.x, ix);
    const y2 = Math.max(boxStart.y, iy);
    currentBox = {x1, y1, x2, y2};
    drawAnnotations();
  });

  canvas.addEventListener("mouseup", (e) => {
    if (e.button !== 0) return;
    if (!drawingBox || mode !== "box" || !boxStart || !currentBox) return;
    drawingBox = false;

    // only keep if box has some size
    const w = Math.abs(currentBox.x2 - currentBox.x1);
    const h = Math.abs(currentBox.y2 - currentBox.y1);
    if (w > 1 && h > 1) {
      boxes.push({
        x1: currentBox.x1,
        y1: currentBox.y1,
        x2: currentBox.x2,
        y2: currentBox.y2,
      });
    }
    boxStart = null;
    currentBox = null;
    drawAnnotations();
    sendPromptsUpdate();
  });

  canvas.addEventListener("mouseleave", (e) => {
    if (drawingBox && mode === "box") {
      drawingBox = false;
      boxStart = null;
      currentBox = null;
      drawAnnotations();
    }
  });

  // Zoom with Ctrl + wheel
  document.getElementById("image-container").addEventListener("wheel", (e) => {
    if (!e.ctrlKey) return;
    e.preventDefault();
    const delta = e.deltaY;
    if (delta < 0) {
      setZoom(scale * 1.1);
    } else {
      setZoom(scale / 1.1);
    }
  }, { passive: false });

  // Toolbar
  document.getElementById("mode-box").onclick = () => setMode("box");
  document.getElementById("mode-pos").onclick = () => setMode("pos");
  document.getElementById("mode-neg").onclick = () => setMode("neg");

  document.getElementById("reset-btn").onclick = () => {
    if (!currentName) return;
    if (!confirm("Reset prompts for this image?")) return;
    fetch("/api/reset_prompts", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({name: currentName}),
    })
    .then(r => r.json())
    .then(data => {
      boxes = [];
      posPoints = [];
      negPoints = [];
      boxStart = null;
      currentBox = null;
      drawAnnotations();
      imgEl.src = "/api/original_image?name=" + encodeURIComponent(currentName) + "&t=" + Date.now();
      if (data.status_lists) {
        updateLists(data.status_lists);
      }
    });
  };

  document.getElementById("save-btn").onclick = () => {
    if (!currentName) return;
    if (!boxes.length && !posPoints.length && !negPoints.length) {
      alert("No prompts defined.");
      return;
    }
    fetch("/api/save", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({name: currentName}),
    })
    .then(r => r.json())
    .then(data => {
      if (!data.ok) {
        alert("Save error: " + (data.error || "unknown"));
        return;
      }
      if (data.status_lists) {
        updateLists(data.status_lists);
      }
      // go to next unsaved
      fetch("/api/next_image", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({name: currentName}),
      })
      .then(r => r.json())
      .then(d => {
        if (d.name) {
          loadImageState(d.name);
        }
      });
    });
  };

  document.getElementById("next-btn").onclick = () => {
    if (!currentName) return;
    fetch("/api/next_image", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({name: currentName}),
    })
    .then(r => r.json())
    .then(d => {
      if (d.name) {
        loadImageState(d.name);
      }
    });
  };

  document.getElementById("prev-btn").onclick = () => {
    if (!currentName) return;
    fetch("/api/prev_image", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({name: currentName}),
    })
    .then(r => r.json())
    .then(d => {
      if (d.name) {
        loadImageState(d.name);
      }
    });
  };

  function sendPromptsUpdate() {
    if (!currentName) return;
    const payload = {
      name: currentName,
      boxes: boxes.map(b => [b.x1, b.y1, b.x2, b.y2]),
      pos_points: posPoints.map(p => [p.x, p.y]),
      neg_points: negPoints.map(p => [p.x, p.y]),
    };
    fetch("/api/update_prompts", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    })
    .then(r => r.json())
    .then(data => {
      if (data.status_lists) {
        updateLists(data.status_lists);
      }
      imgEl.src = "/api/preview_overlay?name=" + encodeURIComponent(currentName) + "&t=" + Date.now();
    });
  }

  function updateLists(status) {
    function fillList(el, arr, cls) {
      el.innerHTML = "";
      for (const n of arr) {
        const li = document.createElement("li");
        li.textContent = n;
        li.classList.add(cls);
        if (n === currentName) {
          li.classList.add("current");
        }
        el.appendChild(li);
      }
    }
    fillList(savedListEl, status.saved || [], "saved");
    fillList(workingListEl, status.working || [], "working");
    fillList(notStartedListEl, status.not_started || [], "not-started");
  }

  function loadImageState(name) {
    fetch("/api/get_state?name=" + encodeURIComponent(name))
    .then(r => r.json())
    .then(data => {
      if (data.error) {
        alert(data.error);
        return;
      }
      currentName = data.name;
      imgWidth = data.width;
      imgHeight = data.height;

      currentImageLabel.textContent = "Current: " + currentName + " (" + imgWidth + "x" + imgHeight + ")";

      const p = data.prompts || {boxes: [], pos: [], neg: []};
      boxes = (p.boxes || []).map(b => ({x1: b[0], y1: b[1], x2: b[2], y2: b[3]}));
      posPoints = (p.pos || []).map(x => ({x: x[0], y: x[1]}));
      negPoints = (p.neg || []).map(x => ({x: x[0], y: x[1]}));
      boxStart = null;
      currentBox = null;

      const zx = maxDisplayWidth / imgWidth;
      const zy = maxDisplayHeight / imgHeight;
      setZoom(Math.min(1.0, zx, zy));

      imgEl.src = "/api/preview_overlay?name=" + encodeURIComponent(currentName) + "&t=" + Date.now();
      drawAnnotations();
      if (data.status_lists) {
        updateLists(data.status_lists);
      }
    });
  }

  setMode("box");
  fetch("/api/get_state")
    .then(r => r.json())
    .then(data => {
      if (data.error) {
        alert(data.error);
        return;
      }
      loadImageState(data.name);
    });
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
