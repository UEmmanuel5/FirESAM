import argparse
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

from firesam.utils.prompts import rasterize_prompts


def detect_input_dtype(sess: ort.InferenceSession) -> np.dtype:
    t = sess.get_inputs()[0].type
    if "float16" in t:
        return np.float16
    if "float" in t:
        return np.float32
    return np.float32


def build_prompts_from_yolo(h, w, boxes_xyxy, point_radius=3) -> np.ndarray:
    prompt_accum = np.zeros((3, h, w), dtype=np.float32)
    for box in boxes_xyxy:
        x1, y1, x2, y2 = box.astype(np.float32)
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

        box_np = np.array([x1, y1, x2, y2], dtype=np.float32)
        pos_points = np.array([[cx, cy]], dtype=np.float32)
        neg_points = np.zeros((0, 2), dtype=np.float32)

        p = rasterize_prompts(h, w, box_np, pos_points, neg_points, point_radius=point_radius).numpy()
        prompt_accum = np.maximum(prompt_accum, p)
    return prompt_accum


def yolo_model_stats(yolo: YOLO) -> Tuple[Optional[int], Optional[float]]:
    """
    Returns (params, gflops) if available.
    """
    params = None
    gflops = None
    try:
        m = yolo.model
        params = sum(p.numel() for p in m.parameters())
    except Exception:
        pass
    try:
        info = yolo.model.info(verbose=False)
    except Exception:
        info = None
    try:
        if hasattr(yolo.model, "flops"):
            gflops = float(yolo.model.flops) / 1e9
    except Exception:
        pass

    return params, gflops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="Path to student ONNX")
    ap.add_argument("--video", required=True, help="Input video path")
    ap.add_argument("--mode", choices=["onnx", "pipeline"], default="onnx",
                    help="onnx = student only, pipeline = YOLO+prompts+student")
    ap.add_argument("--yolo", default=None, help="YOLO weights (required if mode=pipeline)")
    ap.add_argument("--provider", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--h", type=int, default=416)
    ap.add_argument("--w", type=int, default=608)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--fire_class", type=int, default=-1, help="Set to class id, or -1 for all")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--max_frames", type=int, default=0, help="0 = all frames")
    ap.add_argument("--point_radius", type=int, default=3)
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    video_path = Path(args.video)
    if not onnx_path.is_file():
        raise SystemExit(f"ONNX not found: {onnx_path}")
    if not video_path.is_file():
        raise SystemExit(f"Video not found: {video_path}")

    if args.mode == "pipeline" and not args.yolo:
        raise SystemExit("--yolo is required when --mode pipeline")

    providers = ["CPUExecutionProvider"]
    if args.provider == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    in_dtype = detect_input_dtype(sess)

    print(f"Providers: {sess.get_providers()}")
    print(f"ONNX input: {in_name} dtype={in_dtype} shape={sess.get_inputs()[0].shape}")
    print(f"ONNX output: {out_name}")

    yolo = None
    if args.mode == "pipeline":
        yolo = YOLO(args.yolo)
        yp, yg = yolo_model_stats(yolo)
        print(f"YOLO weights: {args.yolo}")
        print(f"YOLO params: {yp:,}" if yp is not None else "YOLO params: N/A")
        print(f"YOLO GFLOPs: {yg:.3f}" if yg is not None else "YOLO GFLOPs: N/A")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    H, W = args.h, args.w

    # Warm-up
    dummy = np.random.rand(1, 6, H, W).astype(np.float32)
    if in_dtype == np.float16:
        dummy = dummy.astype(np.float16)

    for _ in range(args.warmup):
        _ = sess.run([out_name], {in_name: dummy})
    n = 0
    t_total = []
    t_yolo = []
    t_prompts = []
    t_onnx = []
    t_pre = []
    t_post = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if args.max_frames > 0 and n > args.max_frames:
            break

        t0 = time.perf_counter()

        # preprocess
        p0 = time.perf_counter()
        frame_resized = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)
        img_bgr = frame_resized
        img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img_chw = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        p1 = time.perf_counter()

        prompt_ch = np.zeros((3, H, W), dtype=np.float32)
        boxes_xyxy = np.empty((0, 4), dtype=np.float32)

        # YOLO + prompts if pipeline
        y0 = y1 = pr0 = pr1 = 0.0
        if args.mode == "pipeline":
            y0 = time.perf_counter()
            res = yolo.predict(img_bgr, conf=args.conf, verbose=False)
            boxes = res[0].boxes
            if boxes is not None and boxes.xyxy is not None and boxes.xyxy.shape[0] > 0:
                boxes_xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
                if args.fire_class >= 0 and boxes.cls is not None:
                    cls = boxes.cls.cpu().numpy().astype(int)
                    boxes_xyxy = boxes_xyxy[cls == args.fire_class]
            y1 = time.perf_counter()

            pr0 = time.perf_counter()
            if boxes_xyxy.shape[0] > 0:
                prompt_ch = build_prompts_from_yolo(H, W, boxes_xyxy, point_radius=args.point_radius)
            pr1 = time.perf_counter()

        # ONNX input
        inp = np.concatenate([img_chw, prompt_ch], axis=0)[None, ...]  # (1,6,H,W)
        if in_dtype == np.float16:
            inp = inp.astype(np.float16)
        else:
            inp = inp.astype(np.float32)

        # ONNX
        o0 = time.perf_counter()
        logits = sess.run([out_name], {in_name: inp})[0]
        o1 = time.perf_counter()
        s0 = time.perf_counter()
        probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float32)))
        _mask = (probs[0, 0] > 0.5).astype(np.uint8)
        s1 = time.perf_counter()
        t1 = time.perf_counter()
        t_total.append(t1 - t0)
        t_pre.append(p1 - p0)
        t_onnx.append(o1 - o0)
        t_post.append(s1 - s0)
        if args.mode == "pipeline":
            t_yolo.append(y1 - y0)
            t_prompts.append(pr1 - pr0)
    cap.release()

    def summarize(name, xs):
        if not xs:
            return
        arr = np.array(xs, dtype=np.float64)
        mean_ms = 1000.0 * arr.mean()
        p50_ms = 1000.0 * np.percentile(arr, 50)
        p95_ms = 1000.0 * np.percentile(arr, 95)
        fps = 1.0 / arr.mean() if arr.mean() > 0 else 0.0
        print(f"{name:>10}: mean {mean_ms:8.3f} ms | p50 {p50_ms:8.3f} ms | p95 {p95_ms:8.3f} ms | FPS {fps:7.2f}")
    print(f"\nFrames: {len(t_total)}")
    summarize("TOTAL", t_total)
    summarize("pre", t_pre)
    if args.mode == "pipeline":
        summarize("yolo", t_yolo)
        summarize("prompts", t_prompts)
    summarize("onnx", t_onnx)
    summarize("post", t_post)

if __name__ == "__main__":
    main()
