"""Local pose estimation using YOLOv8-pose ONNX.

Runs offline without any API key, outputting standard COCO-17 keypoints
matching the VLM Run vid.pose.kpts envelope.
"""

from __future__ import annotations

import time
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

from .skeleton import KPT_NAMES

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "yolov8n-pose.onnx"


def estimate_poses_local(
    video_path: Path,
    *,
    model_path: Path = DEFAULT_MODEL,
    conf_threshold: float = 0.35,
    nms_threshold: float = 0.45,
    max_frames: int | None = None,
    console = None
) -> tuple[dict, dict]:
    """Run local ONNX pose estimation and return ``(payload, usage)``."""
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Local pose model not found at {model_path}. "
            "Please make sure models/yolov8n-pose.onnx exists."
        )

    t0 = time.perf_counter()
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_meta = []
    items = []

    frame_idx = 0
    while cap.isOpened():
        if max_frames and frame_idx >= max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        frames_meta.append({"frame_id": frame_idx})
        h, w = frame.shape[:2]

        # Letterbox 640x640
        scale = min(640 / w, 640 / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        dx = (640 - nw) // 2
        dy = (640 - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = resized

        # Preprocess
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(rgb, (2, 0, 1))[None, ...]

        # Inference
        preds = sess.run(None, {"images": blob})[0][0].T  # (8400, 56)

        scores = preds[:, 4]
        mask = scores > conf_threshold
        if np.any(mask):
            filtered = preds[mask]
            filtered_scores = scores[mask]
            boxes = []
            for row in filtered:
                cx, cy, bw, bh = row[:4]
                boxes.append([int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh)])

            indices = cv2.dnn.NMSBoxes(boxes, filtered_scores.tolist(), conf_threshold, nms_threshold)

            for rank, idx in enumerate(indices):
                row = filtered[idx]
                cx, cy, bw, bh = row[:4]

                # Convert bbox to normalized [0, 1] relative to original frame
                orig_x = max(0.0, min(1.0, float((cx - bw / 2 - dx) / scale / w)))
                orig_y = max(0.0, min(1.0, float((cy - bh / 2 - dy) / scale / h)))
                orig_w = max(0.0, min(1.0, float(bw / scale / w)))
                orig_h = max(0.0, min(1.0, float(bh / scale / h)))

                # Convert 17 keypoints to normalized [0, 1]
                kpts_raw = row[5:].reshape(17, 3)
                kpts_xy = []
                kpts_scores = []
                for kx, ky, kc in kpts_raw:
                    kx_norm = max(0.0, min(1.0, float((kx - dx) / scale / w)))
                    ky_norm = max(0.0, min(1.0, float((ky - dy) / scale / h)))
                    kpts_xy.append([round(kx_norm, 4), round(ky_norm, 4)])
                    kpts_scores.append(round(float(kc), 3))

                items.append({
                    "frame_id": frame_idx,
                    "track_id": rank,
                    "bbox_xywh": [round(orig_x, 4), round(orig_y, 4), round(orig_w, 4), round(orig_h, 4)],
                    "kpts_xy": kpts_xy,
                    "kpts_scores": kpts_scores,
                })

        frame_idx += 1

    cap.release()
    elapsed = time.perf_counter() - t0

    payload = {
        "content": {
            "object": "vid.pose.kpts",
            "kpts_labels": KPT_NAMES,
            "frames": frames_meta,
            "items": items,
        }
    }

    usage = {
        "cost": 0.0,
        "local": True,
        "model": "yolov8n-pose-onnx",
        "elapsed_seconds": round(elapsed, 2),
        "total_frames": frame_idx,
    }

    return payload, usage
