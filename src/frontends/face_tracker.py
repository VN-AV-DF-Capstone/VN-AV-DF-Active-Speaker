"""Multi-Face Detection, Alignment, and Interpolated Tracking.

Supports InsightFace, MediaPipe, or OpenCV detectors with linear interpolation.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


FACE_SIZE = 112
ALIGN_TEMPLATE = np.array(
    [[38.3, 51.7], [73.5, 51.5], [56.0, 71.7], [41.5, 92.4], [70.7, 92.2]],
    dtype=np.float32,
)


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + 1e-8)


def align_face_and_mouth(frame: np.ndarray, bbox: np.ndarray, kps: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Warp face to 112x112 using 5 landmarks if available, otherwise crop bbox."""
    h, w = frame.shape[:2]
    if kps is not None and kps.shape == (5, 2):
        transform, _ = cv2.estimateAffinePartial2D(
            kps.astype(np.float32), ALIGN_TEMPLATE, method=cv2.LMEDS
        )
        if transform is not None:
            aligned = cv2.warpAffine(
                frame, transform, (FACE_SIZE, FACE_SIZE),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
            gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
            mouth = gray[72:110, 25:87]
            return gray, mouth

    # Fallback to bbox crop
    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - 0.10 * width))
    x2 = min(w, int(x2 + 0.10 * width))
    y1 = max(0, int(y1 - 0.10 * height))
    y2 = min(h, int(y2 + 0.10 * height))

    if x2 <= x1 or y2 <= y1:
        face = np.zeros((FACE_SIZE, FACE_SIZE), dtype=np.uint8)
        mouth = np.zeros((38, 62), dtype=np.uint8)
        return face, mouth

    crop = frame[y1:y2, x1:x2]
    face = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (FACE_SIZE, FACE_SIZE))
    mouth = face[72:110, 25:87]
    return face, mouth


class FaceTracker:
    """Multi-face tracker with modular detector backends."""

    def __init__(self, backend: str = "auto", detect_every: int = 2, det_size: int = 640):
        self.detect_every = max(1, detect_every)
        self.det_size = det_size
        self.backend_name = backend
        self.detector = None
        self._init_detector(backend)

    def _init_detector(self, backend: str):
        # 1. Try InsightFace if requested or auto
        if backend in ("insightface", "auto"):
            try:
                from insightface.app import FaceAnalysis
                app = FaceAnalysis(
                    name="buffalo_l",
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
                app.prepare(ctx_id=0, det_size=(self.det_size, self.det_size))
                self.detector = ("insightface", app)
                self.backend_name = "insightface"
                return
            except Exception:
                if backend == "insightface":
                    print("[FaceTracker] InsightFace initialization failed, trying fallbacks...")

        # 2. Try MediaPipe if available
        if backend in ("mediapipe", "auto"):
            try:
                import mediapipe as mp
                mp_face = mp.solutions.face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=0.5
                )
                self.detector = ("mediapipe", mp_face)
                self.backend_name = "mediapipe"
                return
            except Exception:
                pass

        # 3. Fallback to OpenCV Haar Cascade
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        self.detector = ("opencv", cascade)
        self.backend_name = "opencv"

    def _detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        h, w = frame.shape[:2]
        mode, model = self.detector
        detections = []

        if mode == "insightface":
            for face in model.get(frame):
                score = float(getattr(face, "det_score", 1.0))
                if score < 0.45:
                    continue
                kps = getattr(face, "kps", None)
                detections.append({
                    "bbox": np.asarray(face.bbox, dtype=np.float32),
                    "kps": np.asarray(kps, dtype=np.float32) if kps is not None else None,
                    "score": score,
                })
        elif mode == "mediapipe":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.process(rgb)
            if results and results.detections:
                for det in results.detections:
                    box = det.location_data.relative_bounding_box
                    x1 = box.xmin * w
                    y1 = box.ymin * h
                    x2 = (box.xmin + box.width) * w
                    y2 = (box.ymin + box.height) * h
                    detections.append({
                        "bbox": np.array([x1, y1, x2, y2], dtype=np.float32),
                        "kps": None,
                        "score": float(det.score[0] if det.score else 1.0),
                    })
        elif mode == "opencv":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
            for (x, y, fw, fh) in faces:
                detections.append({
                    "bbox": np.array([x, y, x + fw, y + fh], dtype=np.float32),
                    "kps": None,
                    "score": 0.8,
                })

        return detections

    def track(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Track faces across frames using detection + linear interpolation."""
        tracks: List[Dict[str, Any]] = []

        for frame_idx in range(0, len(frames), self.detect_every):
            dets = self._detect_faces(frames[frame_idx])
            for d in dets:
                d["frame"] = frame_idx

            candidates = []
            for t_idx, track in enumerate(tracks):
                gap = frame_idx - track["points"][-1]["frame"]
                if gap <= max(8, self.detect_every * 4):
                    for d_idx, d in enumerate(dets):
                        iou_val = compute_iou(track["points"][-1]["bbox"], d["bbox"])
                        if iou_val >= 0.25:
                            candidates.append((iou_val, t_idx, d_idx))

            used_tracks, used_dets = set(), set()
            for _, t_idx, d_idx in sorted(candidates, reverse=True):
                if t_idx in used_tracks or d_idx in used_dets:
                    continue
                tracks[t_idx]["points"].append(dets[d_idx])
                used_tracks.add(t_idx)
                used_dets.add(d_idx)

            for d_idx, d in enumerate(dets):
                if d_idx not in used_dets:
                    tracks.append({"track_id": len(tracks), "points": [d]})

        # Dense interpolation across all frames
        dense_tracks = []
        for track in tracks:
            pts = track["points"]
            if len(pts) < 2:
                continue
            start_f, end_f = pts[0]["frame"], pts[-1]["frame"]
            xp = np.array([p["frame"] for p in pts])
            frame_ids = np.arange(start_f, end_f + 1)

            # Interpolate bbox [x1, y1, x2, y2]
            bbox = np.stack([
                np.interp(frame_ids, xp, [p["bbox"][axis] for p in pts])
                for axis in range(4)
            ], axis=1)

            # Interpolate landmarks if available
            has_kps = all(p["kps"] is not None for p in pts)
            kps_arr = None
            if has_kps:
                kps_arr = np.empty((len(frame_ids), 5, 2), dtype=np.float32)
                for lm in range(5):
                    for ax in range(2):
                        kps_arr[:, lm, ax] = np.interp(
                            frame_ids, xp, [p["kps"][lm, ax] for p in pts]
                        )

            dense_tracks.append({
                "track_id": track["track_id"],
                "frames": frame_ids,
                "bbox": bbox,
                "kps": kps_arr,
            })

        return dense_tracks
