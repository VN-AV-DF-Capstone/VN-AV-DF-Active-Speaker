"""Video annotation and visualization for Active Speaker Detection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from src.core.data_types import ClipResult
from .audio_video import TARGET_FPS, decode_video_25fps


def render_annotated_video(
    video_path: str,
    result: ClipResult,
    output_path: str,
) -> str:
    """Generate an annotated demonstration video with HUD and bounding boxes."""
    frames = decode_video_25fps(video_path)
    if not frames:
        raise RuntimeError("No frames to annotate")

    h, w = frames[0].shape[:2]
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, TARGET_FPS, (w, h))

    # Build frame-level lookup for ROI and primary speaking status
    roi_by_frame = {item["frame"]: item for item in result.roi_trajectory}

    for f_idx, frame in enumerate(frames):
        annotated = frame.copy()

        # Draw ROI if present
        if f_idx in roi_by_frame:
            info = roi_by_frame[f_idx]
            x1, y1, x2, y2 = [int(v) for v in info["padded_roi"]]
            is_speaking = info["is_speaking"]

            color = (0, 230, 0) if is_speaking else (160, 160, 160)
            thickness = 3 if is_speaking else 1

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            label = f"Track {info['track_id']} {'[SPEAKING]' if is_speaking else '[LISTENING]'}"
            cv2.putText(
                annotated, label, (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        # Draw Top HUD Banner
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

        hud_text = (
            f"{result.clip_id} | Decision: {result.decision.upper()}"
            f"{f' ({result.reason})' if result.reason else ''} | "
            f"Active: {result.visible_active_speech_ratio:.1%} | Tier: {result.early_exit_tier}"
        )
        status_color = (0, 255, 0) if result.decision == "pass" else (0, 0, 255)
        cv2.putText(
            annotated, hud_text, (15, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2
        )

        writer.write(annotated)

    writer.release()
    return output_path
