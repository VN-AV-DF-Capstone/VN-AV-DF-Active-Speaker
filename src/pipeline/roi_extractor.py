"""Active Face ROI Extraction and Temporal Trajectory Smoothing."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class ROIExtractor:
    """Extracts and smooths bounding box trajectories for the primary active speaker."""

    def __init__(self, smoothing_window: int = 5, padding_scale: float = 0.25):
        self.smoothing_window = max(1, smoothing_window)
        self.padding_scale = padding_scale

    def extract_trajectory(
        self,
        tracks: List[Dict[str, Any]],
        primary_track_id: int,
        active_bins: List[Dict[str, Any]],
        frame_count: int,
        video_shape: tuple[int, int],
    ) -> List[Dict[str, Any]]:
        """Build a smooth, frame-by-frame ROI trajectory for the active speaker.

        Args:
            tracks: List of tracked face dictionaries from FaceTracker.
            primary_track_id: Track ID selected as primary active speaker.
            active_bins: Timeline bins containing speech & active labels.
            frame_count: Total frames in video.
            video_shape: (height, width) of the video.
        """
        h, w = video_shape
        matched_track = next((t for t in tracks if t["track_id"] == primary_track_id), None)
        if matched_track is None:
            return []

        frame_ids = matched_track["frames"]
        raw_bboxes = matched_track["bbox"] # (N, 4)

        # Smooth bounding box coordinates using a moving window
        smoothed_bboxes = np.copy(raw_bboxes)
        half_w = self.smoothing_window // 2
        for i in range(len(raw_bboxes)):
            start_idx = max(0, i - half_w)
            end_idx = min(len(raw_bboxes), i + half_w + 1)
            smoothed_bboxes[i] = np.mean(raw_bboxes[start_idx:end_idx], axis=0)

        # Build frame lookup for active speech status
        # 1 bin = 5 frames (200 ms at 25 fps)
        bin_active_map = {}
        for b in active_bins:
            b_idx = b["bin_index"]
            is_act = (b.get("selected_track_id") == primary_track_id) and b.get("speech", False)
            for f in range(b_idx * 5, (b_idx + 1) * 5):
                bin_active_map[f] = is_act

        trajectory = []
        for f_id, s_box in zip(frame_ids, smoothed_bboxes):
            x1, y1, x2, y2 = s_box
            bw, bh = x2 - x1, y2 - y1

            # Expand with padding scale
            pad_x = bw * self.padding_scale
            pad_y = bh * self.padding_scale
            px1 = max(0, int(round(x1 - pad_x)))
            py1 = max(0, int(round(y1 - pad_y)))
            px2 = min(w, int(round(x2 + pad_x)))
            py2 = min(h, int(round(y2 + pad_y)))

            trajectory.append({
                "frame": int(f_id),
                "is_speaking": bool(bin_active_map.get(int(f_id), False)),
                "raw_bbox": [float(v) for v in s_box],
                "padded_roi": [px1, py1, px2, py2],
                "track_id": primary_track_id,
            })

        return trajectory
