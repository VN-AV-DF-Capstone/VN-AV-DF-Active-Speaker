"""Optical and pixel motion differential for mouth articulation."""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np


def compute_mouth_motion_series(mouth_crops: List[np.ndarray]) -> List[Optional[float]]:
    """Compute frame-by-frame absolute pixel differential of mouth crops.

    Returns:
        List of motion values. The first frame is None as there is no preceding frame.
    """
    if not mouth_crops:
        return []

    motions: List[Optional[float]] = [None]
    prev_mouth = mouth_crops[0]

    for curr_mouth in mouth_crops[1:]:
        if prev_mouth is None or curr_mouth is None:
            motions.append(None)
        else:
            diff = cv2.absdiff(curr_mouth, prev_mouth)
            motion_val = float(np.mean(diff))
            motions.append(motion_val)
        prev_mouth = curr_mouth

    return motions
