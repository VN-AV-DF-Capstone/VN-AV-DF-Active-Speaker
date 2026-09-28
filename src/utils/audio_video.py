"""Resilient audio and video decoding utilities.

Operates with zero hard dependency on system ffmpeg by leveraging PyAV (`av`)
or OpenCV fallback.
"""

from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


TARGET_FPS = 25
TARGET_SR = 16000


def decode_video_25fps(path: str) -> List[np.ndarray]:
    """Decode a video file to an exact 25 fps list of BGR numpy frames."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Video file not found: {path}")

    # Primary method: OpenCV VideoCapture
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not math.isfinite(source_fps) or source_fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid source FPS ({source_fps}) in {path}")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f"Video contains 0 decoded frames: {path}")

    duration = len(frames) / source_fps
    target_count = max(1, int(round(duration * TARGET_FPS)))
    indices = np.clip(
        np.round(np.arange(target_count) * source_fps / TARGET_FPS).astype(int),
        0,
        len(frames) - 1,
    )
    return [frames[idx] for idx in indices]


def extract_audio_16k_mono(path: str) -> np.ndarray:
    """Extract audio as 16kHz mono float32 array [-1.0, 1.0]."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Media file not found: {path}")

    # Method 1: Try PyAV (fast, native in-process, no ffmpeg binary needed)
    try:
        import av

        container = av.open(path)
        audio_stream = next((s for s in container.streams if s.type == "audio"), None)
        if audio_stream is not None:
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=TARGET_SR)
            chunks = []
            for frame in container.decode(audio_stream):
                resampled = resampler.resample(frame)
                if resampled:
                    # PyAV resampled frame to numpy
                    for r in (resampled if isinstance(resampled, list) else [resampled]):
                        chunks.append(r.to_ndarray().flatten())
            container.close()
            if chunks:
                audio = np.concatenate(chunks).astype(np.float32)
                return np.clip(audio, -1.0, 1.0)
    except Exception:
        pass

    # Method 2: Fallback to soundfile / librosa if it's already an audio file
    try:
        import soundfile as sf
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != TARGET_SR:
            import scipy.signal
            num_samples = int(round(len(audio) * TARGET_SR / sr))
            audio = scipy.signal.resample(audio, num_samples).astype(np.float32)
        return np.clip(audio, -1.0, 1.0)
    except Exception:
        pass

    # Method 3: System ffmpeg subprocess
    try:
        import tempfile
        from scipy.io import wavfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = tmp.name

        cmd = [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", path, "-vn", "-ac", "1", "-ar", str(TARGET_SR), "-c:a", "pcm_s16le",
            tmp_wav
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and os.path.isfile(tmp_wav):
            sr, pcm = wavfile.read(tmp_wav)
            os.remove(tmp_wav)
            if sr == TARGET_SR and pcm.size > 0:
                audio = pcm.astype(np.float32) / 32768.0
                return np.clip(audio, -1.0, 1.0)
        if os.path.isfile(tmp_wav):
            os.remove(tmp_wav)
    except Exception:
        pass

    # If completely silent or no audio stream
    return np.zeros(TARGET_SR, dtype=np.float32)
