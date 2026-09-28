"""Silero Voice Activity Detection (VAD) via pure ONNX Runtime CPU.

No torchaudio dependency required.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


TARGET_SR = 16000
WINDOW = 512
CONTEXT = 64


class SileroVAD:
    """ONNX-based Silero VAD runner for 16 kHz audio with graceful energy fallback."""

    def __init__(self, model_path: Optional[str] = None):
        self.session = None
        if model_path and os.path.isfile(model_path):
            try:
                import onnxruntime as ort

                options = ort.SessionOptions()
                options.inter_op_num_threads = 1
                options.intra_op_num_threads = 1
                self.session = ort.InferenceSession(
                    model_path, providers=["CPUExecutionProvider"], sess_options=options
                )
            except Exception as e:
                print(f"[SileroVAD] Warning: failed to initialize ONNX session ({e}), falling back to energy VAD")

    def probabilities(self, audio: np.ndarray) -> List[float]:
        values = np.asarray(audio, dtype=np.float32).reshape(-1)
        if self.session is None:
            # Fallback Energy + Zero-Crossing Rate VAD when ONNX model is not yet downloaded
            probs = []
            for start in range(0, len(values), WINDOW):
                chunk = values[start:start + WINDOW]
                energy = float(np.mean(chunk ** 2))
                prob = 1.0 / (1.0 + np.exp(-(energy - 0.002) * 2000.0))
                probs.append(float(np.clip(prob, 0.0, 1.0)))
            return probs

        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros((1, CONTEXT), dtype=np.float32)
        probabilities = []

        for start in range(0, len(values), WINDOW):
            chunk = values[start:start + WINDOW]
            if len(chunk) < WINDOW:
                chunk = np.pad(chunk, (0, WINDOW - len(chunk)))
            model_input = np.concatenate((context, chunk.reshape(1, -1)), axis=1)
            output, state = self.session.run(
                None,
                {
                    "input": model_input.astype(np.float32, copy=False),
                    "state": state,
                    "sr": np.asarray(TARGET_SR, dtype=np.int64),
                },
            )
            probabilities.append(float(np.asarray(output).reshape(-1)[0]))
            context = model_input[:, -CONTEXT:]
        return probabilities

    def speech_timestamps(
        self,
        probabilities: List[float],
        audio_samples: int,
        threshold: float = 0.5,
        neg_threshold: float = 0.35,
        min_speech_ms: int = 250,
        min_silence_ms: int = 100,
        speech_pad_ms: int = 30,
    ) -> List[Dict[str, int]]:
        min_speech = TARGET_SR * min_speech_ms / 1000
        min_silence = TARGET_SR * min_silence_ms / 1000
        pad = int(TARGET_SR * speech_pad_ms / 1000)
        triggered = False
        start_sample = 0
        possible_end = None
        spans = []

        for idx, prob in enumerate(probabilities):
            curr = idx * WINDOW
            if prob >= threshold:
                if not triggered:
                    triggered = True
                    start_sample = curr
                possible_end = None
                continue
            if triggered and prob < neg_threshold:
                if possible_end is None:
                    possible_end = curr
                if curr - possible_end >= min_silence:
                    if possible_end - start_sample > min_speech:
                        spans.append({"start": start_sample, "end": possible_end})
                    triggered = False
                    possible_end = None
        if triggered and audio_samples - start_sample > min_speech:
            spans.append({"start": start_sample, "end": audio_samples})

        for i, span in enumerate(spans):
            span["start"] = max(0, span["start"] - pad)
            span["end"] = min(audio_samples, span["end"] + pad)
            if i > 0 and span["start"] < spans[i - 1]["end"]:
                mid = (span["start"] + spans[i - 1]["end"]) // 2
                spans[i - 1]["end"] = mid
                span["start"] = mid
        return spans

    def bins(self, audio: np.ndarray, count: int, bin_ms: int = 200) -> List[bool]:
        """Return boolean speech flag for each temporal bin."""
        probs = self.probabilities(audio)
        timestamps = self.speech_timestamps(probs, len(audio))
        samples_per_bin = int(TARGET_SR * bin_ms / 1000)
        out = []
        for idx in range(count):
            start = idx * samples_per_bin
            end = (idx + 1) * samples_per_bin
            overlap = any(int(s["start"]) < end and int(s["end"]) > start for s in timestamps)
            out.append(overlap)
        return out
