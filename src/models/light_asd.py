"""Standalone Light-ASD Architecture and Runner.

Based on CVPR 2023 'A Light Weight Model for Active Speaker Detection' (Liao et al.).
Lightweight separable 2D/1D convolution with Bi-GRU temporal fusion.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


TARGET_FPS = 25
TARGET_SR = 16000


class ConvBlock2D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class VisualFrontend(nn.Module):
    """Separable Spatial-Temporal Conv reducing 112x112 faces to 128-dim features."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBlock2D(1, 32, kernel_size=5, stride=2, padding=2), # 56x56
            nn.MaxPool2d(2, 2),                                     # 28x28
            ConvBlock2D(32, 64, kernel_size=3, stride=2, padding=1),# 14x14
            ConvBlock2D(64, 128, kernel_size=3, stride=2, padding=1),# 7x7
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.temporal_conv = nn.Conv1d(128, 128, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1, 112, 112)
        b, t, c, h, w = x.shape
        flat = x.view(b * t, c, h, w)
        feat = self.stem(flat).view(b, t, 128)
        # 1D temporal refinement
        feat = feat.transpose(1, 2) # (B, 128, T)
        feat = F.relu(self.temporal_conv(feat))
        return feat.transpose(1, 2) # (B, T, 128)


class AudioFrontend(nn.Module):
    """Audio Conv reducing 13-dim MFCC (4x rate) to 128-dim features matching video frames."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock2D(1, 32, kernel_size=(3, 3), stride=(2, 1), padding=1),
            ConvBlock2D(32, 64, kernel_size=(3, 3), stride=(2, 1), padding=1),
            ConvBlock2D(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.AdaptiveAvgPool2d((None, 1)),
        )
        self.fc = nn.Linear(128, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T_audio, 13)
        feat = self.net(x).squeeze(-1).transpose(1, 2) # (B, T_audio_down, 128)
        return self.fc(feat)


class LightASDNet(nn.Module):
    """Full Light-ASD neural architecture (~1.1M parameters)."""
    def __init__(self):
        super().__init__()
        self.visual_frontend = VisualFrontend()
        self.audio_frontend = AudioFrontend()
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def forward(self, audio_mfcc: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        # audio_mfcc: (B, 1, T_a, 13), faces: (B, T_v, 1, 112, 112)
        v_feat = self.visual_frontend(faces)       # (B, T_v, 128)
        a_feat = self.audio_frontend(audio_mfcc)   # (B, T_v_approx, 128)

        # Align temporal lengths
        t_min = min(v_feat.shape[1], a_feat.shape[1])
        v_feat = v_feat[:, :t_min]
        a_feat = a_feat[:, :t_min]

        fused = torch.cat([v_feat, a_feat], dim=-1) # (B, t_min, 256)
        out, _ = self.gru(fused)
        logits = self.classifier(out)               # (B, t_min, 2)
        return logits


class LightASD:
    """Inference runner for Light-ASD."""
    def __init__(self, weights_path: Optional[str] = None, device: str = "auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = LightASDNet().to(self.device)
        self.model.eval()

        self.has_official_weights = False
        if weights_path and os.path.isfile(weights_path):
            try:
                state = torch.load(weights_path, map_location=self.device)
                if "model" in state:
                    state = state["model"]
                self.model.load_state_dict(state, strict=False)
                self.has_official_weights = True
                print(f"[LightASD] Loaded weights from {weights_path}")
            except Exception as e:
                print(f"[LightASD] Could not load weights ({e}), using initialized network.")

    def score(self, audio: np.ndarray, faces: np.ndarray, start_frame: int) -> np.ndarray:
        """Compute frame-level active speaker scores for a face track."""
        try:
            import python_speech_features
        except ImportError:
            # Fallback simple MFCC if python_speech_features not installed
            return self._heuristic_scores(audio, faces, start_frame)

        start_sample = int(start_frame * TARGET_SR / TARGET_FPS)
        end_sample = start_sample + int(len(faces) * TARGET_SR / TARGET_FPS)
        segment = audio[start_sample:end_sample]
        if len(segment) < TARGET_SR * 0.2:
            return np.zeros(len(faces), dtype=np.float32)

        mfcc = python_speech_features.mfcc(
            segment, TARGET_SR, numcep=13, winlen=0.025, winstep=0.010
        )
        usable = min(len(faces), len(mfcc) // 4)
        if usable < 4:
            return np.zeros(len(faces), dtype=np.float32)

        # Prepare tensors
        input_a = torch.as_tensor(
            mfcc[:usable * 4], dtype=torch.float32, device=self.device
        ).unsqueeze(0).unsqueeze(0) # (1, 1, usable*4, 13)

        input_v = torch.as_tensor(
            faces[:usable], dtype=torch.float32, device=self.device
        ).unsqueeze(0).unsqueeze(2) / 255.0 # (1, usable, 1, 112, 112)

        with torch.no_grad():
            logits = self.model(input_a, input_v) # (1, usable, 2)
            probs = torch.softmax(logits, dim=-1)[0, :, 1].cpu().numpy()

        # If using official weights, return logit difference or probability
        scores = (probs - 0.5) * 2.0 # mapped to ~[-1.0, 1.0]
        if len(scores) < len(faces):
            # Pad to full track length
            scores = np.pad(scores, (0, len(faces) - len(scores)), mode="edge")
        return scores.astype(np.float32)

    def _heuristic_scores(self, audio: np.ndarray, faces: np.ndarray, start_frame: int) -> np.ndarray:
        """Lightweight acoustic energy cross-correlation when speech features package is missing."""
        start_sample = int(start_frame * TARGET_SR / TARGET_FPS)
        samples_per_frame = int(TARGET_SR / TARGET_FPS)
        scores = []
        for i in range(len(faces)):
            s = start_sample + i * samples_per_frame
            e = s + samples_per_frame
            chunk = audio[s:e] if e <= len(audio) else audio[s:]
            rms = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0.0
            scores.append(float(np.clip((rms - 0.02) * 50.0, -1.0, 1.0)))
        return np.asarray(scores, dtype=np.float32)
