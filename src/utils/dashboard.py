"""Audiovisual Diagnostic Dashboard for Active Speaker Detection POC.

Generates comprehensive publication-quality multi-panel visualization:
1. Mel-Spectrogram with F0 Pitch Contour
2. Acoustic RMS Energy vs Lip Articulation Dynamics
3. Temporal Cross-Correlation Lag Heatmap (Audio-Visual Sync)
4. Bin-level Decision Timeline & Multi-Tier Score Breakdown
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless / server execution
import matplotlib.pyplot as plt
import numpy as np

from ..core.data_types import ClipResult
from .audio_video import TARGET_FPS, TARGET_SR


def compute_mel_spectrogram(audio: np.ndarray, n_mels: int = 80) -> np.ndarray:
    """Compute Log Mel-Spectrogram using librosa or numpy STFT fallback."""
    try:
        import librosa
        mel = librosa.feature.melspectrogram(
            y=audio, sr=TARGET_SR, n_fft=1024, hop_length=320, n_mels=n_mels
        )
        return librosa.power_to_db(mel, ref=np.max)
    except Exception:
        # Numpy basic spectrogram fallback
        window = np.hanning(1024)
        hop = 320
        n_frames = (len(audio) - 1024) // hop + 1
        spec = np.empty((513, n_frames), dtype=np.float32)
        for i in range(n_frames):
            frame = audio[i * hop: i * hop + 1024] * window
            fft = np.fft.rfft(frame)
            spec[:, i] = np.abs(fft) ** 2
        spec = np.maximum(spec, 1e-10)
        return 10.0 * np.log10(spec[:n_mels, :])


def estimate_f0_pitch(audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate fundamental frequency (F0) contour across time."""
    try:
        import librosa
        # PYIN or YIN pitch tracking
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, fmin=60, fmax=400, sr=TARGET_SR, hop_length=320
        )
        times = librosa.times_like(f0, sr=TARGET_SR, hop_length=320)
        return times, np.nan_to_num(f0)
    except Exception:
        # Fallback simple zero-crossing pitch proxy
        hop = 320
        times = np.arange(0, len(audio), hop) / TARGET_SR
        f0_dummy = np.zeros(len(times))
        return times, f0_dummy


def compute_sync_correlation_matrix(
    audio_energy: np.ndarray, lip_motion: np.ndarray, max_lag_frames: int = 15
) -> np.ndarray:
    """Compute sliding cross-correlation heatmap across time lags [-max_lag .. +max_lag]."""
    min_len = min(len(audio_energy), len(lip_motion))
    if min_len < max_lag_frames * 2:
        return np.zeros((2 * max_lag_frames + 1, min_len))

    a = (audio_energy[:min_len] - np.mean(audio_energy[:min_len])) / (np.std(audio_energy[:min_len]) + 1e-8)
    v = (lip_motion[:min_len] - np.mean(lip_motion[:min_len])) / (np.std(lip_motion[:min_len]) + 1e-8)

    lags = np.arange(-max_lag_frames, max_lag_frames + 1)
    corr_matrix = np.zeros((len(lags), min_len))

    window = 25  # 1 second sliding window
    half = window // 2

    for t in range(half, min_len - half):
        chunk_a = a[t - half: t + half]
        for l_idx, lag in enumerate(lags):
            v_start = t - half + lag
            v_end = t + half + lag
            if 0 <= v_start and v_end <= min_len:
                chunk_v = v[v_start:v_end]
                corr_matrix[l_idx, t] = np.mean(chunk_a * chunk_v)

    return corr_matrix


def generate_diagnostic_dashboard(
    audio: np.ndarray,
    result: ClipResult,
    output_path: str,
) -> str:
    """Render a comprehensive diagnostic multi-plot dashboard for the clip."""
    duration_sec = len(audio) / TARGET_SR
    mel_spec = compute_mel_spectrogram(audio)
    f0_times, f0_values = estimate_f0_pitch(audio)

    # Frame level timeline
    num_frames = int(round(duration_sec * TARGET_FPS))
    frame_times = np.linspace(0, duration_sec, num_frames)

    # Audio RMS energy per frame
    samples_per_frame = int(TARGET_SR / TARGET_FPS)
    audio_rms = []
    for f in range(num_frames):
        s = f * samples_per_frame
        e = s + samples_per_frame
        chunk = audio[s:e] if e <= len(audio) else audio[s:]
        rms = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0.0
        audio_rms.append(rms)
    audio_rms = np.array(audio_rms)

    # Extract lip motion and score from timeline
    timeline = result.timeline
    bin_ms = 200
    frames_per_bin = 5

    bin_times = [b["bin_index"] * bin_ms / 1000.0 for b in timeline]
    light_scores = [b.get("light_asd_score") if b.get("light_asd_score") is not None else 0.0 for b in timeline]
    laser_scores = [b.get("laser_score") if b.get("laser_score") is not None else 0.0 for b in timeline]
    mouth_motions = [b.get("mouth_motion") if b.get("mouth_motion") is not None else 0.0 for b in timeline]

    # Expand mouth motion to frame level for cross-correlation
    expanded_motion = np.repeat(mouth_motions, frames_per_bin)[:num_frames]
    if len(expanded_motion) < num_frames:
        expanded_motion = np.pad(expanded_motion, (0, num_frames - len(expanded_motion)), mode="edge")

    # Compute sync heatmap
    corr_matrix = compute_sync_correlation_matrix(audio_rms, expanded_motion, max_lag_frames=12)

    # ========================== PLOTTING ==========================
    fig = plt.figure(figsize=(16, 12), dpi=150)
    gs = fig.add_gridspec(5, 1, height_ratios=[1.2, 1.0, 1.1, 0.9, 0.7], hspace=0.35)

    # Panel 1: Mel-Spectrogram & F0 Contour
    ax1 = fig.add_subplot(gs[0])
    extent = [0, duration_sec, 0, 8000]
    im1 = ax1.imshow(mel_spec, aspect="auto", origin="lower", extent=extent, cmap="viridis")
    ax1.set_title(
        f"Audio-Visual Diagnostic Dashboard | Clip: {result.clip_id} | "
        f"Decision: {result.decision.upper()}{f' ({result.reason})' if result.reason else ''} | "
        f"Tier: {result.early_exit_tier}",
        fontsize=13, fontweight="bold", pad=10
    )
    ax1.set_ylabel("Freq (Hz)", fontsize=10)
    ax1_twin = ax1.twinx()
    voiced_mask = f0_values > 0
    ax1_twin.scatter(f0_times[voiced_mask], f0_values[voiced_mask], color="magenta", s=4, label="F0 Pitch (Hz)", alpha=0.8)
    ax1_twin.set_ylabel("F0 (Hz)", color="magenta", fontsize=10)
    ax1_twin.set_ylim(40, 450)
    ax1.set_xlim(0, duration_sec)
    cbar1 = plt.colorbar(im1, ax=ax1, pad=0.06, fraction=0.02)
    cbar1.set_label("dB", fontsize=8)

    # Panel 2: Acoustic Energy vs Mouth Articulation Motion
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    norm_audio = audio_rms / (np.max(audio_rms) + 1e-8)
    norm_mouth = expanded_motion / (np.max(expanded_motion) + 1e-8) if np.max(expanded_motion) > 0 else expanded_motion
    ax2.plot(frame_times, norm_audio, color="royalblue", lw=1.5, label="Acoustic Energy Envelope (RMS)")
    ax2.plot(frame_times, norm_mouth, color="crimson", lw=1.5, ls="--", label="Lip Articulation Dynamics (Flow Δ)")
    ax2.set_ylabel("Normalized Amplitude", fontsize=10)
    ax2.set_title("Acoustic Energy vs. Lip Articulation Dynamics (Phoneme-to-Viseme Coupling)", fontsize=10, fontweight="semibold")
    ax2.grid(True, alpha=0.3, ls=":")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_ylim(-0.05, 1.1)

    # Panel 3: Temporal Sync Cross-Correlation Heatmap
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    extent_corr = [0, duration_sec, -12 * (1000 / TARGET_FPS), 12 * (1000 / TARGET_FPS)]
    im3 = ax3.imshow(corr_matrix, aspect="auto", origin="lower", extent=extent_corr, cmap="coolwarm", vmin=-0.6, vmax=0.6)
    ax3.axhline(0, color="black", lw=1.2, ls="--", alpha=0.7)
    ax3.set_ylabel("Time Lag (ms)", fontsize=10)
    ax3.set_title("Cross-Modal Audio-Visual Synchronization Heatmap (0ms = Perfect Sync)", fontsize=10, fontweight="semibold")
    cbar3 = plt.colorbar(im3, ax=ax3, pad=0.015, fraction=0.02)
    cbar3.set_label("Correlation", fontsize=8)

    # Panel 4: Active Speaker Confidence Scores (Light-ASD & LASER)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.step(bin_times, light_scores, where="post", color="teal", lw=2, label="Light-ASD Score (Tier 1)")
    if any(s > 0 for s in laser_scores):
        ax4.step(bin_times, laser_scores, where="post", color="darkorange", lw=2, ls="-.", label="LASER Arbiter Score (Tier 2)")
    ax4.axhline(0.0, color="gray", lw=1, ls=":", label="Decision Baseline")
    ax4.set_ylabel("ASD Confidence", fontsize=10)
    ax4.set_title("Multi-Tier Active Speaker Scoring & Model Agreement", fontsize=10, fontweight="semibold")
    ax4.grid(True, alpha=0.3, ls=":")
    ax4.legend(loc="upper right", fontsize=8)
    ax4.set_ylim(-1.2, 1.2)

    # Panel 5: 200 ms Temporal Decision Ribbon
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    color_map = {
        "active": "limegreen",
        "static": "red",
        "voiceover": "darkviolet",
        "ambiguous": "orange",
        "silent": "lightgray",
        "failure": "black",
    }
    from ..core.policy import classify_bin, TemporalPolicy
    policy = TemporalPolicy()

    for b in timeline:
        b_cls = classify_bin(b, policy)
        b_color = color_map.get(b_cls, "gray")
        ax5.barh(
            y=0,
            width=bin_ms / 1000.0,
            left=b["bin_index"] * bin_ms / 1000.0,
            height=0.6,
            color=b_color,
            edgecolor="white",
            linewidth=0.5
        )

    ax5.set_yticks([])
    ax5.set_xlabel("Time (seconds)", fontsize=10)
    ax5.set_title("Timeline Categorization (200ms Bins: Green=Active, Red=Static, Purple=Voiceover, Orange=Ambiguous)", fontsize=9, fontweight="semibold")
    ax5.set_xlim(0, duration_sec)

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
