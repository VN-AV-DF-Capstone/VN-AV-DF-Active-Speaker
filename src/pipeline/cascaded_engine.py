"""Cascaded Active-Speaker & ROI Tracker (CAST) Pipeline Engine.

Implements the 3-Tier Cascade:
- Tier 0: Fast VAD (Silero) & Zero-Face Short-Circuit (CPU)
- Tier 1: High-Throughput Screening (Light-ASD + Mouth Flow)
- Tier 2: Selective Deep Arbiter (LoCoNet + LASER) for ambiguous bins
- Decision & ROI Handover: Pure Temporal Policy + Smoothed BBox Trajectory
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.core.data_types import BinEvidence, ClipResult
from src.core.policy import TemporalPolicy, summarize_timeline
from src.frontends.face_tracker import FaceTracker, align_face_and_mouth
from src.frontends.mouth_motion import compute_mouth_motion_series
from src.frontends.vad import SileroVAD
from src.models.laser_adapter import LaserAdapter
from src.models.light_asd import LightASD
from src.utils.audio_video import TARGET_FPS, TARGET_SR, decode_video_25fps, extract_audio_16k_mono
from .roi_extractor import ROIExtractor


MIN_LIGHT_TRACK_FRAMES = 5


class CascadedEngine:
    """Master active-speaker detection and ROI tracking engine."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # 1. Initialize Policy
        pol_cfg = config.get("policy", {})
        self.policy = TemporalPolicy(
            bin_ms=pol_cfg.get("bin_ms", 200),
            min_contiguous_bad_ms=pol_cfg.get("min_contiguous_bad_ms", 800),
            min_cumulative_bad_ms=pol_cfg.get("min_cumulative_bad_ms", 500),
            min_bad_voiced_ratio=pol_cfg.get("min_bad_voiced_ratio", 0.20),
            light_active_threshold=pol_cfg.get("light_active_threshold", 0.0),
            light_margin=pol_cfg.get("light_margin", 0.5),
            laser_active_threshold=pol_cfg.get("laser_active_threshold", 0.5),
            laser_margin=pol_cfg.get("laser_margin", 0.15),
            mouth_freeze_threshold=pol_cfg.get("mouth_freeze_threshold", 1.0),
        )

        # 2. Frontends
        vad_cfg = config.get("tier0_vad", {})
        self.vad = SileroVAD(vad_cfg.get("model_path"))

        trk_cfg = config.get("tracker", {})
        self.tracker = FaceTracker(
            backend=trk_cfg.get("backend", "auto"),
            detect_every=trk_cfg.get("detect_every", 2),
            det_size=trk_cfg.get("det_size", 640),
        )

        # 3. Models
        l_cfg = config.get("tier1_light_asd", {})
        self.light_asd = LightASD(
            weights_path=l_cfg.get("weights_path"),
            device=l_cfg.get("device", "auto"),
        )

        lz_cfg = config.get("tier2_laser", {})
        self.laser = LaserAdapter(
            weights_path=lz_cfg.get("weights_path"),
            device=lz_cfg.get("device", "auto"),
        )

        # 4. ROI Extractor
        roi_cfg = config.get("roi", {})
        self.roi_extractor = ROIExtractor(
            smoothing_window=roi_cfg.get("smoothing_window", 5),
            padding_scale=roi_cfg.get("padding_scale", 0.25),
        )

    def process_clip(self, video_path: str, clip_id: Optional[str] = None) -> ClipResult:
        if clip_id is None:
            clip_id = Path(video_path).stem

        # Decode media
        frames = decode_video_25fps(video_path)
        audio = extract_audio_16k_mono(video_path)

        bin_ms = self.policy.bin_ms
        bin_frames = int(TARGET_FPS * bin_ms / 1000)
        bin_count = max(1, math.ceil(len(frames) / bin_frames))

        # ==============================================================
        # TIER 0: Fast VAD & Zero-Face Short Circuit
        # ==============================================================
        speech_flags = self.vad.bins(audio, bin_count, bin_ms)
        has_speech = any(speech_flags)

        tracks = self.tracker.track(frames)
        has_faces = len(tracks) > 0

        # Short-circuit: Voiceover with zero faces
        if has_speech and not has_faces:
            timeline = [
                BinEvidence(
                    clip_id=clip_id,
                    bin_index=b,
                    start_ms=b * bin_ms,
                    end_ms=(b + 1) * bin_ms,
                    speech=speech_flags[b],
                    face_visible=False,
                ).to_dict()
                for b in range(bin_count)
            ]
            summary = summarize_timeline(timeline, self.policy)
            return ClipResult(
                clip_id=clip_id,
                file_path=video_path,
                decision=summary["temporal_decision"],
                reason=summary["temporal_reason"],
                voiced_ms=summary["voiced_ms"],
                visible_active_speech_ratio=summary["visible_active_speech_ratio"],
                unexplained_speech_ratio=summary["unexplained_speech_ratio"],
                longest_unexplained_speech_ms=summary["longest_unexplained_speech_ms"],
                static_ms=summary["static_ms"],
                voiceover_ms=summary["voiceover_ms"],
                ambiguous_ms=summary["ambiguous_ms"],
                primary_speaker_track_id=-1,
                early_exit_tier=0,
                timeline=timeline,
                roi_trajectory=[],
            )

        # Short-circuit: Complete silence
        if not has_speech:
            timeline = [
                BinEvidence(
                    clip_id=clip_id,
                    bin_index=b,
                    start_ms=b * bin_ms,
                    end_ms=(b + 1) * bin_ms,
                    speech=False,
                    face_visible=has_faces,
                ).to_dict()
                for b in range(bin_count)
            ]
            summary = summarize_timeline(timeline, self.policy)
            return ClipResult(
                clip_id=clip_id,
                file_path=video_path,
                decision=summary["temporal_decision"],
                reason=summary["temporal_reason"],
                voiced_ms=0,
                visible_active_speech_ratio=0.0,
                unexplained_speech_ratio=0.0,
                longest_unexplained_speech_ms=0,
                static_ms=0,
                voiceover_ms=0,
                ambiguous_ms=0,
                primary_speaker_track_id=tracks[0]["track_id"] if has_faces else -1,
                early_exit_tier=0,
                timeline=timeline,
                roi_trajectory=[],
            )

        # ==============================================================
        # TIER 1: High-Throughput Screening (Light-ASD + Mouth Flow)
        # ==============================================================
        per_frame_evidence: List[List[Dict[str, Any]]] = [[] for _ in frames]
        tracked_face_present = [False for _ in frames]
        prepared_mouths: Dict[int, List[np.ndarray]] = {}

        for track in tracks:
            faces, mouths, valid_f = [], [], []
            t_id = track["track_id"]
            kps_list = track.get("kps")

            for idx, f_id in enumerate(track["frames"]):
                f_int = int(f_id)
                tracked_face_present[f_int] = True
                kps = kps_list[idx] if kps_list is not None else None
                face_crop, mouth_crop = align_face_and_mouth(frames[f_int], track["bbox"][idx], kps)
                faces.append(face_crop)
                mouths.append(mouth_crop)
                valid_f.append(f_int)

            if len(faces) < MIN_LIGHT_TRACK_FRAMES:
                continue

            prepared_mouths[t_id] = mouths
            scores = self.light_asd.score(audio, np.asarray(faces), valid_f[0])
            motions = compute_mouth_motion_series(mouths)
            usable = min(len(scores), len(valid_f))

            for i in range(usable):
                per_frame_evidence[valid_f[i]].append({
                    "track_id": t_id,
                    "light_asd_score": float(scores[i]),
                    "mouth_motion": motions[i],
                    "mouth_crop": mouths[i],
                })

        # Aggregate frame-level evidence into 200 ms bins
        preliminary_timeline: List[Dict[str, Any]] = []
        needs_tier2 = False

        for b_idx in range(bin_count):
            start = b_idx * bin_frames
            end = min(len(frames), (b_idx + 1) * bin_frames)

            candidates: Dict[int, List[Dict[str, Any]]] = {}
            for f_ev in per_frame_evidence[start:end]:
                for item in f_ev:
                    candidates.setdefault(item["track_id"], []).append(item)

            track_evals = []
            for t_id, items in candidates.items():
                med_score = float(np.median([it["light_asd_score"] for it in items]))
                valid_motions = [it["mouth_motion"] for it in items if it["mouth_motion"] is not None]
                med_motion = float(np.median(valid_motions)) if valid_motions else None
                track_evals.append((med_score, t_id, med_motion))

            track_evals.sort(reverse=True, key=lambda x: x[0])
            best = track_evals[0] if track_evals else (None, -1, None)

            # Trigger conditions for Tier 2 Arbiter
            competing = len(track_evals) > 1 and track_evals[1][0] is not None and (
                best[0] is None or abs(best[0] - track_evals[1][0]) < self.policy.light_margin
            )
            near = best[0] is not None and abs(best[0] - self.policy.light_active_threshold) < self.policy.light_margin
            motion_conflict = (
                best[0] is not None and best[2] is not None
                and best[0] >= self.policy.light_active_threshold + self.policy.light_margin
                and best[2] <= self.policy.mouth_freeze_threshold
            )

            req_laser = bool(speech_flags[b_idx] and (near or competing or motion_conflict))
            if req_laser:
                needs_tier2 = True

            preliminary_timeline.append({
                "clip_id": clip_id,
                "bin_index": b_idx,
                "start_ms": b_idx * bin_ms,
                "end_ms": (b_idx + 1) * bin_ms,
                "speech": bool(speech_flags[b_idx]),
                "face_visible": bool(track_evals),
                "selected_track_id": int(best[1]),
                "face_track_count": len(track_evals),
                "mouth_motion": best[2],
                "mouth_frozen": best[2] is not None and best[2] <= self.policy.mouth_freeze_threshold,
                "light_asd_score": best[0],
                "laser_score": None,
                "laser_requested": req_laser,
                "asd_disagreement": False,
                "multiple_competing_faces": competing,
                "inference_failure": bool(
                    speech_flags[b_idx]
                    and any(tracked_face_present[start:end])
                    and not track_evals
                ),
            })

        # ==============================================================
        # TIER 2: Deep Arbiter (LoCoNet + LASER) - Selective Run
        # ==============================================================
        early_exit = 1
        if needs_tier2 and self.config.get("tier2_laser", {}).get("enabled", True):
            early_exit = 2
            samples_per_bin = int(TARGET_SR * bin_ms / 1000)

            for b_row in preliminary_timeline:
                if not b_row["laser_requested"]:
                    continue

                b_idx = b_row["bin_index"]
                s_sample = b_idx * samples_per_bin
                e_sample = (b_idx + 1) * samples_per_bin
                audio_seg = audio[s_sample:e_sample]

                # Collect candidate mouth series for this bin
                start_f = b_idx * bin_frames
                end_f = min(len(frames), (b_idx + 1) * bin_frames)
                cand_mouths = []
                for t_id, m_list in prepared_mouths.items():
                    if len(m_list) >= end_f:
                        cand_mouths.append(m_list[start_f:end_f])

                l_score = self.laser.score_bin(audio_seg, [], cand_mouths)
                b_row["laser_score"] = l_score

                # Check model disagreement
                best_light = b_row["light_asd_score"]
                if best_light is not None:
                    disagree = (
                        (best_light >= self.policy.light_active_threshold + self.policy.light_margin
                         and l_score <= self.policy.laser_active_threshold - self.policy.laser_margin)
                        or
                        (best_light <= self.policy.light_active_threshold - self.policy.light_margin
                         and l_score >= self.policy.laser_active_threshold + self.policy.laser_margin)
                    )
                    b_row["asd_disagreement"] = bool(disagree)

        # ==============================================================
        # Decision & Active ROI Handover
        # ==============================================================
        summary = summarize_timeline(preliminary_timeline, self.policy)

        # Determine dominant primary speaker track
        voiced_tracks = [
            b["selected_track_id"] for b in preliminary_timeline
            if b["speech"] and b["selected_track_id"] >= 0
        ]
        if voiced_tracks:
            vals, counts = np.unique(voiced_tracks, return_counts=True)
            primary_track_id = int(vals[np.argmax(counts)])
        elif tracks:
            primary_track_id = int(tracks[0]["track_id"])
        else:
            primary_track_id = -1

        # Smooth ROI trajectory
        h, w = frames[0].shape[:2]
        roi_traj = self.roi_extractor.extract_trajectory(
            tracks=tracks,
            primary_track_id=primary_track_id,
            active_bins=preliminary_timeline,
            frame_count=len(frames),
            video_shape=(h, w),
        )

        return ClipResult(
            clip_id=clip_id,
            file_path=video_path,
            decision=summary["temporal_decision"],
            reason=summary["temporal_reason"],
            voiced_ms=summary["voiced_ms"],
            visible_active_speech_ratio=summary["visible_active_speech_ratio"],
            unexplained_speech_ratio=summary["unexplained_speech_ratio"],
            longest_unexplained_speech_ms=summary["longest_unexplained_speech_ms"],
            static_ms=summary["static_ms"],
            voiceover_ms=summary["voiceover_ms"],
            ambiguous_ms=summary["ambiguous_ms"],
            primary_speaker_track_id=primary_track_id,
            early_exit_tier=early_exit,
            timeline=preliminary_timeline,
            roi_trajectory=roi_traj,
        )
