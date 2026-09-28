"""Core data structures and contracts for the Active Speaker module."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class FaceTrack:
    track_id: int
    frames: List[int]
    bboxes: List[BoundingBox]
    landmarks: Optional[List[Any]] = None  # 5-pt or 68-pt landmarks if available


@dataclass
class BinEvidence:
    """Evidence aggregated over a single temporal bin (default 200 ms)."""
    clip_id: str
    bin_index: int
    start_ms: int
    end_ms: int
    speech: bool = False
    face_visible: bool = False
    selected_track_id: int = -1
    face_track_count: int = 0
    mouth_motion: Optional[float] = None
    mouth_frozen: bool = False
    light_asd_score: Optional[float] = None
    laser_score: Optional[float] = None
    laser_requested: bool = False
    asd_disagreement: bool = False
    multiple_competing_faces: bool = False
    inference_failure: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClipResult:
    """Final output of the cascaded engine for one clip."""
    clip_id: str
    file_path: str
    decision: str            # "pass", "reject", "manual"
    reason: str              # "", "static", "voiceover", "ambiguous", "inference_failure"
    voiced_ms: int
    visible_active_speech_ratio: float
    unexplained_speech_ratio: float
    longest_unexplained_speech_ms: int
    static_ms: int
    voiceover_ms: int
    ambiguous_ms: int
    primary_speaker_track_id: int
    early_exit_tier: int     # 0, 1, or 2
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    roi_trajectory: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
