"""Pure temporal policy for active-speaker curation and validation.

Evaluates 200 ms evidence bins into conservative clip-level decisions.
Intentionally has NO torch or CV dependencies so it can be unit-tested anywhere.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Mapping, Tuple


DECISIONS = {"pass", "reject", "manual"}
REASONS = {"", "static", "voiceover", "ambiguous", "inference_failure"}


@dataclass(frozen=True)
class TemporalPolicy:
    bin_ms: int = 200
    min_contiguous_bad_ms: int = 800
    min_cumulative_bad_ms: int = 500
    min_bad_voiced_ratio: float = 0.20
    light_active_threshold: float = 0.0
    light_margin: float = 0.5
    laser_active_threshold: float = 0.5
    laser_margin: float = 0.15
    mouth_freeze_threshold: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def longest_true_run_ms(flags: Iterable[bool], bin_ms: int) -> int:
    longest = current = 0
    for flag in flags:
        if flag:
            current += bin_ms
            longest = max(longest, current)
        else:
            current = 0
    return longest


def laser_state(score: Any, policy: TemporalPolicy) -> str:
    if score is None:
        return "missing"
    try:
        val = float(score)
    except (TypeError, ValueError):
        return "missing"
    low = policy.laser_active_threshold - policy.laser_margin
    high = policy.laser_active_threshold + policy.laser_margin
    if val <= low:
        return "inactive"
    if val >= high:
        return "active"
    return "ambiguous"


def classify_bin(row: Mapping[str, Any], policy: TemporalPolicy) -> str:
    """Return active/static/voiceover/ambiguous/silent/failure for one bin."""
    if not _as_bool(row.get("speech", False)):
        return "silent"
    if _as_bool(row.get("inference_failure", False)):
        return "failure"

    visible = _as_bool(row.get("face_visible", False))
    frozen = _as_bool(row.get("mouth_frozen", False))
    if row.get("mouth_motion") not in (None, ""):
        try:
            frozen = float(row["mouth_motion"]) <= policy.mouth_freeze_threshold
        except (TypeError, ValueError):
            return "failure"

    light = row.get("light_asd_score")
    try:
        light = None if light in (None, "") else float(light)
    except (TypeError, ValueError):
        light = None
    laser = laser_state(row.get("laser_score"), policy)

    # Selective LASER requests: fail-closed boundary
    if _as_bool(row.get("laser_requested", False)) and laser == "missing":
        return "ambiguous"
    if _as_bool(row.get("asd_disagreement", False)):
        return "ambiguous"

    if not visible:
        return "voiceover"
    if light is None:
        return "failure"

    light_active = light >= policy.light_active_threshold + policy.light_margin
    light_inactive = light <= policy.light_active_threshold - policy.light_margin
    disagreement = _as_bool(row.get("multiple_competing_faces", False))

    if light_active and not frozen and not disagreement:
        return "active"

    # Static mouth plus confident non-speaking evidence -> static rejection
    if frozen and light_inactive:
        return "static"
    if frozen and laser == "inactive":
        return "static"

    # Visible faces but non-speaking -> voiceover
    if not frozen and light_inactive:
        return "voiceover"
    if not frozen and laser == "inactive":
        return "voiceover"

    # Positive LASER can rescue a near-threshold bin if no multi-face competition
    if laser == "active" and not disagreement:
        return "active"

    return "ambiguous"


def is_material(total_ms: int, longest_ms: int, voiced_ms: int,
                policy: TemporalPolicy) -> bool:
    return (
        longest_ms >= policy.min_contiguous_bad_ms
        or (
            total_ms >= policy.min_cumulative_bad_ms
            and voiced_ms > 0
            and total_ms / voiced_ms >= policy.min_bad_voiced_ratio
        )
    )


def summarize_timeline(rows: Iterable[Mapping[str, Any]], policy: TemporalPolicy) -> Dict[str, Any]:
    row_list = list(rows)
    classes = [classify_bin(r, policy) for r in row_list]
    voiced = [c != "silent" for c in classes]
    static = [c == "static" for c in classes]
    voiceover = [c == "voiceover" for c in classes]
    ambiguous = [c == "ambiguous" for c in classes]
    failure = [c == "failure" for c in classes]
    active = [c == "active" for c in classes]

    voiced_ms = sum(voiced) * policy.bin_ms
    static_ms = sum(static) * policy.bin_ms
    voiceover_ms = sum(voiceover) * policy.bin_ms
    unexplained = [a or b for a, b in zip(static, voiceover)]
    unexplained_ms = sum(unexplained) * policy.bin_ms
    longest_static = longest_true_run_ms(static, policy.bin_ms)
    longest_voiceover = longest_true_run_ms(voiceover, policy.bin_ms)
    longest_unexplained = longest_true_run_ms(unexplained, policy.bin_ms)
    ambiguous_ms = sum(ambiguous) * policy.bin_ms

    if any(failure):
        decision, reason = "manual", "inference_failure"
    else:
        static_mat = is_material(static_ms, longest_static, voiced_ms, policy)
        voiceover_mat = is_material(voiceover_ms, longest_voiceover, voiced_ms, policy)
        ambiguous_mat = is_material(
            ambiguous_ms, longest_true_run_ms(ambiguous, policy.bin_ms), voiced_ms, policy
        )

        if ambiguous_mat:
            decision, reason = "manual", "ambiguous"
        elif static_mat or voiceover_mat:
            decision = "reject"
            reason = "static" if longest_static >= longest_voiceover else "voiceover"
        else:
            decision, reason = "pass", ""

    disagreements = sum(
        _as_bool(r.get("asd_disagreement", False)) for r in row_list
        if _as_bool(r.get("speech", False))
    )

    return {
        "voiced_ms": voiced_ms,
        "visible_active_speech_ratio": (sum(active) * policy.bin_ms / voiced_ms) if voiced_ms else 0.0,
        "unexplained_speech_ratio": (unexplained_ms / voiced_ms) if voiced_ms else 0.0,
        "longest_unexplained_speech_ms": longest_unexplained,
        "static_speech_ratio": (static_ms / voiced_ms) if voiced_ms else 0.0,
        "asd_disagreement_ratio": (disagreements / sum(voiced)) if any(voiced) else 0.0,
        "temporal_decision": decision,
        "temporal_reason": reason,
        "static_ms": static_ms,
        "voiceover_ms": voiceover_ms,
        "ambiguous_ms": ambiguous_ms,
        "longest_static_ms": longest_static,
        "longest_voiceover_ms": longest_voiceover,
    }
