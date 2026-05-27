"""
Edit decision engine — converts analysis into a cut list.

Camera switching rules:
1. Speaker A talking → Camera A
2. Speaker B talking → Camera B
3. Both speakers (overlap) → Wide shot
4. Same speaker > variety_threshold → cut to wide for variety_break seconds, then back
5. Minimum shot duration enforced (no rapid cuts)
6. Transition dissolve on variety/overlap cuts
"""

from __future__ import annotations

from typing import Optional

from .models import (
    AnalysisResult, CameraSource, ClipExtract, CutReason, EditDecision,
    EditResult, FlagType, ProjectConfig, Transition,
)
from .analyze import get_speaker_timeline


def generate_edit_decisions(
    analysis: AnalysisResult,
    config: ProjectConfig,
) -> list[EditDecision]:
    """
    Generate the sequence of edit decisions from the speaker timeline.
    """
    timeline = get_speaker_timeline(analysis.words)
    if not timeline:
        return []

    # Build camera mapping
    cameras: dict[str, CameraSource] = {s.name: s for s in config.sources}
    speaker_to_camera: dict[str, str] = _map_speakers_to_cameras(config)
    wide_name = config.wide_camera or _find_wide_camera(config.sources)

    # First, subdivide long segments so we can inject variety breaks
    # Any segment longer than variety_threshold gets split into chunks
    subdivided: list[dict] = []
    for seg in timeline:
        dur = seg["end"] - seg["start"]
        if (dur > config.variety_threshold * 1.5
                and not seg.get("is_overlap")
                and wide_name
                and seg.get("dominant")):
            # Split into chunks: [speaker] [wide] [speaker] [wide] ...
            t = seg["start"]
            is_speaker = True
            while t < seg["end"]:
                if is_speaker:
                    chunk_dur = min(config.variety_threshold, seg["end"] - t)
                else:
                    chunk_dur = min(config.wide_break_duration, seg["end"] - t)
                if chunk_dur < 0.1:
                    break
                if is_speaker:
                    subdivided.append({
                        "start": t, "end": t + chunk_dur,
                        "speakers": seg["speakers"],
                        "dominant": seg["dominant"],
                        "is_overlap": False,
                    })
                else:
                    subdivided.append({
                        "start": t, "end": t + chunk_dur,
                        "speakers": seg["speakers"],
                        "dominant": seg["dominant"],
                        "is_overlap": True,  # force wide for variety
                        "_variety_break": True,
                    })
                t += chunk_dur
                is_speaker = not is_speaker
        else:
            subdivided.append(seg)

    timeline = subdivided

    decisions: list[EditDecision] = []
    current_source: Optional[str] = None
    same_source_since: float = 0.0

    for i, segment in enumerate(timeline):
        start = segment["start"]
        end = segment["end"]
        dominant = segment["dominant"]
        is_overlap = segment["is_overlap"]
        is_variety = segment.get("_variety_break", False)

        # ── Choose target camera ──────────────────────────────────────────
        target: str
        reason: CutReason

        if is_overlap:
            # Both speakers → wide (or variety break → wide)
            if wide_name:
                target = wide_name
                reason = CutReason.VARIETY if is_variety else CutReason.OVERLAP
            else:
                target = speaker_to_camera.get(dominant, speaker_to_camera.get(
                    next(iter(speaker_to_camera), ""), ""))
                reason = CutReason.OVERLAP
        elif not dominant or dominant not in speaker_to_camera:
            # Unknown or unmapped speaker → wide or first camera
            target = wide_name or (config.sources[0].name if config.sources else "")
            reason = CutReason.SPEAKER_CHANGE
        else:
            target = speaker_to_camera[dominant]
            reason = CutReason.SPEAKER_CHANGE

        # ── Enforce minimum shot duration ─────────────────────────────────
        if current_source == target:
            same_source_since += (end - start)
            continue  # no cut needed, same source

        # Don't cut if we just cut recently
        if decisions:
            last_cut = decisions[-1].time
            if start - last_cut < config.min_shot_duration:
                continue  # skip this cut, too close to last one

        # ── Emit the cut ──────────────────────────────────────────────────
        transition = Transition.DISSOLVE if reason != CutReason.SPEAKER_CHANGE else Transition.CUT
        decisions.append(EditDecision(
            time=start,
            source=target,
            transition=transition,
            transition_duration=config.dissolve_duration if transition == Transition.DISSOLVE else 0.0,
            reason=reason,
        ))

        current_source = target
        same_source_since = 0.0

    return decisions


def generate_clips(
    analysis: AnalysisResult,
    config: ProjectConfig,
    max_clips: int = 5,
) -> list[ClipExtract]:
    """
    Extract the best standalone clips for social media.
    Uses the pre-computed clip_score, filtered by duration.
    """
    candidates = [
        s for s in analysis.segments
        if (config.social_clip_min <= s.end - s.start <= config.social_clip_max)
        and s.clip_score > 0.4
    ]
    candidates.sort(key=lambda s: s.clip_score, reverse=True)

    clips: list[ClipExtract] = []
    for seg in candidates[:max_clips]:
        # Generate a title from the first sentence
        first_sentence = seg.transcript.split(".")[0].strip()
        title = first_sentence[:80]
        if len(first_sentence) > 80:
            title += "..."

        clips.append(ClipExtract(
            start=seg.start,
            end=seg.end,
            title=title,
            description=seg.topic or "",
            score=seg.clip_score,
        ))

    return clips


def generate_flags(analysis: AnalysisResult) -> list:
    """Collect all flagged segments for review."""
    return [s for s in analysis.segments if s.flags]


def run_edit_pipeline(
    analysis: AnalysisResult,
    config: ProjectConfig,
) -> EditResult:
    """Full edit pipeline: decisions + clips + flags."""
    decisions = generate_edit_decisions(analysis, config)
    clips = generate_clips(analysis, config)
    flags = generate_flags(analysis)

    # Add clip boundaries as markers
    for clip in clips:
        decisions.append(EditDecision(
            time=clip.start,
            source="",  # marker only
            reason=CutReason.TOPIC_BOUNDARY,
            note=f"[CLIP START] {clip.title}",
        ))
        decisions.append(EditDecision(
            time=clip.end,
            source="",
            reason=CutReason.TOPIC_BOUNDARY,
            note=f"[CLIP END] {clip.title}",
        ))

    # Sort all decisions by time
    decisions.sort(key=lambda d: d.time)

    return EditResult(
        decisions=decisions,
        clips=clips,
        flags=flags,
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _map_speakers_to_cameras(config: ProjectConfig) -> dict[str, str]:
    """
    Map diarization speaker labels to camera names.
    Uses config.speakers if provided, otherwise matches by position.
    """
    angle_sources = [s for s in config.sources if s.role == "angle"]
    wide_sources = [s for s in config.sources if s.role == "wide"]

    # If speakers are explicitly mapped in config
    if config.speakers:
        mapping: dict[str, str] = {}
        for i, (spk_label, _human_name) in enumerate(config.speakers.items()):
            if i < len(angle_sources):
                mapping[spk_label] = angle_sources[i].name
            elif wide_sources:
                mapping[spk_label] = wide_sources[0].name
        return mapping

    # Fallback: label cameras by diarization labels that appear in transcript
    # This requires analysis to have run first — handled at call time
    return {s.name.upper().replace(" ", "_"): s.name for s in angle_sources}


def _find_wide_camera(sources: list[CameraSource]) -> str:
    """Find the first wide-angle source, or fall back to first source."""
    for s in sources:
        if s.role == "wide":
            return s.name
    return sources[0].name if sources else ""
