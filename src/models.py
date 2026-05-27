"""
Data models for the podcast editor pipeline.
All modules communicate through these Pydantic models.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────────

class FlagType(str, Enum):
    REPEAT = "repeat"
    LOW_DENSITY = "low_density"
    SOCIAL_CLIP = "social_clip"
    REVIEW_RECOMMENDED = "review_recommended"
    AUDIO_ISSUE = "audio_issue"


class CutReason(str, Enum):
    SPEAKER_CHANGE = "speaker_change"
    OVERLAP = "overlap"           # both speaking → wide
    VARIETY = "variety"           # stuck on same person too long → wide
    TOPIC_BOUNDARY = "topic_boundary"


class Transition(str, Enum):
    CUT = "cut"
    DISSOLVE = "dissolve"


# ── Ingest / Sync ────────────────────────────────────────────────────────────

class CameraSource(BaseModel):
    """One video file from one camera angle."""
    name: str                           # "Camera A", "Wide", "Guest iPhone"
    file_path: str                      # absolute path
    label: str = ""                     # human-readable label (defaults to name)
    role: str = "angle"                 # "primary" | "wide" | "angle"
    offset_seconds: float = 0.0         # sync offset relative to primary
    channels: dict[str, str] = {}       # speaker name → channel ("left"/"right"/"mono")
    is_audio_source: bool = True        # use this file's audio for transcription
    quality_score: float = 0.0          # auto-computed audio quality


class ProjectConfig(BaseModel):
    """Top-level project configuration."""
    name: str
    sources: list[CameraSource]
    speakers: dict[str, str] = {}       # diarization label → human name, e.g. "SPEAKER_00": "Alice"
    wide_camera: str = ""               # name of the wide-shot source
    min_shot_duration: float = 2.0      # seconds — no cut shorter than this
    variety_threshold_min: float = 10.0 # seconds — same speaker > this → consider wide break
    variety_threshold_max: float = 25.0 # seconds — actual threshold is random(min, max)
    wide_break_min: float = 2.0         # seconds — minimum wide break duration
    wide_break_max: float = 5.0         # seconds — maximum wide break duration
    dissolve_duration: float = 0.5      # seconds — crossfade length (0 = hard cut)
    social_clip_min: float = 30.0       # seconds — min clip length for social
    social_clip_max: float = 90.0       # seconds — max clip length for social
    target_duration: float = 0.0        # seconds — time-constrained edit target (0 = full edit)
    exclude_themes: list[str] = []      # themes to drop when time-constrained

    @property
    def variety_threshold(self) -> float:
        """Randomized variety threshold to avoid predictability."""
        import random
        return random.uniform(self.variety_threshold_min, self.variety_threshold_max)

    @property
    def wide_break_duration(self) -> float:
        """Randomized wide break duration."""
        import random
        return random.uniform(self.wide_break_min, self.wide_break_max)


# ── Analysis ─────────────────────────────────────────────────────────────────

class Word(BaseModel):
    word: str
    start: float
    end: float
    speaker: str = ""
    confidence: float = 0.0


class Segment(BaseModel):
    """A semantic segment (topic chunk) of the recording."""
    id: str
    start: float
    end: float
    speaker: str                       # dominant speaker (or "MULTIPLE")
    transcript: str
    topic: str = ""
    clip_score: float = 0.0            # 0-1, how good a standalone clip this is
    flags: list[FlagType] = []
    repeats_segment: str = ""          # id of segment this one repeats
    repeat_similarity: float = 0.0     # cosine sim to repeated segment


class AnalysisResult(BaseModel):
    """Full analysis output for one recording session."""
    duration: float
    words: list[Word]
    segments: list[Segment]
    audio_events: list[AudioEvent] = []
    sync_offsets: dict[str, float] = {}  # camera name → offset seconds


class AudioEvent(BaseModel):
    type: str                           # "laughter", "applause", "music", "silence"
    start: float
    end: float
    confidence: float = 0.0


# ── Edit Decisions ───────────────────────────────────────────────────────────

class EditDecision(BaseModel):
    """One edit operation in the generated timeline."""
    time: float                         # when this edit happens (seconds)
    source: str                         # camera name to cut to
    transition: Transition = Transition.CUT
    transition_duration: float = 0.0
    reason: CutReason = CutReason.SPEAKER_CHANGE
    note: str = ""


class ClipExtract(BaseModel):
    """A suggested social-media clip extracted from the edit."""
    start: float
    end: float
    title: str
    description: str = ""
    score: float = 0.0                  # clipability score


class EditResult(BaseModel):
    """Full edit output."""
    decisions: list[EditDecision]
    clips: list[ClipExtract]
    flags: list[Segment]                # segments flagged for review


# ── Review ───────────────────────────────────────────────────────────────────

class ReviewAction(BaseModel):
    """User override of an edit decision."""
    action: str                          # "replace_segment", "cut_section", "keep_section",
                                         # "change_camera", "change_speaker_label", "add_marker"
    target_start: float
    target_end: float = 0.0
    params: dict = {}
