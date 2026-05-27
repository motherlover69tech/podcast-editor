"""Tests for edit decision engine."""

from src.models import (
    AnalysisResult, CameraSource, FlagType, ProjectConfig, Segment, Word,
)
from src.decide import generate_edit_decisions, run_edit_pipeline


def make_config(speaker_map=None, wide="Wide", variety_min=10.0, variety_max=10.0):
    """Create a test ProjectConfig with 3 cameras."""
    return ProjectConfig(
        name="Test Podcast",
        sources=[
            CameraSource(name="CamA", file_path="/tmp/a.mp4", role="angle", label="Alice Cam"),
            CameraSource(name="CamB", file_path="/tmp/b.mp4", role="angle", label="Bob Cam"),
            CameraSource(name="Wide", file_path="/tmp/w.mp4", role="wide", label="Wide Shot"),
        ],
        speakers=speaker_map or {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"},
        wide_camera=wide,
        variety_threshold_min=variety_min,
        variety_threshold_max=variety_max,
        min_shot_duration=1.0,
        wide_break_min=2.0,
        wide_break_max=2.0,
    )


def test_simple_speaker_switch():
    """Two speakers alternating should produce camera switches."""
    words = []
    # Alice speaks for 5s
    for i in range(50):
        words.append(Word(word=f"word{i}", start=i*0.1, end=(i+1)*0.1, speaker="SPEAKER_00"))
    # Bob speaks for 5s
    for i in range(50):
        words.append(Word(word=f"word{i}", start=5.0+i*0.1, end=5.0+(i+1)*0.1, speaker="SPEAKER_01"))
    # Alice again for 3s
    for i in range(30):
        words.append(Word(word=f"word{i}", start=10.0+i*0.1, end=10.0+(i+1)*0.1, speaker="SPEAKER_00"))

    analysis = AnalysisResult(duration=13.0, words=words, segments=[])
    config = make_config()
    decisions = generate_edit_decisions(analysis, config)

    camera_cuts = [d for d in decisions if d.source]
    assert len(camera_cuts) >= 2, "Should have at least 2 cuts (A→B, B→A)"


def test_same_speaker_no_unnecessary_cuts():
    """Same speaker continuously should NOT produce cuts."""
    words = []
    for i in range(200):
        words.append(Word(word=f"word{i}", start=i*0.1, end=(i+1)*0.1, speaker="SPEAKER_00"))

    analysis = AnalysisResult(duration=20.0, words=words, segments=[])
    config = make_config(variety_min=999.0, variety_max=999.0)  # disable variety breaks
    decisions = generate_edit_decisions(analysis, config)

    camera_cuts = [d for d in decisions if d.source]
    assert len(camera_cuts) <= 1, "Should be at most 1 cut (initial) for same speaker"


def test_variety_break_long_monologue():
    """Same speaker for a long time should trigger a wide break."""
    words = []
    for i in range(500):  # 50 seconds of same speaker
        words.append(Word(word=f"word{i}", start=i*0.1, end=(i+1)*0.1, speaker="SPEAKER_00"))

    analysis = AnalysisResult(duration=50.0, words=words, segments=[])
    config = make_config(variety_min=10.0, variety_max=10.0)  # trigger after 10s
    decisions = generate_edit_decisions(analysis, config)

    # Should have at least a wide insert
    wide_cuts = [d for d in decisions if d.source == "Wide" and d.reason.value == "variety"]
    assert len(wide_cuts) >= 1, "Should have at least one variety wide cut"
