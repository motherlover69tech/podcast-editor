"""Tests for the sync module."""

import numpy as np
import pytest

from src.sync import cross_correlate_offset, detect_claps, sync_by_claps


def test_detect_claps_synthetic():
    """Detect synthetic claps in a generated signal."""
    sr = 16000
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = np.zeros_like(t)

    # Insert two sharp transients at 1.0s and 3.0s
    clap = np.exp(-np.linspace(0, 0.05, int(0.05 * sr)) * 100) * np.sin(
        2 * np.pi * 2000 * np.linspace(0, 0.05, int(0.05 * sr))
    )
    audio[int(1.0 * sr):int(1.0 * sr) + len(clap)] = clap
    audio[int(3.0 * sr):int(3.0 * sr) + len(clap)] = clap * 0.8

    claps = detect_claps(audio, sr, threshold_db=-30)
    assert len(claps) >= 2, f"Expected at least 2 claps, got {len(claps)}"


def test_cross_correlate_offset_known_shift():
    """Cross-correlation should recover a known time offset."""
    sr = 16000
    # Create a reference signal with some structure
    t = np.linspace(0, 5, sr * 5, endpoint=False)
    ref = np.sin(2 * np.pi * 440 * t) * np.exp(-((t - 2.5) ** 2) / 0.5)

    # Create a target that's delayed by 0.5 seconds
    delay_samples = int(0.5 * sr)
    target = np.zeros_like(ref)
    target[delay_samples:] = ref[:len(ref) - delay_samples]

    offset = cross_correlate_offset(ref, target, sr, max_offset_sec=2.0)
    assert abs(offset - 0.5) < 0.01, f"Expected ~0.5, got {offset:.4f}"


def test_sync_by_claps_matching():
    """Clap-based sync should find correct offsets."""
    sr = 16000
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    clap = np.exp(-np.linspace(0, 0.03, int(0.03 * sr)) * 150)
    clap *= np.sin(2 * np.pi * 3000 * np.linspace(0, 0.03, int(0.03 * sr)))

    # Camera A (ref): claps at 1.0, 2.5, 4.0
    a = np.zeros_like(t)
    for ts in [1.0, 2.5, 4.0]:
        a[int(ts * sr):int(ts * sr) + len(clap)] = clap

    # Camera B: claps at 1.2, 2.7, 4.2 (delayed by 0.2s)
    b = np.zeros_like(t)
    for ts in [1.2, 2.7, 4.2]:
        b[int(ts * sr):int(ts * sr) + len(clap)] = clap * 0.9

    offsets = sync_by_claps(
        [("A", a), ("B", b)], sr, tolerance_sec=0.3
    )

    assert "A" in offsets
    assert "B" in offsets
    assert abs(offsets["A"]) < 0.01
    assert abs(offsets["B"] - 0.2) < 0.15
