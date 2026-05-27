"""
Multi-camera sync via audio cross-correlation + clap detection.

Strategy:
1. Extract audio from each video source (16kHz mono WAV).
2. Cross-correlate each secondary source against the primary.
3. Also detect sharp transients (claps) as a fallback / verification.
4. Output per-camera offsets stored in the analysis result.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np


# ── Audio extraction ─────────────────────────────────────────────────────────

def extract_audio(video_path: str, sample_rate: int = 16000) -> tuple[np.ndarray, int]:
    """Extract mono 16kHz audio from a video file via ffmpeg → numpy."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-ac", "1", "-ar", str(sample_rate),
            "-f", "wav", "-v", "error",
            tmp.name,
        ]
        subprocess.run(cmd, check=True, timeout=120)
        import soundfile as sf
        audio, sr = sf.read(tmp.name)
        return audio.astype(np.float32), sr


# ── Cross-correlation sync ───────────────────────────────────────────────────

def cross_correlate_offset(
    ref: np.ndarray,
    target: np.ndarray,
    sample_rate: int,
    max_offset_sec: float = 30.0,
) -> float:
    """
    Find the time offset of `target` relative to `ref` using cross-correlation.
    Positive offset means target is delayed relative to ref (target starts later).
    Returns offset in seconds.
    """
    max_lag = int(max_offset_sec * sample_rate)

    # Use full signals for short files, chunks for long ones
    max_full_samples = 120 * sample_rate  # 2 minutes max for full correlation

    if len(ref) <= max_full_samples and len(target) <= max_full_samples * 2:
        ref_chunk = ref
        target_chunk = target
    else:
        # Take chunks from the first third (usually has intro/energy)
        chunk_start = len(ref) // 4
        window = min(60 * sample_rate, len(ref) // 2)
        ref_chunk = ref[chunk_start : chunk_start + window]
        target_chunk = target[chunk_start : chunk_start + window * 2]

    # Full cross-correlation: corr[k] = sum(target[n] * ref[n - (k - zero_lag)])
    corr = np.correlate(target_chunk, ref_chunk, mode="full")
    zero_lag_idx = len(ref_chunk) - 1

    # Search within ±max_lag around zero lag
    search_start = max(0, zero_lag_idx - max_lag)
    search_end = min(len(corr), zero_lag_idx + max_lag + 1)
    best_idx = int(np.argmax(np.abs(corr[search_start:search_end]))) + search_start

    # Positive lag = target is delayed (target[n] aligns with ref[n - lag])
    lag_samples = best_idx - zero_lag_idx
    offset_sec = lag_samples / sample_rate

    return offset_sec


# ── Clap detection ───────────────────────────────────────────────────────────

def detect_claps(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = -20.0,
    min_distance_ms: float = 100.0,
) -> list[float]:
    """
    Detect sharp transients (claps, slate hits) in audio.
    Returns list of timestamps in seconds.
    """
    from scipy.signal import find_peaks

    # Compute short-time energy envelope
    frame_len = int(0.01 * sample_rate)  # 10ms frames
    hop = frame_len // 2
    n_frames = (len(audio) - frame_len) // hop + 1

    energy = np.array([
        np.sqrt(np.mean(audio[i*hop : i*hop + frame_len] ** 2))
        for i in range(n_frames)
    ])

    # Normalize to dB
    energy_db = 20 * np.log10(np.maximum(energy, 1e-10))
    energy_db -= np.max(energy_db)  # peak is 0 dB

    # Find peaks (sharp transients)
    min_distance_frames = int(min_distance_ms / 10)
    peaks, _ = find_peaks(
        energy_db,
        height=threshold_db,
        distance=max(min_distance_frames, 1),
        prominence=6.0,  # must be sharp, not just loud
    )

    timestamps = [p * hop / sample_rate for p in peaks]
    return timestamps


def sync_by_claps(
    sources: list[tuple[str, np.ndarray]],
    sample_rate: int,
    tolerance_sec: float = 0.5,
) -> dict[str, float]:
    """
    Match clap transients across cameras to find offsets.
    Assumes at least one clap is present in all cameras.
    Returns {name: offset_seconds} relative to first source.
    """
    all_claps: list[list[float]] = []
    for _name, audio in sources:
        claps = detect_claps(audio, sample_rate)
        all_claps.append(claps)

    ref_claps = all_claps[0]
    offsets: dict[str, float] = {}

    for i, (name, _) in enumerate(sources):
        if i == 0:
            offsets[name] = 0.0
            continue

        # Find a clap in source i that matches a clap in ref within tolerance
        best_offset = None
        best_count = 0
        for t_ref in ref_claps[:10]:  # check first 10 claps
            for t_src in all_claps[i][:10]:
                offset = t_src - t_ref
                # Count how many other claps align with this offset
                count = 0
                for t2 in all_claps[i]:
                    for t1 in ref_claps:
                        if abs((t2 - t1) - offset) < tolerance_sec:
                            count += 1
                if count > best_count:
                    best_count = count
                    best_offset = offset

        offsets[name] = best_offset if best_offset is not None else 0.0

    return offsets


# ── Main sync entry point ────────────────────────────────────────────────────

def sync_cameras(
    sources: list[tuple[str, str]],  # [(name, file_path), ...]
    method: str = "auto",
) -> dict[str, float]:
    """
    Sync multiple camera sources. Returns {name: offset_seconds}.

    method: "cross_correlation" | "clap" | "auto" (tries both)
    """
    if len(sources) <= 1:
        return {sources[0][0]: 0.0}

    sr = 16000
    audios: list[tuple[str, np.ndarray]] = []

    for name, path in sources:
        audio, _ = extract_audio(path, sample_rate=sr)
        audios.append((name, audio))

    offsets: dict[str, float] = {}

    if method in ("cross_correlation", "auto"):
        ref_audio = audios[0][1]
        offsets[audios[0][0]] = 0.0
        for name, audio in audios[1:]:
            offset = cross_correlate_offset(ref_audio, audio, sr)
            offsets[name] = offset

    if method == "clap":
        offsets = sync_by_claps(audios, sr)

    # Clamp offsets to reasonable range
    for k in offsets:
        if abs(offsets[k]) > 120:
            offsets[k] = 0.0  # probably bad match

    return offsets
