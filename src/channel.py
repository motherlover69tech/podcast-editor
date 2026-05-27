"""
Stereo audio channel extraction and quality analysis.

Handles the common podcast scenario:
- Camera A records stereo: right channel = Peter, left channel = Mark
- Camera B records stereo: same mapping (or swapped)
- Extract the correct mono channel per speaker, ignore the other microphones
- Analyze audio quality to pick the best source per speaker
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np


def extract_channel(
    video_path: str,
    channel: str,  # "left", "right", "mono"
    output_path: str | None = None,
    sample_rate: int = 16000,
) -> str:
    """
    Extract a single mono channel from a stereo video/audio file.

    channel: "left" (channel 0), "right" (channel 1), "mono" (mixdown)
    Returns path to the extracted WAV file.
    """
    if output_path is None:
        output_path = tempfile.mktemp(suffix=f"_{channel}.wav")

    channel_map = {
        "left": "pan=mono|c0=c0",
        "right": "pan=mono|c0=c1",
        "mono": "pan=mono|c0=0.5*c0+0.5*c1",
    }

    pan_filter = channel_map.get(channel, channel_map["mono"])

    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-af", pan_filter,
        "-ac", "1", "-ar", str(sample_rate),
        "-vn",
        output_path,
    ], check=True, timeout=120, capture_output=True)

    return output_path


def analyze_channel_quality(audio_path: str) -> dict:
    """
    Analyze audio quality metrics for a channel.
    Returns {rms_db, snr_estimate, clipping_ratio, silence_ratio, speech_ratio}.
    Higher values (except silence_ratio and clipping_ratio) = better.
    """
    import soundfile as sf
    audio, sr = sf.read(audio_path)

    if len(audio) == 0:
        return {"rms_db": -999, "snr_estimate": 0, "clipping_ratio": 1.0,
                "silence_ratio": 1.0, "speech_ratio": 0.0, "quality_score": 0.0}

    audio = audio.astype(np.float32)

    # RMS energy
    rms = np.sqrt(np.mean(audio ** 2))
    rms_db = 20 * np.log10(max(rms, 1e-10))

    # Clipping detection (samples at or near max amplitude)
    clipping = np.mean(np.abs(audio) > 0.95)

    # Silence ratio (very quiet frames)
    frame_len = int(0.05 * sr)  # 50ms frames
    hop = frame_len
    n_frames = (len(audio) - frame_len) // hop + 1
    if n_frames <= 0:
        n_frames = 1

    frame_rms = np.array([
        np.sqrt(np.mean(audio[i*hop : i*hop + frame_len] ** 2))
        for i in range(n_frames)
    ])
    silence_ratio = np.mean(frame_rms < 0.001)

    # Speech-like energy ratio (frames in the middle energy range)
    # Too quiet = silence, too loud = noise/distortion
    frame_rms_db = 20 * np.log10(np.maximum(frame_rms, 1e-10))
    if frame_rms_db.size > 0:
        speech_mask = (frame_rms_db > -40) & (frame_rms_db < -5)
        speech_ratio = np.mean(speech_mask)
    else:
        speech_ratio = 0.0

    # SNR estimate: ratio of speech frames to silence frames
    speech_energy = np.mean(frame_rms[frame_rms > 0.001]) if np.any(frame_rms > 0.001) else 0
    noise_energy = np.mean(frame_rms[frame_rms <= 0.001]) if np.any(frame_rms <= 0.001) else 1e-10
    snr_estimate = 10 * np.log10(max(speech_energy / noise_energy, 1))

    # Composite quality score (0-1)
    quality_score = (
        0.30 * min(max(rms_db / -20, 0), 1) +      # decent level
        0.20 * (1 - clipping) +                       # no clipping
        0.25 * (1 - silence_ratio) +                  # not silent
        0.25 * speech_ratio                           # has speech-like content
    )

    return {
        "rms_db": round(rms_db, 1),
        "snr_estimate": round(snr_estimate, 1),
        "clipping_ratio": round(clipping, 3),
        "silence_ratio": round(silence_ratio, 3),
        "speech_ratio": round(speech_ratio, 3),
        "quality_score": round(quality_score, 3),
    }


def pick_best_audio_channel(
    sources: list[dict],  # [{file_path, channels: {speaker_name: "left"|"right"}}]
    work_dir: str,
) -> dict[str, str]:
    """
    For each speaker, extract their channel from each source,
    analyze quality, and pick the best source.

    Returns {speaker_name: path_to_best_audio_wav}
    """
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    # Collect all (speaker, source) audio files
    speaker_candidates: dict[str, list[tuple[str, str, float]]] = {}

    for src in sources:
        file_path = src["file_path"]
        channels = src.get("channels", {})

        for speaker, ch in channels.items():
            wav_path = work / f"{Path(file_path).stem}_{speaker}_{ch}.wav"
            extracted = extract_channel(file_path, ch, str(wav_path))
            quality = analyze_channel_quality(extracted)
            score = quality["quality_score"]

            if speaker not in speaker_candidates:
                speaker_candidates[speaker] = []
            speaker_candidates[speaker].append((extracted, src.get("label", file_path), score))

    # For each speaker, pick the best-quality source
    best_per_speaker: dict[str, str] = {}
    for speaker, candidates in speaker_candidates.items():
        candidates.sort(key=lambda x: x[2], reverse=True)
        best_per_speaker[speaker] = candidates[0][0]

    return best_per_speaker


def mix_speaker_audio(
    speaker_audio: dict[str, str],  # {speaker: wav_path}
    output_path: str,
) -> str:
    """
    Mix multiple mono speaker tracks into a single stereo WAV
    for WhisperX transcription (it handles multi-speaker better with
    all voices audible in one file).
    """
    # Load all audio files
    import soundfile as sf

    tracks: list[np.ndarray] = []
    max_len = 0

    for path in speaker_audio.values():
        audio, _ = sf.read(path)
        tracks.append(audio.astype(np.float32))
        max_len = max(max_len, len(audio))

    # Pad shorter tracks with zeros
    padded = []
    for track in tracks:
        if len(track) < max_len:
            p = np.zeros(max_len, dtype=np.float32)
            p[:len(track)] = track
            padded.append(p)
        else:
            padded.append(track)

    # Mix down to mono
    mixed = np.sum(padded, axis=0)
    # Normalize
    peak = np.max(np.abs(mixed))
    if peak > 0:
        mixed /= peak * 1.1  # leave headroom

    sf.write(output_path, mixed, 16000)
    return output_path
