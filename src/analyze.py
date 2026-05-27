"""
Analysis pipeline: transcription, diarization, topic segmentation,
clip scoring, repetition detection.

Default: faster-whisper (CPU) + pyannote diarization (CPU).
Same model quality as WhisperX — just slower without GPU.
WhisperX (NVIDIA GPU) is optional when CUDA is available.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .models import (
    AnalysisResult, AudioEvent, FlagType, Segment, Word,
)


# ── Environment ──────────────────────────────────────────────────────────────

HF_TOKEN = os.environ.get("HF_TOKEN", "")


# ── Transcription ────────────────────────────────────────────────────────────

def transcribe(
    audio_path: str,
    model: str = "large-v3",
    language: str = "en",
    device: str = "cpu",
    hf_token: str = "",
) -> dict:
    """
    Transcribe audio with word-level timestamps.

    Uses faster-whisper by default (CPU-friendly, same model quality as WhisperX).
    Falls back to WhisperX CLI if available and device="cuda" (NVIDIA only).
    """
    if device == "cuda":
        return _transcribe_whisperx(audio_path, model, language, hf_token or HF_TOKEN)
    else:
        return _transcribe_faster_whisper(audio_path, model, language)


def _transcribe_faster_whisper(
    audio_path: str,
    model: str = "large-v3",
    language: str = "en",
) -> dict:
    """
    Transcribe with faster-whisper (CTranslate2 backend, heavily CPU-optimized).
    Uses int8 quantization for speed. Same model weights = identical quality.
    """
    from faster_whisper import WhisperModel

    # Determine compute type: int8 for CPU, float16 for GPU
    compute = "int8"
    num_workers = min(4, os.cpu_count() or 4)
    model_instance = WhisperModel(
        model,
        device="cpu",
        compute_type=compute,
        num_workers=num_workers,
    )
    segments, info = model_instance.transcribe(
        audio_path,
        language=language,
        word_timestamps=True,
        vad_filter=True,          # skip silence, faster
        vad_parameters=dict(
            threshold=0.5,
            min_silence_duration_ms=500,
        ),
    )

    words = []
    text_segments = []
    for seg in segments:
        text_segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
        })
        if seg.words:
            for w in seg.words:
                words.append({
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                    "score": w.probability,
                })

    return {
        "segments": text_segments,
        "word_segments": words,
        "language": info.language,
    }


def _transcribe_whisperx(
    audio_path: str,
    model: str = "large-v3",
    language: str = "en",
    hf_token: str = "",
) -> dict:
    """
    Run WhisperX CLI (NVIDIA GPU required — CUDA only, no AMD/ROCm support).
    Falls back to faster-whisper if whisperx not found.
    """
    import shutil
    if not shutil.which("whisperx"):
        return _transcribe_faster_whisper(audio_path, model, language)

    cmd = [
        "whisperx", audio_path,
        "--model", model,
        "--language", language,
        "--device", "cuda",
        "--output_format", "json",
        "--output_dir", str(Path(audio_path).parent),
        "--compute_type", "float16",
    ]
    if hf_token:
        cmd += ["--hf_token", hf_token]

    subprocess.run(cmd, check=True, timeout=3600)

    json_path = Path(audio_path).with_suffix(".json")
    if not json_path.exists():
        raise FileNotFoundError(f"WhisperX output not found at {json_path}")

    with open(json_path) as f:
        return json.load(f)


# ── Diarization (CPU) ─────────────────────────────────────────────────────

def diarize_cpu(
    audio_path: str,
    hf_token: str = "",
    num_speakers: int = 2,
    min_speakers: int = 1,
    max_speakers: int = 5,
) -> list[dict]:
    """
    Speaker diarization using pyannote.audio on CPU.
    Same model quality as GPU — just slower.

    Returns list of {start, end, speaker} segments.
    Requires: pip install pyannote.audio
    Falls back gracefully if pyannote is not installed.
    """
    token = hf_token or HF_TOKEN
    if not token:
        # Without a HF token, we can't download pyannote models.
        # Return empty — caller handles this gracefully.
        return []

    try:
        from pyannote.audio import Pipeline
    except ImportError:
        return []

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token,
        )

        # Run diarization (CPU-friendly, just slower)
        diarization = pipeline(
            audio_path,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
            })
        return segments

    except Exception:
        return []


def apply_diarization(
    words: list[Word],
    diarization: list[dict],
) -> list[Word]:
    """
    Overlay diarization results onto transcribed words.
    Each word gets the speaker label of the diarization segment it falls within.
    """
    if not diarization:
        return words

    # Build fast lookup: for each time point, who's speaking
    diarization.sort(key=lambda d: d["start"])

    for w in words:
        mid = (w.start + w.end) / 2
        # Binary search for the diarization segment containing this midpoint
        for d in diarization:
            if d["start"] <= mid <= d["end"]:
                w.speaker = d["speaker"]
                break

    return words


def whisperx_result_to_words(raw: dict) -> list[Word]:
    """Convert WhisperX JSON output to our Word model list."""
    words: list[Word] = []
    for seg in raw.get("segments", []):
        for w in seg.get("words", []):
            word = w.get("word", "").strip()
            if not word:
                continue
            # WhisperX may or may not include speaker from diarization
            speaker = w.get("speaker", "")
            if not speaker:
                speaker = seg.get("speaker", "")
            words.append(Word(
                word=word,
                start=w.get("start", seg.get("start", 0)),
                end=w.get("end", seg.get("end", 0)),
                speaker=speaker,
                confidence=w.get("score", w.get("confidence", 0.0)),
            ))
    return words


# ── Audio event detection ────────────────────────────────────────────────────

def detect_audio_events(audio_path: str) -> list[AudioEvent]:
    """
    Detect laughter, applause, music, long silences.
    Uses YAMNet via tensorflow-hub if available, otherwise basic heuristics.
    """
    events: list[AudioEvent] = []

    # Basic silence detection using ffmpeg silencedetect filter
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", audio_path, "-af",
             "silencedetect=n=-40dB:d=2.0", "-f", "null", "-"],
            capture_output=True, text=True, timeout=120,
        )
        for line in result.stderr.splitlines():
            if "silence_start" in line:
                start = float(line.split("silence_start: ")[1].split(" ")[0])
                events.append(AudioEvent(type="silence", start=start, end=start + 2.0))
    except Exception:
        pass

    # TODO: YAMNet integration for laughter/applause/music
    return events


# ── Topic segmentation ───────────────────────────────────────────────────────

def segment_by_topics(
    words: list[Word],
    method: str = "semantic",
) -> list[Segment]:
    """
    Split the transcript into topic-based segments.

    method="semantic": uses sentence-transformers to find topic boundaries
    method="sentence": simple sentence-based segmentation
    """
    if not words:
        return []

    if method == "semantic":
        return _semantic_segmentation(words)
    else:
        return _sentence_segmentation(words)


def _sentence_segmentation(words: list[Word]) -> list[Segment]:
    """Simple segmentation: break on sentence-ending punctuation and long pauses."""
    segments: list[Segment] = []
    current_words: list[Word] = []
    seg_start = 0.0

    for i, w in enumerate(words):
        current_words.append(w)
        # Break on sentence-ending punctuation or >1.5s pause
        is_sentence_end = w.word.rstrip().endswith((".", "?", "!", ".\"", "?'"))
        pause = 0.0
        if i + 1 < len(words):
            pause = words[i + 1].start - w.end

        if is_sentence_end or pause > 1.5:
            if current_words:
                speaker = _dominant_speaker(current_words)
                text = " ".join(w.word for w in current_words)
                segments.append(Segment(
                    id=f"seg_{len(segments):03d}",
                    start=seg_start,
                    end=w.end,
                    speaker=speaker,
                    transcript=text,
                ))
            seg_start = words[i + 1].start if i + 1 < len(words) else w.end
            current_words = []

    # Last segment
    if current_words:
        speaker = _dominant_speaker(current_words)
        text = " ".join(w.word for w in current_words)
        segments.append(Segment(
            id=f"seg_{len(segments):03d}",
            start=seg_start,
            end=current_words[-1].end,
            speaker=speaker,
            transcript=text,
        ))

    return segments


def _semantic_segmentation(words: list[Word]) -> list[Segment]:
    """
    Segment using embedding-based topic boundary detection.
    Groups consecutive sentences by cosine similarity of their embeddings.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        # Fall back to sentence segmentation
        return _sentence_segmentation(words)

    # First split into sentences
    raw_segments = _sentence_segmentation(words)
    if len(raw_segments) <= 3:
        return raw_segments

    # Embed each segment
    texts = [s.transcript for s in raw_segments]
    embeddings = model.encode(texts)

    from sklearn.metrics.pairwise import cosine_similarity

    # Merge adjacent segments with high similarity
    merged: list[Segment] = []
    buffer_segs: list[Segment] = [raw_segments[0]]
    buffer_embs: list[np.ndarray] = [embeddings[0]]

    threshold = 0.6  # cosine similarity threshold for topic merge

    for i in range(1, len(raw_segments)):
        # Compare current buffer average to next segment
        buf_avg = np.mean(buffer_embs, axis=0).reshape(1, -1)
        sim = cosine_similarity(buf_avg, embeddings[i].reshape(1, -1))[0][0]

        if sim > threshold:
            buffer_segs.append(raw_segments[i])
            buffer_embs.append(embeddings[i])
        else:
            # Flush buffer as one segment
            merged.append(_merge_segments(buffer_segs, len(merged)))
            buffer_segs = [raw_segments[i]]
            buffer_embs = [embeddings[i]]

    if buffer_segs:
        merged.append(_merge_segments(buffer_segs, len(merged)))

    return merged


def _merge_segments(segs: list[Segment], idx: int) -> Segment:
    """Merge consecutive segments into one."""
    return Segment(
        id=f"seg_{idx:03d}",
        start=segs[0].start,
        end=segs[-1].end,
        speaker=_dominant_speaker([w for s in segs for w in _fake_words(s.transcript)]),
        transcript=" ".join(s.transcript for s in segs),
    )


def _dominant_speaker(words: list[Word]) -> str:
    """Return the most frequent speaker in a list of words."""
    from collections import Counter
    speakers = [w.speaker for w in words if w.speaker]
    if not speakers:
        return "UNKNOWN"
    return Counter(speakers).most_common(1)[0][0]


def _fake_words(text: str) -> list[Word]:
    """Create dummy Word objects from text — used for speaker counting."""
    return [Word(word=t, start=0, end=0, speaker="") for t in text.split()]


# ── Clip scoring ─────────────────────────────────────────────────────────────

def score_clips(segments: list[Segment], min_dur: float = 30.0, max_dur: float = 90.0) -> list[Segment]:
    """
    Score each segment for "clipability" — how good it would be as a standalone
    social media clip. Scores 0-1.
    """
    if not segments:
        return segments

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [s.transcript for s in segments]
        embeddings = model.encode(texts)
    except ImportError:
        return segments

    from sklearn.metrics.pairwise import cosine_similarity

    durations = [s.end - s.start for s in segments]
    max_dur_val = max(durations) if durations else 1

    for i, seg in enumerate(segments):
        dur = seg.end - seg.start

        # Duration fit: ideal is between min_dur and max_dur
        if dur < min_dur:
            dur_score = dur / min_dur
        elif dur > max_dur:
            dur_score = max(0, 1 - (dur - max_dur) / max_dur)
        else:
            dur_score = 1.0

        # Information density: ratio of unique words to segment length
        words = seg.transcript.split()
        unique_ratio = len(set(w.lower() for w in words)) / max(len(words), 1)

        # Self-containedness: low similarity to segments before/after
        boundary_sim = 0.0
        if i > 0:
            boundary_sim = max(boundary_sim,
                cosine_similarity(embeddings[i].reshape(1, -1), embeddings[i-1].reshape(1, -1))[0][0])
        if i < len(embeddings) - 1:
            boundary_sim = max(boundary_sim,
                cosine_similarity(embeddings[i].reshape(1, -1), embeddings[i+1].reshape(1, -1))[0][0])
        containment_score = 1.0 - boundary_sim  # low sim to neighbors = self-contained

        # Combine scores
        seg.clip_score = round(0.35 * dur_score + 0.30 * unique_ratio + 0.35 * containment_score, 3)

    return segments


# ── Repetition detection ─────────────────────────────────────────────────────

def detect_repetitions(segments: list[Segment], similarity_threshold: float = 0.85) -> list[Segment]:
    """
    Find segments that repeat the same point as an earlier segment.
    Marks them with FlagType.REPEAT.
    """
    if len(segments) < 2:
        return segments

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [s.transcript for s in segments]
        embeddings = model.encode(texts)
    except ImportError:
        return segments

    from sklearn.metrics.pairwise import cosine_similarity

    for i in range(len(segments)):
        for j in range(i + 1, min(i + 20, len(segments))):  # look ahead up to 20 segments
            sim = cosine_similarity(embeddings[i].reshape(1, -1), embeddings[j].reshape(1, -1))[0][0]
            if sim > similarity_threshold:
                segments[j].flags.append(FlagType.REPEAT)
                segments[j].repeats_segment = segments[i].id
                segments[j].repeat_similarity = round(sim, 3)
                break  # mark once per segment

    return segments


# ── Low-density detection ────────────────────────────────────────────────────

def detect_low_density(segments: list[Segment], min_duration_sec: float = 30.0) -> list[Segment]:
    """
    Flag segments that have low information density:
    - High filler word ratio
    - Very short LLM-summarizable content for their duration
    - Long segments of generic language
    """
    filler_words = {"um", "uh", "er", "ah", "like", "you know", "i mean", "sort of", "kind of"}

    for seg in segments:
        dur = seg.end - seg.start
        if dur < min_duration_sec:
            continue

        words = seg.transcript.lower().split()
        if len(words) < 10:
            continue

        # Filler ratio
        filler_count = sum(1 for w in words if w.rstrip(".,?!") in filler_words)
        filler_ratio = filler_count / len(words)

        # Unique word ratio
        unique_ratio = len(set(words)) / len(words)

        if filler_ratio > 0.08 or unique_ratio < 0.35:
            seg.flags.append(FlagType.LOW_DENSITY)
            if FlagType.REVIEW_RECOMMENDED not in seg.flags:
                seg.flags.append(FlagType.REVIEW_RECOMMENDED)

    return segments


# ── Main analysis entry point ────────────────────────────────────────────────

def analyze(
    audio_path: str,
    whisper_method: str = "faster-whisper",
    whisper_model: str = "large-v3",
    device: str = "cpu",
    hf_token: str = "",
    segment_method: str = "semantic",
    num_speakers: int = 2,
) -> AnalysisResult:
    """
    Run the full analysis pipeline on an audio file.

    Default: faster-whisper on CPU + pyannote diarization on CPU.
    Set device="cuda" to use WhisperX (NVIDIA GPU required).

    Returns AnalysisResult with words, segments, scores, and flags.
    """
    # 1. Transcription
    if whisper_method == "whisperx" or device == "cuda":
        raw = _transcribe_whisperx(audio_path, model=whisper_model, hf_token=hf_token or HF_TOKEN)
    else:
        raw = _transcribe_faster_whisper(audio_path, model=whisper_model)

    # 2. Parse into Word objects
    words = whisperx_result_to_words(raw)
    if not words:
        raise ValueError("No words found in transcription — check audio quality")

    duration = words[-1].end

    # 3. Diarization (CPU-friendly pyannote)
    diar_segments = diarize_cpu(audio_path, hf_token=hf_token or HF_TOKEN, num_speakers=num_speakers)
    if diar_segments:
        words = apply_diarization(words, diar_segments) 

    # 3. Audio event detection
    audio_events = detect_audio_events(audio_path)

    # 4. Topic segmentation
    segments = segment_by_topics(words, method=segment_method)

    # 5. Clip scoring
    segments = score_clips(segments)

    # 6. Repetition detection
    segments = detect_repetitions(segments)

    # 7. Low-density flags
    segments = detect_low_density(segments)

    # 8. Assign topics via LLM (optional — use a cheap model)
    # TODO: bulk topic labeling pass

    return AnalysisResult(
        duration=duration,
        words=words,
        segments=segments,
        audio_events=audio_events,
    )


# ── Speaker-activity timeline ────────────────────────────────────────────────

def get_speaker_timeline(words: list[Word]) -> list[dict]:
    """
    Build a timeline of who is speaking at each moment.
    Returns list of {start, end, speaker, is_overlap}.
    """
    if not words:
        return []

    speakers = sorted(set(w.speaker for w in words if w.speaker))
    if not speakers:
        return []

    # Build a per-frame speaker activity matrix
    # Use 100ms granularity
    frame_dur = 0.1
    total_duration = words[-1].end
    n_frames = int(total_duration / frame_dur) + 1

    activity = np.zeros((len(speakers), n_frames), dtype=bool)

    for w in words:
        if not w.speaker:
            continue
        spk_idx = speakers.index(w.speaker)
        start_frame = int(w.start / frame_dur)
        end_frame = min(int(w.end / frame_dur) + 1, n_frames)
        activity[spk_idx, start_frame:end_frame] = True

    # Identify contiguous segments of consistent speaker activity
    timeline: list[dict] = []
    t = 0.0
    current_active = tuple(activity[:, 0])

    for frame in range(1, n_frames):
        new_active = tuple(activity[:, frame])
        if new_active != current_active or frame == n_frames - 1:
            active_speakers = [speakers[i] for i, a in enumerate(current_active) if a]
            timeline.append({
                "start": t,
                "end": frame * frame_dur,
                "speakers": active_speakers,
                "dominant": active_speakers[0] if active_speakers else "",
                "is_overlap": len(active_speakers) > 1,
            })
            t = frame * frame_dur
            current_active = new_active

    return timeline
