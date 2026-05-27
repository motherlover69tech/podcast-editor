# Podcast Editor

AI-assisted podcast/video editor — drop your footage in, get a DaVinci Resolve timeline out.

## What it does

1. **Multi-camera sync** — cross-correlates audio tracks to align cameras (no timecode needed, clap detection as fallback)
2. **Transcription + diarization** — WhisperX with word-level timestamps and speaker labels
3. **Auto-cut timeline** — switches between cameras based on who's speaking, inserts wide shots for variety
4. **Social clip extraction** — scores segments for "clipability" and suggests the best ones
5. **Content flags** — detects repeated points, low-density sections, and areas to review
6. **DaVinci Resolve output** — FCPXML that imports directly into Resolve with markers for everything

## Quick Start

```bash
# Install
cd podcast-editor
pip install -e ".[gpu]"    # with GPU support (WhisperX)
# or
pip install -e ".[cpu]"    # CPU-only (faster-whisper)

# Initialize a project from a folder of video files
podcast-edit init /path/to/footage --name "My Podcast Ep 42"

# If you know which camera is the wide shot:
podcast-edit init /path/to/footage --name "Ep 42" --wide "Wide Shot"

# Run the full pipeline
podcast-edit run /path/to/footage

# Review the results
cat /path/to/footage/.podcast-editor/review_manifest.md

# Open in DaVinci Resolve
# File → Import → Timeline → FCPXML → <project>_auto_edit.fcpxml
```

## Workflow

```
┌──────────────────────────────────────────────────────────────┐
│  1. Drop video files in a folder                              │
│     - Camera_A.mp4, Camera_B.mp4, Wide_Shot.mp4               │
│                                                               │
│  2. podcast-edit init                                         │
│     - Auto-detects cameras and roles                          │
│     - Creates project config                                  │
│                                                               │
│  3. podcast-edit run                                          │
│     - Extracts + syncs audio                                  │
│     - Transcribes with speaker labels                         │
│     - Generates edit decisions                                │
│     - Exports FCPXML                                          │
│                                                               │
│  4. Open in DaVinci Resolve                                   │
│     - Full timeline with camera switches                      │
│     - Markers for flags and clip suggestions                  │
│                                                               │
│  5. Review + iterate                                          │
│     - Create review_actions.json with changes                 │
│     - podcast-edit review → regenerated FCPXML                │
└──────────────────────────────────────────────────────────────┘
```

## Camera Switching Rules

| Condition | Camera |
|---|---|
| Speaker A talking | Camera A |
| Speaker B talking | Camera B |
| Both speaking (overlap) | Wide shot |
| Same speaker > 15s | Cut to wide for 3s (variety break) |
| Cut too close to previous cut (< 2s) | Skip (minimum shot duration) |

## Interactive Review

After running the pipeline, edit `review_actions.json`:

```json
[
  {"action": "cut_section", "target_start": 120.0, "target_end": 135.0},
  {"action": "change_camera", "target_start": 300.0, "target_end": 320.0, "params": {"camera": "Wide"}},
  {"action": "add_marker", "target_start": 45.0, "params": {"note": "Audio glitch here"}}
]
```

Then re-run:

```bash
podcast-edit review /path/to/footage --actions review_actions.json
```

This generates a new `<project>_auto_edit_v2.fcpxml` with your changes applied.

## Commands

```
podcast-edit init <folder>          Initialize project
podcast-edit run <folder>           Full pipeline: sync → analyze → edit → FCPXML
podcast-edit review <folder>        Apply review actions, regenerate FCPXML
podcast-edit status <folder>        Show project state
podcast-edit suggest <folder> "..."  Semantic search the transcript
```

## Prerequisites

- **ffmpeg** — audio extraction and processing
- **Python 3.10+**

### For transcription (choose one):
- **WhisperX** (GPU, best quality) — `pip install whisperx`
- **faster-whisper** (CPU, good quality) — `pip install faster-whisper`

### Optional:
- **sentence-transformers** — topic segmentation, clip scoring, semantic search
- **scikit-learn** — similarity computations

## Project Structure

```
podcast-editor/
├── src/
│   ├── models.py       # Pydantic data models
│   ├── sync.py         # Multi-camera audio sync
│   ├── analyze.py      # Transcription, diarization, scoring
│   ├── decide.py       # Edit decision engine
│   ├── fcpxml_out.py   # FCPXML generation
│   ├── pipeline.py     # Orchestrator + review loop
│   └── cli.py          # CLI (typer)
├── tests/
│   ├── test_sync.py
│   └── test_decide.py
└── pyproject.toml
```

## Limitations & Next Steps

- **Resolve Python API** integration (for real-time control, requires Resolve Studio)
- **GPU-accelerated rendering** (NVENC/VAAPI through FFmpeg for preview renders)
- **LLM-powered topic labeling** (bulk GPT-4o-mini pass to name chapters)
- **LLM review assistant** — "make the cuts snappier", "emphasize Alice more"
- **Web UI** for visual review before Resolve import
