"""
Pipeline orchestrator — ties ingest → sync → analyze → decide → FCPXML together.
Supports the folder-drop workflow and interactive review.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import yaml

from .models import (
    AnalysisResult, CameraSource, EditResult, ProjectConfig, ReviewAction,
)
from .sync import sync_cameras
from .analyze import analyze

try:
    from .decide import run_edit_pipeline
    from .fcpxml_out import generate_fcpxml
except ImportError:
    pass


# ── Project file handling ────────────────────────────────────────────────────

def create_project(
    name: str,
    input_dir: str,
    output_dir: str = "",
    wide_camera: str = "",
    speakers: Optional[dict[str, str]] = None,
) -> ProjectConfig:
    """
    Scan an input directory for video files and create a project config.
    Attempts to auto-detect cameras and assign roles.
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Find video files
    video_exts = {".mp4", ".mov", ".mkv", ".mts", ".m2ts", ".avi", ".webm"}
    video_files = sorted([
        f for f in input_path.iterdir()
        if f.suffix.lower() in video_exts and f.is_file()
    ])

    if not video_files:
        raise ValueError(f"No video files found in {input_dir}")

    # Try to categorize cameras
    sources: list[CameraSource] = []
    for vf in video_files:
        name_lower = vf.stem.lower()
        role = "angle"

        if any(kw in name_lower for kw in ("wide", "full", "master", "room")):
            role = "wide"
            if not wide_camera:
                wide_camera = vf.stem
        elif any(kw in name_lower for kw in ("main", "primary", "host", "a-cam")):
            role = "primary"

        sources.append(CameraSource(
            name=vf.stem,
            file_path=str(vf.absolute()),
            label=vf.stem,
            role=role,
        ))

    config = ProjectConfig(
        name=name,
        sources=sources,
        speakers=speakers or {},
        wide_camera=wide_camera or _auto_wide(sources),
    )

    # Save config alongside the source files
    if output_dir:
        config_dir = Path(output_dir)
    else:
        config_dir = input_path / ".podcast-editor"
    config_dir.mkdir(parents=True, exist_ok=True)

    config_path = config_dir / "project.yaml"
    config_path.write_text(yaml.dump(config.model_dump(), default_flow_style=False))

    return config


def load_project(project_dir: str) -> tuple[ProjectConfig, Path]:
    """Load an existing project from its directory."""
    pd = Path(project_dir)
    config_path = pd / ".podcast-editor" / "project.yaml"

    if not config_path.exists():
        # Maybe the user gave us the .podcast-editor dir directly
        if (pd / "project.yaml").exists():
            config_path = pd / "project.yaml"
        else:
            raise FileNotFoundError(
                f"No project config found in {project_dir}. "
                f"Run 'podcast-edit init {project_dir}' first."
            )

    data = yaml.safe_load(config_path.read_text())
    config = ProjectConfig(**data)
    return config, config_path.parent


# ── Ingest pipeline ──────────────────────────────────────────────────────────

def ingest(config: ProjectConfig, project_dir: Path) -> Path:
    """
    Pre-process all sources:
    1. Extract audio from each video
    2. Sync cameras via cross-correlation
    3. Mix down primary audio for transcription
    Returns path to the mixed audio WAV.
    """
    audio_dir = project_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    # Extract audio from each source
    audio_paths: list[tuple[str, Path]] = []
    for src in config.sources:
        out_path = audio_dir / f"{src.name}.wav"
        if not out_path.exists():
            subprocess.run([
                "ffmpeg", "-y", "-i", src.file_path,
                "-ac", "1", "-ar", "16000", "-vn",
                str(out_path),
            ], check=True, timeout=120)
        audio_paths.append((src.name, str(out_path)))

    # Sync if multiple cameras
    if len(audio_paths) > 1:
        offsets = sync_cameras(
            [(name, p) for name, p in audio_paths],
            method="cross_correlation",
        )
        for src in config.sources:
            if src.name in offsets:
                src.offset_seconds = offsets[src.name]
        # Save updated config with offsets
        config_path = project_dir / "project.yaml"
        config_path.write_text(yaml.dump(config.model_dump(), default_flow_style=False))

    # Mix down primary audio for transcription
    primary = config.sources[0]
    mixed_path = audio_dir / "mixed_primary.wav"

    # If only one source, just copy it
    if len(audio_paths) == 1:
        # The primary is already extracted
        primary_wav = audio_dir / f"{primary.name}.wav"
        if primary_wav.exists():
            return primary_wav
        shutil.copy(audio_paths[0][1], mixed_path)
    else:
        # Mix all synced audio tracks into one stereo/mono track
        # Use the primary audio as-is (WhisperX handles mixed audio fine)
        primary_wav = audio_dir / f"{primary.name}.wav"
        if primary_wav.exists():
            return primary_wav
        shutil.copy(audio_paths[0][1], mixed_path)

    return mixed_path if mixed_path.exists() else audio_paths[0][1]


# ── Main pipeline ────────────────────────────────────────────────────────────

def run_pipeline(
    config: ProjectConfig,
    project_dir: Path,
    whisper_method: str = "whisperx",
    whisper_model: str = "large-v3",
    device: str = "cuda",
    hf_token: str = "",
    skip_ingest: bool = False,
    skip_analyze: bool = False,
) -> tuple[AnalysisResult, EditResult, Path]:
    """
    Run the full pipeline: ingest → analyze → decide → FCPXML.

    Returns (analysis, edit_result, fcpxml_path).
    """
    # 1. Ingest
    if not skip_ingest:
        audio_path = ingest(config, project_dir)
    else:
        audio_path = project_dir / "audio" / "mixed_primary.wav"
        if not audio_path.exists():
            audio_path = project_dir / "audio" / f"{config.sources[0].name}.wav"

    # 2. Analyze
    analysis_path = project_dir / "analysis.json"
    if not skip_analyze or not analysis_path.exists():
        analysis_result = analyze(
            str(audio_path),
            whisper_method=whisper_method,
            whisper_model=whisper_model,
            device=device,
            hf_token=hf_token,
        )
        # Save analysis for review/reuse
        analysis_path.write_text(analysis_result.model_dump_json(indent=2))
    else:
        analysis_result = AnalysisResult.model_validate_json(analysis_path.read_text())

    # 3. Decide
    edit_result = run_edit_pipeline(analysis_result, config)

    # Save edit decisions
    edit_path = project_dir / "edit_decisions.json"
    edit_path.write_text(edit_result.model_dump_json(indent=2))

    # 4. Generate FCPXML
    fcpxml_path = project_dir / f"{config.name}_auto_edit.fcpxml"
    generate_fcpxml(edit_result, config, str(fcpxml_path))

    # 5. Generate review manifest
    review_path = project_dir / "review_manifest.json"
    _generate_review_manifest(edit_result, analysis_result, config, review_path)

    return analysis_result, edit_result, fcpxml_path


# ── Interactive review ───────────────────────────────────────────────────────

def apply_review_actions(
    review_file: str,
    config: ProjectConfig,
    project_dir: Path,
) -> EditResult:
    """
    Apply user review actions from a JSON file and regenerate the FCPXML.

    The review file is a JSON array of ReviewAction objects.
    After applying, regenerates the FCPXML.
    """
    actions_data = json.loads(Path(review_file).read_text())
    actions = [ReviewAction(**a) for a in actions_data]

    # Load the current edit decisions
    edit_path = project_dir / "edit_decisions.json"
    edit_result = EditResult.model_validate_json(edit_path.read_text())

    # Apply each action
    for action in actions:
        edit_result = _apply_action(edit_result, action, config)

    # Save updated decisions
    edit_path.write_text(edit_result.model_dump_json(indent=2))

    # Regenerate FCPXML
    fcpxml_path = project_dir / f"{config.name}_auto_edit_v2.fcpxml"
    generate_fcpxml(edit_result, config, str(fcpxml_path))

    return edit_result


def _apply_action(
    edit_result: EditResult,
    action: ReviewAction,
    config: ProjectConfig,
) -> EditResult:
    """Apply a single review action to the edit result."""
    if action.action == "cut_section":
        # Remove all decisions in range, add a gap
        edit_result.decisions = [
            d for d in edit_result.decisions
            if not (action.target_start <= d.time <= action.target_end)
        ]
    elif action.action == "change_camera":
        # Override camera for a time range
        new_cam = action.params.get("camera", "")
        for d in edit_result.decisions:
            if action.target_start <= d.time <= action.target_end:
                d.source = new_cam
    elif action.action == "replace_segment":
        # Replace a clip with another
        pass  # TODO
    elif action.action == "keep_section":
        # Remove the "cut me" flag from a section
        pass  # TODO
    elif action.action == "add_marker":
        # Add a marker note
        from .models import EditDecision, CutReason
        edit_result.decisions.append(EditDecision(
            time=action.target_start,
            source="",
            reason=CutReason.TOPIC_BOUNDARY,
            note=action.params.get("note", "Review note"),
        ))
        edit_result.decisions.sort(key=lambda d: d.time)

    return edit_result


# ── Review manifest ──────────────────────────────────────────────────────────

def _generate_review_manifest(
    edit_result: EditResult,
    analysis: AnalysisResult,
    config: ProjectConfig,
    output_path: Path,
) -> None:
    """
    Generate a human-readable review manifest.
    This is what the user (or an LLM assistant) can read and respond to with changes.
    """
    lines = []
    lines.append(f"# Review Manifest — {config.name}")
    lines.append(f"Duration: {analysis.duration / 60:.1f} min")
    lines.append(f"Total cuts: {len([d for d in edit_result.decisions if d.source])}")
    lines.append(f"Flagged segments: {len(edit_result.flags)}")
    lines.append(f"Social clips: {len(edit_result.clips)}")
    lines.append("")

    # Camera assignments
    lines.append("## Camera Assignments")
    for src in config.sources:
        lines.append(f"- **{src.name}** ({src.role}) — {Path(src.file_path).name}  [offset: {src.offset_seconds:.3f}s]")
    lines.append("")

    # Edit summary
    lines.append("## Edit Summary")
    prev_cam = ""
    for d in edit_result.decisions:
        if d.source:
            dur = ""
            if prev_cam:
                dur = f"  (after {prev_cam})"
            lines.append(f"- **{_fmt_time(d.time)}** → {d.source}  [{d.reason.value}]{dur}  {d.note}")
            prev_cam = d.source

    # Flagged segments
    if edit_result.flags:
        lines.append("")
        lines.append("## ⚠ Flagged Segments")
        for seg in edit_result.flags:
            flags_str = ", ".join(f.value for f in seg.flags)
            lines.append(f"- **{_fmt_time(seg.start)}–{_fmt_time(seg.end)}** [{flags_str}]")
            lines.append(f"  \"{seg.transcript[:120]}...\"")
            if seg.repeats_segment:
                lines.append(f"  → Repeats: {seg.repeats_segment} (similarity: {seg.repeat_similarity:.2f})")

    # Social clips
    if edit_result.clips:
        lines.append("")
        lines.append("## 🎬 Social Media Clips")
        for clip in edit_result.clips:
            lines.append(f"- **{clip.title}**  [{_fmt_time(clip.start)}–{_fmt_time(clip.end)}]  score: {clip.score:.2f}")

    # How to provide review feedback
    lines.append("")
    lines.append("## How to Request Changes")
    lines.append("Create a `review_actions.json` file in this directory with an array of actions:")
    lines.append("```json")
    lines.append("""[
  {"action": "cut_section", "target_start": 120.0, "target_end": 135.0},
  {"action": "change_camera", "target_start": 300.0, "target_end": 320.0, "params": {"camera": "Wide"}},
  {"action": "add_marker", "target_start": 45.0, "params": {"note": "Check audio glitch here"}}
]""")
    lines.append("```")
    lines.append("")
    lines.append("Then run: `podcast-edit review --project-dir .`")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def _fmt_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _auto_wide(sources: list[CameraSource]) -> str:
    """Return the name of the wide-angle camera if found."""
    for s in sources:
        if s.role == "wide":
            return s.name
    return ""
