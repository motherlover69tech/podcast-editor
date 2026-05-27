"""
FastAPI web backend for the podcast editor.

Endpoints:
  POST  /api/projects/                    — create project, upload files
  GET   /api/projects/{id}                — project status
  POST  /api/projects/{id}/configure       — set channel mapping, params
  POST  /api/projects/{id}/analyze        — run audio quality analysis
  POST  /api/projects/{id}/run            — full pipeline
  GET   /api/projects/{id}/summary         — themes, titles, stats
  POST  /api/projects/{id}/time-edit       — time-constrained edit
  GET   /api/projects/{id}/download        — download FCPXML
  POST  /api/projects/{id}/review          — apply review actions
"""

from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .models import (
    AnalysisResult, CameraSource, EditResult, ProjectConfig,
)
from .channel import analyze_channel_quality, extract_channel, pick_best_audio_channel
from .themes import extract_themes, generate_summary, suggest_titles


app = FastAPI(title="Podcast Editor", version="0.2.0")


def main():
    """Entry point for `podcast-edit-web` command."""
    import uvicorn
    uvicorn.run("src.web:app", host="0.0.0.0", port=8890, reload=True)

# ── In-memory project store (replace with DB for production) ─────────────────

# ── Storage: configurable via PODCAST_EDITOR_DATA env var ─────────────────────
# On Unraid, set to: /mnt/user/media/podcast-editor-projects/

WORK_DIR = Path(os.environ.get("PODCAST_EDITOR_DATA", "./projects"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

projects: dict[str, dict] = {}


def _project_dir(project_id: str) -> Path:
    return WORK_DIR / project_id


def _get_config(project_id: str) -> ProjectConfig:
    cfg_path = _project_dir(project_id) / "project.yaml"
    if not cfg_path.exists():
        raise HTTPException(404, "Project not found")
    import yaml
    data = yaml.safe_load(cfg_path.read_text())
    return ProjectConfig(**data)


def _get_analysis(project_id: str) -> Optional[AnalysisResult]:
    path = _project_dir(project_id) / "analysis.json"
    if path.exists():
        return AnalysisResult.model_validate_json(path.read_text())
    return None


def _get_edit(project_id: str) -> Optional[EditResult]:
    path = _project_dir(project_id) / "edit_decisions.json"
    if path.exists():
        return EditResult.model_validate_json(path.read_text())
    return None


# ── Models ───────────────────────────────────────────────────────────────────

class ChannelConfig(BaseModel):
    file: str          # camera source name
    speaker: str       # "Peter", "Mark", etc.
    channel: str       # "left", "right", "mono"
    is_audio_source: bool = True


class ProjectCreateResponse(BaseModel):
    project_id: str
    sources: list[dict]


class RunRequest(BaseModel):
    whisper_method: str = "faster-whisper"
    whisper_model: str = "large-v3"
    device: str = "cpu"


class TimeEditRequest(BaseModel):
    target_duration: float       # seconds
    exclude_themes: list[str] = []


class ReviewRequest(BaseModel):
    actions: list[dict]


class ConfigUpdateRequest(BaseModel):
    variety_threshold_min: Optional[float] = None
    variety_threshold_max: Optional[float] = None
    wide_break_min: Optional[float] = None
    wide_break_max: Optional[float] = None
    min_shot_duration: Optional[float] = None
    dissolve_duration: Optional[float] = None
    wide_camera: Optional[str] = None
    channel_mappings: Optional[list[ChannelConfig]] = None


# ── API Endpoints ────────────────────────────────────────────────────────────

@app.post("/api/projects/", response_model=ProjectCreateResponse)
async def create_project(
    name: str = Form(...),
    files: list[UploadFile] = File(...),
):
    """Create a new project from uploaded video files."""
    project_id = uuid.uuid4().hex[:12]
    proj_dir = _project_dir(project_id)
    proj_dir.mkdir(parents=True, exist_ok=True)
    media_dir = proj_dir / "media"
    media_dir.mkdir(exist_ok=True)

    # Save uploaded files
    sources = []
    for f in files:
        safe_name = f.filename.replace("/", "_").replace("\\", "_")
        dest = media_dir / safe_name
        with open(dest, "wb") as out:
            while chunk := await f.read(1024 * 1024):
                out.write(chunk)

        # Guess role from filename
        name_lower = safe_name.lower()
        role = "angle"
        if any(kw in name_lower for kw in ("wide", "full", "master", "room")):
            role = "wide"
        elif any(kw in name_lower for kw in ("main", "primary", "host", "a-cam")):
            role = "primary"

        sources.append({
            "name": Path(safe_name).stem,
            "filename": safe_name,
            "role": role,
            "file_path": str(dest),
            "channels": {},
        })

    # Detect wide camera
    wide_cam = ""
    for s in sources:
        if s["role"] == "wide":
            wide_cam = s["name"]
            break

    config = ProjectConfig(
        name=name,
        sources=[CameraSource(
            name=s["name"],
            file_path=s["file_path"],
            label=s["name"],
            role=s["role"],
        ) for s in sources],
        wide_camera=wide_cam,
    )

    import yaml
    (proj_dir / "project.yaml").write_text(
        yaml.dump(config.model_dump(), default_flow_style=False)
    )

    projects[project_id] = {
        "status": "created",
        "name": name,
        "config": config,
        "sources": sources,
    }

    return ProjectCreateResponse(project_id=project_id, sources=sources)


@app.get("/api/projects/{project_id}")
async def get_project(project_id: str):
    """Get project status and config."""
    config = _get_config(project_id)
    analysis = _get_analysis(project_id)

    return {
        "project_id": project_id,
        "name": config.name,
        "sources": [s.model_dump() for s in config.sources],
        "config": {
            "variety_threshold_min": config.variety_threshold_min,
            "variety_threshold_max": config.variety_threshold_max,
            "wide_break_min": config.wide_break_min,
            "wide_break_max": config.wide_break_max,
            "min_shot_duration": config.min_shot_duration,
            "dissolve_duration": config.dissolve_duration,
            "wide_camera": config.wide_camera,
            "target_duration": config.target_duration,
        },
        "has_analysis": analysis is not None,
        "analysis": {
            "duration_min": round(analysis.duration / 60, 1),
            "word_count": len(analysis.words),
            "segment_count": len(analysis.segments),
        } if analysis else None,
    }


@app.post("/api/projects/{project_id}/configure")
async def configure_project(project_id: str, req: ConfigUpdateRequest):
    """Update project configuration — channel mapping, edit params."""
    config = _get_config(project_id)

    # Update scalar fields
    updates = {}
    for field in ["variety_threshold_min", "variety_threshold_max",
                  "wide_break_min", "wide_break_max", "min_shot_duration",
                  "dissolve_duration", "wide_camera"]:
        val = getattr(req, field, None)
        if val is not None:
            updates[field] = val

    if updates:
        for k, v in updates.items():
            setattr(config, k, v)

    # Apply channel mappings
    if req.channel_mappings:
        for mapping in req.channel_mappings:
            for src in config.sources:
                if src.name == mapping.file or Path(src.file_path).stem == mapping.file:
                    src.channels[mapping.speaker] = mapping.channel
                    src.is_audio_source = mapping.is_audio_source

    # Save
    import yaml
    (_project_dir(project_id) / "project.yaml").write_text(
        yaml.dump(config.model_dump(), default_flow_style=False)
    )

    return {"status": "ok", "message": "Configuration updated"}


@app.post("/api/projects/{project_id}/analyze")
async def analyze_audio(project_id: str):
    """
    Extract and analyze audio channels from all sources.
    Identifies which source has the best audio per speaker.
    """
    config = _get_config(project_id)
    proj_dir = _project_dir(project_id)

    # Build source list with channel configs
    source_list = []
    for src in config.sources:
        source_list.append({
            "file_path": src.file_path,
            "label": src.name,
            "channels": src.channels,
        })

    if not any(s["channels"] for s in source_list):
        return {"status": "warning", "message": "No channel mappings configured. Use /configure first."}

    # Pick best audio per speaker
    work_dir = proj_dir / "audio"
    best_per_speaker = pick_best_audio_channel(source_list, str(work_dir))

    # Analyze each extracted channel
    quality = {}
    for speaker, wav_path in best_per_speaker.items():
        quality[speaker] = analyze_channel_quality(wav_path)

    # Store results
    (proj_dir / "audio_quality.json").write_text(json.dumps(quality, indent=2))

    return {
        "status": "ok",
        "best_sources": best_per_speaker,
        "quality": quality,
    }


@app.post("/api/projects/{project_id}/run")
async def run_pipeline(project_id: str, req: RunRequest = RunRequest()):
    """
    Run the full editing pipeline. This is async — returns immediately
    and the frontend polls for completion.
    """
    import asyncio
    import threading

    config = _get_config(project_id)
    proj_dir = _project_dir(project_id)

    def _run():
        from .pipeline import run_pipeline as rp
        result = rp(
            config=config,
            project_dir=proj_dir,
            whisper_method=req.whisper_method,
            whisper_model=req.whisper_model,
            device=req.device,
        )
        # Store completion
        projects.setdefault(project_id, {})["status"] = "complete"
        projects.setdefault(project_id, {})["result"] = result

    projects.setdefault(project_id, {})["status"] = "running"

    # Run in background thread
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return {"status": "running", "message": "Pipeline started"}


@app.get("/api/projects/{project_id}/summary")
async def project_summary(project_id: str):
    """Get episode summary: themes, suggested titles, stats."""
    analysis = _get_analysis(project_id)
    if not analysis:
        raise HTTPException(404, "No analysis yet. Run the pipeline first.")

    # Generate summary
    summary = generate_summary(analysis.segments, analysis.duration)

    # Extract themes
    themes = extract_themes(analysis.segments)

    # Suggest titles for multiple styles
    titles = {}
    for style in ["balanced", "clickbait", "educational"]:
        titles[style] = suggest_titles(analysis.segments, themes, style=style)

    return {
        "summary": summary,
        "themes": {t: [s.model_dump() for s in segs] for t, segs in themes.items()},
        "suggested_titles": titles,
    }


@app.post("/api/projects/{project_id}/time-edit")
async def time_edit(project_id: str, req: TimeEditRequest):
    """
    Generate a time-constrained edit that fits within target_duration.
    Emits FCPXML decisions prioritizing best-scored segments by theme.
    """
    analysis = _get_analysis(project_id)
    if not analysis:
        raise HTTPException(404, "No analysis yet.")

    from .time_edit import time_constrained_edit, format_time_edit_summary

    edit = time_constrained_edit(
        segments=analysis.segments,
        target_duration=req.target_duration,
        exclude_themes=req.exclude_themes,
    )

    # Store for download
    proj_dir = _project_dir(project_id)
    (proj_dir / "time_edit.json").write_text(json.dumps({
        "selected": [s.model_dump() for s in edit.selected_segments],
        "dropped": [s.model_dump() for s in edit.dropped_segments],
        "dropped_themes": edit.dropped_themes,
        "total_duration": edit.total_duration,
        "fill_pct": edit.fill_pct,
    }, indent=2))

    # Generate FCPXML for this edit
    config = _get_config(project_id)
    temp_edit = EditResult(
        decisions=edit.decisions,
        clips=[],
        flags=[s for s in analysis.segments if s.flags
               and s in edit.selected_segments],
    )
    from .fcpxml_out import generate_fcpxml
    fcpxml_path = proj_dir / f"{config.name}_time_{int(req.target_duration/60)}min.fcpxml"
    generate_fcpxml(temp_edit, config, str(fcpxml_path))

    return {
        "summary": format_time_edit_summary(edit),
        "total_duration": edit.total_duration,
        "target_duration": req.target_duration,
        "fill_pct": edit.fill_pct,
        "selected_count": len(edit.selected_segments),
        "dropped_count": len(edit.dropped_segments),
        "dropped_themes": edit.dropped_themes,
        "fcpxml_path": str(fcpxml_path),
    }


@app.post("/api/projects/{project_id}/review")
async def apply_review(project_id: str, req: ReviewRequest):
    """Apply review actions and regenerate FCPXML."""
    config = _get_config(project_id)
    proj_dir = _project_dir(project_id)

    from .pipeline import apply_review_actions

    # Write actions to temp file
    actions_path = proj_dir / "review_actions.json"
    actions_path.write_text(json.dumps(req.actions))

    try:
        edit_result = apply_review_actions(str(actions_path), config, proj_dir)
        return {"status": "ok", "decisions_count": len(edit_result.decisions)}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/api/projects/{project_id}/download")
async def download_fcpxml(
    project_id: str,
    version: str = Query("auto_edit", description="'auto_edit', 'auto_edit_v2', or 'time_{n}min'"),
):
    """Download the FCPXML file."""
    config = _get_config(project_id)
    proj_dir = _project_dir(project_id)

    filename = f"{config.name}_{version}.fcpxml"
    path = proj_dir / filename

    if not path.exists():
        raise HTTPException(404, f"FCPXML not found: {filename}")

    return FileResponse(
        path,
        media_type="application/xml",
        filename=filename,
    )


@app.get("/api/projects/{project_id}/titles")
async def suggest_project_titles(
    project_id: str,
    style: str = Query("balanced"),
    n: int = Query(5),
):
    """Get suggested episode titles."""
    analysis = _get_analysis(project_id)
    if not analysis:
        raise HTTPException(404, "No analysis yet.")

    themes = extract_themes(analysis.segments)
    titles = suggest_titles(analysis.segments, themes, style=style, n=n)
    return {"titles": titles, "style": style}


# ── Preview Player Endpoints ─────────────────────────────────────────────────

@app.get("/api/projects/{project_id}/cutlist")
async def get_cutlist(project_id: str):
    """
    Return a clean JSON cut list for the preview player.
    Parses edit_decisions.json into {time, source, duration} entries.
    """
    edit = _get_edit(project_id)
    if not edit:
        raise HTTPException(404, "No edit decisions yet. Run the pipeline first.")

    config = _get_config(project_id)

    # Build source name → file URL mapping
    sources = {
        s.name: {
            "url": f"/api/projects/{project_id}/video/{s.name}",
            "label": s.label or s.name,
            "offset": s.offset_seconds,
            "role": s.role,
        }
        for s in config.sources
    }

    # Convert decisions to cut list
    camera_cuts = [d for d in edit.decisions if d.source and d.source != ""]
    if not camera_cuts:
        return {"cuts": [], "duration": 0, "sources": sources,
                "markers": [], "flags": []}

    cuts = []
    prev = camera_cuts[0]
    for curr in camera_cuts[1:]:
        start = prev.time
        dur = curr.time - start
        if dur > 0.05:  # skip negligible cuts
            cuts.append({
                "time": round(start, 3),
                "duration": round(dur, 3),
                "source": prev.source,
                "reason": prev.reason.value,
                "note": prev.note,
            })
        prev = curr

    # Final segment
    final_time = camera_cuts[-1].time
    total_duration = final_time + 5  # pad end

    # Extract markers (non-source decisions with notes)
    markers = [
        {"time": round(d.time, 3), "note": d.note}
        for d in edit.decisions
        if not d.source or d.source == ""
    ]

    # Flag segments
    flags = [
        {
            "start": round(f.start, 3),
            "end": round(f.end, 3),
            "flags": [fl.value for fl in f.flags],
            "text": f.transcript[:120],
        }
        for f in edit.flags
    ]

    return {
        "cuts": cuts,
        "duration": round(total_duration, 3),
        "sources": sources,
        "markers": markers,
        "flags": flags,
    }


@app.get("/api/projects/{project_id}/video/{source_name}")
async def serve_video(project_id: str, source_name: str):
    """
    Serve a video source file with HTTP range request support.
    This enables browser-native video seeking in the preview player.
    """
    from fastapi import Request
    from fastapi.responses import StreamingResponse

    config = _get_config(project_id)
    source = None
    for s in config.sources:
        if s.name == source_name:
            source = s
            break

    if not source:
        # Try matching by stem
        for s in config.sources:
            if Path(s.file_path).stem == source_name:
                source = s
                break

    if not source:
        raise HTTPException(404, f"Source not found: {source_name}")

    video_path = Path(source.file_path)
    if not video_path.exists():
        raise HTTPException(404, f"Video file not found: {video_path}")

    file_size = video_path.stat().st_size
    content_type = "video/mp4"

    # Detect mime type from extension
    ext = video_path.suffix.lower()
    mime_map = {".mp4": "video/mp4", ".mov": "video/quicktime",
                ".webm": "video/webm", ".mkv": "video/x-matroska",
                ".mts": "video/mp2t", ".avi": "video/x-msvideo"}
    content_type = mime_map.get(ext, "video/mp4")

    # Return full file (FastAPI FileResponse handles range automatically)
    return FileResponse(
        str(video_path),
        media_type=content_type,
        filename=video_path.name,
    )


@app.get("/api/projects/{project_id}/waveform")
async def get_waveform(project_id: str):
    """
    Return waveform data for the audio visualization overlay.
    Reads from analysis.json or computes from audio.
    """
    analysis = _get_analysis(project_id)
    if not analysis:
        raise HTTPException(404, "No analysis yet.")

    import numpy as np

    # Build waveform from word-level timestamps — energy peaks at speech
    # Downsample to ~200 data points for the canvas
    total_dur = analysis.duration
    num_points = 200
    bin_width = max(total_dur / num_points, 0.1)

    waveform = np.zeros(num_points)
    for w in analysis.words:
        mid = (w.start + w.end) / 2
        idx = int(mid / bin_width)
        if 0 <= idx < num_points:
            waveform[idx] += w.confidence

    # Normalize
    peak = np.max(waveform)
    if peak > 0:
        waveform = waveform / peak

    # Detect speaker presence per bin
    speakers = set(w.speaker for w in analysis.words if w.speaker)
    speaker_bands = {}
    if speakers:
        for spk in speakers:
            band = np.zeros(num_points)
            for w in analysis.words:
                if w.speaker == spk:
                    mid = (w.start + w.end) / 2
                    idx = int(mid / bin_width)
                    if 0 <= idx < num_points:
                        band[idx] += 1
            peak_spk = np.max(band)
            if peak_spk > 0:
                band = band / peak_spk
            speaker_bands[spk] = band.round(3).tolist()

    return {
        "duration": round(total_dur, 3),
        "num_points": num_points,
        "bin_width": round(bin_width, 3),
        "waveform": waveform.round(3).tolist(),
        "speaker_bands": speaker_bands,
    }


# ── LUT Management ──────────────────────────────────────────────────────────

@app.post("/api/projects/{project_id}/lut")
async def upload_lut(project_id: str, name: str = Form(...), file: UploadFile = File(...)):
    """Upload a .cube LUT file for the project."""
    proj_dir = _project_dir(project_id)
    luts_dir = proj_dir / "luts"
    luts_dir.mkdir(exist_ok=True)

    safe_name = name.replace("/", "_").replace("\\", "_")
    if not safe_name.endswith(".cube"):
        safe_name += ".cube"

    dest = luts_dir / safe_name
    with open(dest, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            out.write(chunk)

    # Parse .cube file to return metadata
    info = _parse_cube_file(dest)
    return {"name": safe_name, "size": dest.stat().st_size, **info}


@app.get("/api/projects/{project_id}/luts")
async def list_luts(project_id: str):
    """List available LUTs for the project."""
    proj_dir = _project_dir(project_id)
    luts_dir = proj_dir / "luts"
    if not luts_dir.exists():
        return {"luts": [], "default": ""}

    luts = []
    default_lut = (proj_dir / "default_lut.txt")
    default = default_lut.read_text().strip() if default_lut.exists() else ""

    for f in sorted(luts_dir.glob("*.cube")):
        info = _parse_cube_file(f)
        luts.append({"name": f.name, "size": f.stat().st_size, **info})

    return {"luts": luts, "default": default}


@app.delete("/api/projects/{project_id}/lut/{lut_name}")
async def delete_lut(project_id: str, lut_name: str):
    """Delete a LUT file."""
    proj_dir = _project_dir(project_id)
    path = proj_dir / "luts" / lut_name
    if path.exists():
        path.unlink()
    # Clear default if it was this one
    default_file = proj_dir / "default_lut.txt"
    if default_file.exists() and default_file.read_text().strip() == lut_name:
        default_file.unlink()
    return {"status": "deleted"}


@app.post("/api/projects/{project_id}/lut/default")
async def set_default_lut(project_id: str, name: str = Form(...)):
    """Set the default LUT for the project (auto-applied on preview load)."""
    proj_dir = _project_dir(project_id)
    (proj_dir / "default_lut.txt").write_text(name)
    return {"default": name}


@app.get("/api/projects/{project_id}/lut/{lut_name}")
async def serve_lut(project_id: str, lut_name: str):
    """Serve a .cube LUT file for the WebGL shader."""
    proj_dir = _project_dir(project_id)
    path = proj_dir / "luts" / lut_name
    if not path.exists():
        raise HTTPException(404, "LUT not found")
    return FileResponse(str(path), media_type="text/plain")


def _parse_cube_file(path: Path) -> dict:
    """Parse a .cube LUT file to extract title and size info."""
    try:
        text = path.read_text()
        title = ""
        size = 32
        for line in text.splitlines():
            if line.upper().startswith("TITLE"):
                title = line.split('"')[1] if '"' in line else line.split(maxsplit=1)[-1].strip()
            elif line.upper().startswith("LUT_3D_SIZE"):
                size = int(line.split()[-1])
        return {"title": title or path.stem, "lut_size": size}
    except Exception:
        return {"title": path.stem, "lut_size": 32}


# ── Frame.io-style Comments ─────────────────────────────────────────────────

class CommentIn(BaseModel):
    timestamp: float
    text: str
    author: str = "Anonymous"
    color: str = "#7c5cfc"       # marker color
    annotation: Optional[str] = None  # future: drawing data


class ReplyIn(BaseModel):
    text: str
    author: str = "Anonymous"


def _load_comments(project_id: str) -> list[dict]:
    path = _project_dir(project_id) / "comments.json"
    if path.exists():
        return json.loads(path.read_text())
    return []


def _save_comments(project_id: str, comments: list[dict]):
    (_project_dir(project_id) / "comments.json").write_text(json.dumps(comments, indent=2))


@app.get("/api/projects/{project_id}/comments")
async def get_comments(project_id: str):
    """Get all review comments, sorted by timestamp."""
    comments = _load_comments(project_id)
    comments.sort(key=lambda c: c.get("timestamp", 0))
    return {"comments": comments}


@app.post("/api/projects/{project_id}/comments")
async def add_comment(project_id: str, comment: CommentIn):
    """Add a timestamped review comment."""
    comments = _load_comments(project_id)
    entry = {
        "id": uuid.uuid4().hex[:8],
        "timestamp": comment.timestamp,
        "text": comment.text,
        "author": comment.author,
        "color": comment.color,
        "annotation": comment.annotation,
        "created_at": _now_iso(),
        "replies": [],
    }
    comments.append(entry)
    _save_comments(project_id, comments)
    return entry


@app.delete("/api/projects/{project_id}/comments/{comment_id}")
async def delete_comment(project_id: str, comment_id: str):
    comments = _load_comments(project_id)
    comments = [c for c in comments if c["id"] != comment_id]
    _save_comments(project_id, comments)
    return {"status": "deleted"}


@app.post("/api/projects/{project_id}/comments/{comment_id}/reply")
async def reply_comment(project_id: str, comment_id: str, reply: ReplyIn):
    comments = _load_comments(project_id)
    for c in comments:
        if c["id"] == comment_id:
            c.setdefault("replies", []).append({
                "id": uuid.uuid4().hex[:6],
                "text": reply.text,
                "author": reply.author,
                "created_at": _now_iso(),
            })
            _save_comments(project_id, comments)
            return c
    raise HTTPException(404, "Comment not found")


@app.post("/api/projects/{project_id}/comments/{comment_id}/resolve")
async def resolve_comment(project_id: str, comment_id: str):
    """Toggle resolved status on a comment."""
    comments = _load_comments(project_id)
    for c in comments:
        if c["id"] == comment_id:
            c["resolved"] = not c.get("resolved", False)
            _save_comments(project_id, comments)
            return c
    raise HTTPException(404, "Comment not found")


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# ── Static Frontend ──────────────────────────────────────────────────────────

@app.get("/")
async def index():
    """Serve the web frontend."""
    return HTMLResponse(_FRONTEND_HTML)


# Minimal embedded frontend (utility-first, mobile-safe, Peter-style)
_FRONTEND_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Podcast Editor</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; background: #0f0f11; color: #e1e1e6; line-height: 1.5; }
h1, h2, h3 { color: #f0f0f5; }
.container { max-width: 900px; margin: 0 auto; padding: 16px; }
.panel { background: #1a1a1f; border: 1px solid #2a2a2f; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
.panel h3 { margin-bottom: 8px; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px; color: #888; }
label { display: block; font-size: 13px; color: #999; margin: 8px 0 4px; }
input, select, textarea { width: 100%; padding: 8px 10px; background: #25252a; border: 1px solid #3a3a3f; border-radius: 4px; color: #e1e1e6; font-size: 14px; }
input:focus, select:focus { border-color: #7c5cfc; outline: none; }
input[type="range"] { accent-color: #7c5cfc; }
.row { display: flex; gap: 12px; flex-wrap: wrap; }
.col { flex: 1; min-width: 180px; }
.col2 { flex: 2; min-width: 250px; }
.btn { display: inline-block; padding: 8px 16px; border-radius: 4px; border: none; cursor: pointer; font-size: 14px; font-weight: 500; }
.btn-primary { background: #7c5cfc; color: #fff; }
.btn-primary:hover { background: #6a4ae8; }
.btn-secondary { background: #2a2a2f; color: #ccc; border: 1px solid #3a3a3f; }
.btn-secondary:hover { background: #333; }
.btn-danger { background: #e04444; color: #fff; }
.btn-danger:hover { background: #c33; }
.btn-sm { padding: 4px 10px; font-size: 12px; }
.btn-group { display: flex; gap: 8px; margin-top: 12px; }
table { width: 100%; border-collapse: collapse; margin-top: 8px; }
th, td { text-align: left; padding: 8px; border-bottom: 1px solid #2a2a2f; font-size: 13px; }
th { color: #888; font-weight: 500; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 500; }
.badge-green { background: #1a3a2a; color: #4ade80; }
.badge-yellow { background: #3a2a1a; color: #fbbf24; }
.badge-red { background: #3a1a1a; color: #f87171; }
.badge-purple { background: #2a1a3a; color: #a78bfa; }
.theme-bar { display: flex; height: 24px; border-radius: 4px; overflow: hidden; margin: 8px 0; }
.theme-segment { height: 100%; min-width: 2px; }
.theme-legend { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 6px; }
.theme-legend span { font-size: 11px; display: flex; align-items: center; gap: 4px; }
.theme-legend .swatch { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
.hidden { display: none !important; }
.status-msg { padding: 8px 12px; border-radius: 4px; font-size: 13px; margin-top: 8px; }
.status-info { background: #1a2a3a; color: #60a5fa; }
.status-success { background: #1a3a2a; color: #4ade80; }
.status-error { background: #3a1a1a; color: #f87171; }
.tabs { display: flex; gap: 2px; margin-bottom: 16px; border-bottom: 1px solid #2a2a2f; }
.tab { padding: 8px 16px; background: none; border: none; color: #888; cursor: pointer; font-size: 14px; border-bottom: 2px solid transparent; }
.tab.active { color: #e1e1e6; border-bottom-color: #7c5cfc; }
.tab-content { display: none; }
.tab-content.active { display: block; }
@media (max-width: 600px) { .row { flex-direction: column; } .col, .col2 { min-width: 100%; } .tabs { overflow-x: auto; } }
</style>
</head>
<body>
<div class="container">
  <h1 style="margin:16px 0 4px">🎙️ Podcast Editor</h1>
  <p style="color:#666;margin-bottom:16px;font-size:13px">Drop footage, configure, auto-edit → DaVinci Resolve</p>

  <div class="tabs">
    <button class="tab active" data-tab="upload">1. Upload</button>
    <button class="tab" data-tab="configure">2. Configure</button>
    <button class="tab" data-tab="run">3. Edit</button>
    <button class="tab" data-tab="results">4. Results</button>
    <button class="tab" data-tab="preview">5. Preview</button>
  </div>

  <!-- Tab 1: Upload -->
  <div id="tab-upload" class="tab-content active">
    <div class="panel">
      <h3>New Project</h3>
      <label>Project Name</label>
      <input type="text" id="project-name" placeholder="My Podcast Ep 42" />
      <label>Video Files</label>
      <input type="file" id="file-input" multiple accept="video/*" />
      <button class="btn btn-primary" onclick="createProject()" style="margin-top:12px">Create Project</button>
      <div id="upload-status"></div>
    </div>
  </div>

  <!-- Tab 2: Configure -->
  <div id="tab-configure" class="tab-content">
    <div class="panel">
      <h3>Camera Assignment</h3>
      <div id="source-list"></div>
    </div>
    <div class="panel">
      <h3>Edit Parameters</h3>
      <div class="row">
        <div class="col">
          <label>Min variety threshold (s)</label>
          <input type="number" id="var-min" value="10" step="1" min="3" />
        </div>
        <div class="col">
          <label>Max variety threshold (s)</label>
          <input type="number" id="var-max" value="25" step="1" min="5" />
        </div>
      </div>
      <p style="font-size:11px;color:#666;margin-top:4px">Cut to wide randomly between these limits to avoid predictability</p>
      <div class="row" style="margin-top:8px">
        <div class="col">
          <label>Min wide break (s)</label>
          <input type="number" id="wb-min" value="2" step="0.5" min="0.5" />
        </div>
        <div class="col">
          <label>Max wide break (s)</label>
          <input type="number" id="wb-max" value="5" step="0.5" min="1" />
        </div>
      </div>
      <div class="row" style="margin-top:8px">
        <div class="col">
          <label>Min shot duration (s)</label>
          <input type="number" id="min-shot" value="2" step="0.5" min="0.5" />
        </div>
        <div class="col">
          <label>Dissolve duration (s)</label>
          <input type="number" id="dissolve" value="0.5" step="0.1" min="0" />
        </div>
      </div>
      <button class="btn btn-primary" onclick="saveConfig()" style="margin-top:12px">Save Configuration</button>
      <div id="config-status"></div>
    </div>

    <div class="panel">
      <h3>Audio Channel Analysis</h3>
      <p style="font-size:12px;color:#888">Extract and analyze each speaker's audio channel to pick best quality.</p>
      <button class="btn btn-secondary" onclick="analyzeAudio()">Analyze Audio Quality</button>
      <div id="audio-quality"></div>
    </div>
  </div>

  <!-- Tab 3: Run Edit -->
  <div id="tab-run" class="tab-content">
    <div class="panel">
      <h3>Run Editing Pipeline</h3>
      <div class="row">
        <div class="col">
          <label>Whisper method</label>
          <select id="whisper-method">
            <option value="faster-whisper" selected>faster-whisper (CPU)</option>
            <option value="whisperx">WhisperX (NVIDIA GPU only)</option>
          </select>
        </div>
        <div class="col">
          <label>Model</label>
          <select id="whisper-model">
            <option value="large-v3">large-v3</option>
            <option value="medium">medium</option>
            <option value="small">small</option>
          </select>
        </div>
        <div class="col">
          <label>Device</label>
          <select id="whisper-device">
            <option value="cpu" selected>cpu</option>
            <option value="cuda">cuda (NVIDIA only)</option>
          </select>
        </div>
      </div>
      <button class="btn btn-primary" onclick="runPipeline()" style="margin-top:12px">Run Full Edit</button>
      <div id="run-status"></div>
    </div>

    <div class="panel">
      <h3>Time-Constrained Edit</h3>
      <div class="row">
        <div class="col">
          <label>Target duration (minutes)</label>
          <input type="number" id="target-minutes" value="10" step="1" min="1" max="120" />
        </div>
        <div class="col">
          <label>Exclude themes (comma-separated)</label>
          <input type="text" id="exclude-themes" placeholder="Weak Theme, Off Topic" />
        </div>
      </div>
      <button class="btn btn-secondary" onclick="runTimeEdit()" style="margin-top:12px">Generate Time Edit</button>
      <div id="time-edit-status"></div>
    </div>
  </div>

  <!-- Tab 4: Results -->
  <div id="tab-results" class="tab-content">
    <div class="panel">
      <h3>Episode Summary</h3>
      <div id="summary-content"><p style="color:#666">Run the pipeline to see results</p></div>
    </div>
    <div class="panel">
      <h3>Suggested Titles</h3>
      <select id="title-style" onchange="loadTitles()" style="width:auto;margin-bottom:8px">
        <option value="balanced">Balanced</option>
        <option value="clickbait">Clickbait</option>
        <option value="educational">Educational</option>
        <option value="interview">Interview</option>
      </select>
      <div id="titles-content"></div>
    </div>
    <div class="panel">
      <h3>Downloads</h3>
      <div id="download-links"></div>
    </div>
  </div>

  <!-- Tab 5: Preview Player -->
  <div id="tab-preview" class="tab-content">
    <div class="panel">
      <h3>Preview Auto-Edit</h3>
      <p style="font-size:12px;color:#666;margin-bottom:8px">Multi-source player — swaps between cameras per edit decisions</p>
      <div id="player-container" style="position:relative;background:#000;border-radius:4px;overflow:hidden;aspect-ratio:16/9;max-height:60vh">
        <!-- Video elements (hidden, one per source) -->
        <div id="video-stack" style="position:absolute;inset:0"></div>
        <!-- Cut indicator overlay -->
        <div id="cut-indicator" style="position:absolute;top:8px;right:8px;background:rgba(0,0,0,0.75);color:#fff;padding:4px 10px;border-radius:4px;font-size:13px;pointer-events:none;opacity:0;transition:opacity 0.15s">Cut</div>
        <!-- Center play overlay -->
        <div id="play-overlay" onclick="togglePlay()" style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;cursor:pointer;background:rgba(0,0,0,0.3)">
          <svg width="64" height="64" viewBox="0 0 24 24" fill="#fff" opacity="0.8"><polygon points="5,3 19,12 5,21"/></svg>
        </div>
      </div>
      <!-- Controls -->
      <div style="display:flex;align-items:center;gap:10px;margin-top:8px;flex-wrap:wrap">
        <button class="btn btn-primary btn-sm" onclick="togglePlay()" id="play-btn">▶ Play</button>
        <span id="time-display" style="font-size:13px;color:#ccc;font-variant-numeric:tabular-nums;min-width:100px">0:00 / 0:00</span>
        <input type="range" id="seek-bar" min="0" max="100" value="0" style="flex:1;min-width:120px"
               oninput="seekTo(this.value/100)" />
        <span id="current-source" class="badge badge-purple" style="min-width:70px;text-align:center">—</span>
        <!-- LUT selector -->
        <select id="lut-selector" style="width:auto;padding:4px 8px;font-size:12px" onchange="applyLUT(this.value)">
          <option value="">No LUT</option>
        </select>
        <button class="btn btn-secondary btn-sm" onclick="addCommentAtCurrentTime()" title="Add comment at current position">💬</button>
      </div>
      <!-- Timeline scrubber with cuts -->
      <canvas id="timeline-canvas" width="900" height="40" style="width:100%;height:40px;margin-top:4px;border-radius:4px;cursor:pointer"
              onclick="canvasSeek(event)"></canvas>
      <div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:6px">
        <label style="display:flex;align-items:center;gap:4px;font-size:12px;cursor:pointer">
          <input type="checkbox" id="show-waveform" checked onchange="drawTimeline()"> Show waveform
        </label>
        <label style="display:flex;align-items:center;gap:4px;font-size:12px;cursor:pointer">
          <input type="checkbox" id="show-flags" checked onchange="drawTimeline()"> Show flags
        </label>
        <label style="display:flex;align-items:center;gap:4px;font-size:12px;cursor:pointer">
          <input type="checkbox" id="show-speakers" checked onchange="drawTimeline()"> Show speakers
        </label>
      </div>
    </div>
    <div class="panel">
      <h3>Segment Info</h3>
      <div id="segment-info" style="font-size:12px;color:#888">Play to see current segment details</div>
    </div>
    <!-- LUT management -->
    <div class="panel">
      <h3>LUTs (Color Grade)</h3>
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
        <input type="file" id="lut-file-input" accept=".cube" style="width:auto;flex:1;min-width:150px" />
        <button class="btn btn-secondary btn-sm" onclick="uploadLUT()">Upload .cube</button>
      </div>
      <div id="lut-list" style="margin-top:8px;font-size:12px;color:#666"></div>
    </div>
    <!-- Comments panel -->
    <div class="panel">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <h3>Review Comments <span id="comment-count" style="font-size:12px;color:#666"></span></h3>
        <button class="btn btn-secondary btn-sm" onclick="addCommentAtCurrentTime()">+ Add at current time</button>
      </div>
      <div id="comments-list" style="max-height:300px;overflow-y:auto;margin-top:8px"></div>
      <!-- Add comment form (shown inline) -->
      <div id="comment-form" style="display:none;margin-top:8px;padding:8px;background:#25252a;border-radius:4px">
        <input type="text" id="comment-author" placeholder="Your name" style="margin-bottom:4px" value="Peter" />
        <textarea id="comment-text" placeholder="Note or feedback..." rows="2" style="margin-bottom:4px"></textarea>
        <div style="display:flex;gap:8px">
          <button class="btn btn-primary btn-sm" onclick="submitComment()">Save</button>
          <button class="btn btn-secondary btn-sm" onclick="cancelComment()">Cancel</button>
        </div>
        <input type="hidden" id="comment-timestamp" value="0" />
      </div>
    </div>
  </div>
</div>

<script>
let PROJECT_ID = '';
let SOURCES = [];

// ── Navigation ──────────────────────────────────────────────────────────────

document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
  });
});

// ── API Helpers ─────────────────────────────────────────────────────────────

async function api(url, opts = {}) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Request failed');
  }
  return res.json();
}

function setStatus(el, msg, cls = 'status-info') {
  el.innerHTML = `<div class="status-msg ${cls}">${msg}</div>`;
}

// ── Upload ──────────────────────────────────────────────────────────────────

async function createProject() {
  const name = document.getElementById('project-name').value || 'Untitled';
  const files = document.getElementById('file-input').files;
  const status = document.getElementById('upload-status');

  if (!files.length) { setStatus(status, 'Please select video files', 'status-error'); return; }

  const form = new FormData();
  form.append('name', name);
  for (const f of files) form.append('files', f);

  setStatus(status, 'Uploading...', 'status-info');

  try {
    const data = await api('/api/projects/', { method: 'POST', body: form });
    PROJECT_ID = data.project_id;
    SOURCES = data.sources;
    setStatus(status, `Project created: ${PROJECT_ID} (${SOURCES.length} files)`, 'status-success');
    renderSources();
    document.querySelector('[data-tab="configure"]').click();
  } catch (e) {
    setStatus(status, `Error: ${e.message}`, 'status-error');
  }
}

// ── Configure ───────────────────────────────────────────────────────────────

function renderSources() {
  const div = document.getElementById('source-list');
  div.innerHTML = SOURCES.map((s, i) => `
    <div style="border:1px solid #333;border-radius:4px;padding:10px;margin:8px 0">
      <strong>${s.name}</strong> <span class="badge badge-purple">${s.role}</span>
      <span style="color:#666;font-size:12px;margin-left:8px">${s.filename}</span>
      <div class="row" style="margin-top:6px">
        <div class="col">
          <label style="margin:0;font-size:11px">Audio source?</label>
          <select onchange="updateChannel(${i}, 'is_audio_source', this.value === 'true')">
            <option value="true" selected>Yes</option>
            <option value="false">No (ignore)</option>
          </select>
        </div>
        <div class="col">
          <label style="margin:0;font-size:11px">Speaker A channel</label>
          <select onchange="updateChannel(${i}, 'speaker_a', this.value)">
            <option value="">None</option>
            <option value="left">Left</option>
            <option value="right">Right</option>
            <option value="mono">Mono</option>
          </select>
        </div>
        <div class="col">
          <label style="margin:0;font-size:11px">Speaker B channel</label>
          <select onchange="updateChannel(${i}, 'speaker_b', this.value)">
            <option value="">None</option>
            <option value="left">Left</option>
            <option value="right">Right</option>
            <option value="mono">Mono</option>
          </select>
        </div>
      </div>
    </div>
  `).join('');
}

function updateChannel(idx, key, val) {
  if (!SOURCES[idx].channels) SOURCES[idx].channels = {};
  SOURCES[idx][key] = val;
}

async function saveConfig() {
  if (!PROJECT_ID) return;
  const status = document.getElementById('config-status');

  const channel_mappings = [];
  for (const src of SOURCES) {
    const ch = src.channels || {};
    if (ch.speaker_a) {
      channel_mappings.push({ file: src.name, speaker: 'Speaker A', channel: ch.speaker_a, is_audio_source: src.is_audio_source !== false });
    }
    if (ch.speaker_b) {
      channel_mappings.push({ file: src.name, speaker: 'Speaker B', channel: ch.speaker_b, is_audio_source: src.is_audio_source !== false });
    }
  }

  const body = {
    variety_threshold_min: parseFloat(document.getElementById('var-min').value),
    variety_threshold_max: parseFloat(document.getElementById('var-max').value),
    wide_break_min: parseFloat(document.getElementById('wb-min').value),
    wide_break_max: parseFloat(document.getElementById('wb-max').value),
    min_shot_duration: parseFloat(document.getElementById('min-shot').value),
    dissolve_duration: parseFloat(document.getElementById('dissolve').value),
    channel_mappings,
  };

  try {
    await api(`/api/projects/${PROJECT_ID}/configure`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    setStatus(status, 'Configuration saved', 'status-success');
  } catch (e) {
    setStatus(status, `Error: ${e.message}`, 'status-error');
  }
}

async function analyzeAudio() {
  if (!PROJECT_ID) return;
  const div = document.getElementById('audio-quality');
  div.innerHTML = '<div class="status-msg status-info">Analyzing audio channels...</div>';

  try {
    const data = await api(`/api/projects/${PROJECT_ID}/analyze`, { method: 'POST' });
    let html = '<table><tr><th>Speaker</th><th>Best Source</th><th>RMS dB</th><th>SNR</th><th>Quality</th></tr>';
    for (const [speaker, q] of Object.entries(data.quality)) {
      const cls = q.quality_score > 0.5 ? 'badge-green' : q.quality_score > 0.3 ? 'badge-yellow' : 'badge-red';
      html += `<tr>
        <td>${speaker}</td>
        <td style="font-size:12px;color:#888">${data.best_sources[speaker]?.split('/').pop() || '-'}</td>
        <td>${q.rms_db} dB</td>
        <td>${q.snr_estimate} dB</td>
        <td><span class="badge ${cls}">${q.quality_score}</span></td>
      </tr>`;
    }
    html += '</table>';
    div.innerHTML = html;
  } catch (e) {
    div.innerHTML = `<div class="status-msg status-error">${e.message}</div>`;
  }
}

// ── Run ─────────────────────────────────────────────────────────────────────

async function runPipeline() {
  if (!PROJECT_ID) return;
  const status = document.getElementById('run-status');
  setStatus(status, 'Pipeline running — this may take several minutes...', 'status-info');

  try {
    await api(`/api/projects/${PROJECT_ID}/run`, { method: 'POST' });
    setStatus(status, 'Pipeline started! Polling for completion...', 'status-info');

    // Poll for completion
    const poll = setInterval(async () => {
      try {
        const proj = await api(`/api/projects/${PROJECT_ID}`);
        if (proj.has_analysis) {
          setStatus(status, 'Edit complete!', 'status-success');
          clearInterval(poll);
          document.querySelector('[data-tab="results"]').click();
          loadResults();
        }
      } catch (e) {}
    }, 5000);

    setTimeout(() => clearInterval(poll), 600000); // 10 min timeout
  } catch (e) {
    setStatus(status, `Error: ${e.message}`, 'status-error');
  }
}

async function runTimeEdit() {
  if (!PROJECT_ID) return;
  const status = document.getElementById('time-edit-status');
  const minutes = parseFloat(document.getElementById('target-minutes').value);
  const exclude = document.getElementById('exclude-themes').value.split(',').map(s => s.trim()).filter(Boolean);

  setStatus(status, 'Generating time-constrained edit...', 'status-info');

  try {
    const data = await api(`/api/projects/${PROJECT_ID}/time-edit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target_duration: minutes * 60, exclude_themes: exclude }),
    });

    let html = `<pre style="font-size:12px;color:#ccc;white-space:pre-wrap">${data.summary}</pre>`;
    html += `<p style="margin-top:8px"><span class="badge badge-green">${data.fill_pct}% fill</span> ${data.selected_count} selected, ${data.dropped_count} dropped</p>`;
    if (data.dropped_themes.length) {
      html += `<p style="font-size:12px;color:#f87171">Dropped themes: ${data.dropped_themes.join(', ')}</p>`;
    }
    setStatus(status, html, 'status-success');
    loadDownloads();
  } catch (e) {
    setStatus(status, `Error: ${e.message} (run full pipeline first)`, 'status-error');
  }
}

// ── Results ─────────────────────────────────────────────────────────────────

async function loadResults() {
  loadSummary();
  loadTitles();
  loadDownloads();
}

async function loadSummary() {
  if (!PROJECT_ID) return;
  const div = document.getElementById('summary-content');
  try {
    const data = await api(`/api/projects/${PROJECT_ID}/summary`);
    const s = data.summary;

    let html = `<p><strong>Duration:</strong> ${s.total_duration_min} min | <strong>Segments:</strong> ${s.segment_count} | <strong>Themes:</strong> ${s.theme_count}</p>`;

    // Theme bar
    const colors = ['#7c5cfc','#4ade80','#fbbf24','#f87171','#60a5fa','#a78bfa','#34d399','#fb923c'];
    html += '<div class="theme-bar">';
    let ci = 0;
    for (const [theme, stats] of Object.entries(s.themes)) {
      html += `<div class="theme-segment" style="width:${stats.duration_pct}%;background:${colors[ci % colors.length]}" title="${theme}: ${stats.duration_pct}%"></div>`;
      ci++;
    }
    html += '</div>';

    html += '<div class="theme-legend">';
    ci = 0;
    for (const [theme, stats] of Object.entries(s.themes)) {
      html += `<span><span class="swatch" style="background:${colors[ci % colors.length]}"></span> ${theme} (${stats.duration_sec / 60:.1f}m, score ${stats.avg_clip_score.toFixed(2)})</span>`;
      ci++;
    }
    html += '</div>';

    html += `<p style="margin-top:8px;font-size:12px;color:#888">Speaking turns: ${s.speaking_turns} | Flagged: ${s.flagged_segments}</p>`;
    div.innerHTML = html;
  } catch (e) {
    div.innerHTML = `<p style="color:#f87171">${e.message}</p>`;
  }
}

async function loadTitles() {
  if (!PROJECT_ID) return;
  const style = document.getElementById('title-style').value;
  const div = document.getElementById('titles-content');
  try {
    const data = await api(`/api/projects/${PROJECT_ID}/titles?style=${style}&n=5`);
    let html = '';
    for (const t of data.titles) {
      html += `<div style="padding:8px;border:1px solid #2a2a2f;border-radius:4px;margin:4px 0">
        <strong>${t.title}</strong>
        <span class="badge badge-purple" style="margin-left:8px">${t.style}</span>
        <span style="color:#666;font-size:11px;margin-left:8px">score: ${t.score}</span>
      </div>`;
    }
    div.innerHTML = html || '<p style="color:#666">Run the pipeline to generate titles</p>';
  } catch (e) {
    div.innerHTML = `<p style="color:#f87171">${e.message}</p>`;
  }
}

function loadDownloads() {
  if (!PROJECT_ID) return;
  const div = document.getElementById('download-links');
  div.innerHTML = `
    <a href="/api/projects/${PROJECT_ID}/download?version=auto_edit" class="btn btn-primary btn-sm" style="margin:4px">Full Edit FCPXML</a>
    <a href="/api/projects/${PROJECT_ID}/download?version=auto_edit_v2" class="btn btn-secondary btn-sm" style="margin:4px">Reviewed FCPXML</a>
    <p style="font-size:11px;color:#666;margin-top:4px">After time edit: also check <code>time_Nmin.fcpxml</code></p>
  `;
}

// ── Preview Player ──────────────────────────────────────────────────────────

let CUTLIST = null;
let WAVEFORM = null;
let VIDEOS = {};      // source_name → <video> element
let activeSource = '';
let playing = false;
let playerInterval = null;

async function initPlayer() {
  if (!PROJECT_ID) return;
  const overlay = document.getElementById('play-overlay');

  try {
    const data = await api(`/api/projects/${PROJECT_ID}/cutlist`);
    CUTLIST = data;

    // Load waveform
    try { WAVEFORM = await api(`/api/projects/${PROJECT_ID}/waveform`); } catch(e) { WAVEFORM = null; }

    // Create video elements for each source
    const stack = document.getElementById('video-stack');
    stack.innerHTML = '';
    VIDEOS = {};

    for (const [name, info] of Object.entries(data.sources)) {
      const vid = document.createElement('video');
      vid.src = info.url;
      vid.preload = 'auto';
      vid.playsInline = true;
      vid.muted = false;  // unmuted for audio
      vid.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;object-fit:contain;display:none';
      vid.setAttribute('playsinline', '');
      stack.appendChild(vid);
      VIDEOS[name] = vid;

      // Show first frame when loaded
      vid.addEventListener('loadeddata', () => { vid.currentTime = 0.1; });
    }

    // Show first video
    if (data.cuts.length > 0) {
      switchTo(data.cuts[0].source);
      activeSource = data.cuts[0].source;
    } else if (Object.keys(VIDEOS).length > 0) {
      switchTo(Object.keys(VIDEOS)[0]);
      activeSource = Object.keys(VIDEOS)[0];
    }

    // Draw timeline
    drawTimeline();

    // Click to play
    overlay.style.display = 'flex';
    overlay.onclick = togglePlay;

    // Video ended
    for (const vid of Object.values(VIDEOS)) {
      vid.addEventListener('ended', () => {
        if (playing) { pause(); }
      });
    }

    updateSourceBadge();
  } catch (e) {
    document.getElementById('video-stack').innerHTML =
      `<p style="color:#f87171;padding:20px">No edit decisions yet. Run the pipeline first.</p>`;
  }
}

function switchTo(sourceName) {
  for (const [name, vid] of Object.entries(VIDEOS)) {
    vid.style.display = (name === sourceName) ? 'block' : 'none';
  }
  activeSource = sourceName;
  updateSourceBadge();
}

function togglePlay() {
  if (playing) { pause(); return; }
  if (!CUTLIST || CUTLIST.cuts.length === 0) return;

  // Sync all videos to the seek position
  const seekBar = document.getElementById('seek-bar');
  const seekFrac = parseFloat(seekBar.value) / 100;
  const seekTime = seekFrac * CUTLIST.duration;

  // Find which cut we're in
  const activeVid = VIDEOS[activeSource];
  if (activeVid) activeVid.currentTime = seekTime;

  for (const [name, vid] of Object.entries(VIDEOS)) {
    if (name !== activeSource) {
      // Pre-seek other videos too (offset adjusted)
      const src = CUTLIST.sources[name];
      vid.currentTime = Math.max(0, seekTime - (src?.offset || 0));
    }
  }

  // Play the active one
  if (activeVid) {
    activeVid.play().catch(e => console.warn('Play failed:', e));
    playing = true;
    document.getElementById('play-overlay').style.display = 'none';
    document.getElementById('play-btn').textContent = '⏸ Pause';
    startPlayerLoop();
  }
}

function pause() {
  for (const vid of Object.values(VIDEOS)) {
    vid.pause();
  }
  playing = false;
  document.getElementById('play-overlay').style.display = 'flex';
  document.getElementById('play-btn').textContent = '▶ Play';
  if (playerInterval) { clearInterval(playerInterval); playerInterval = null; }
}

function startPlayerLoop() {
  if (playerInterval) clearInterval(playerInterval);
  playerInterval = setInterval(() => {
    if (!playing || !CUTLIST) return;

    const activeVid = VIDEOS[activeSource];
    if (!activeVid) return;

    const t = activeVid.currentTime;

    // Check if we've crossed a cut point
    const nextCut = findNextCut(t);
    if (nextCut && Math.abs(t - nextCut.time) < 0.1) {
      // Switch camera
      activeSource = nextCut.source;
      const src = CUTLIST.sources[activeSource];
      const offsetTime = Math.max(0, t - (src?.offset || 0));
      switchTo(activeSource);
      const newVid = VIDEOS[activeSource];
      if (newVid) {
        newVid.currentTime = offsetTime;
        newVid.play().catch(() => {});
      }
      flashCutIndicator(nextCut);
    } else if (nextCut && t >= nextCut.time) {
      // We overshot — seek to the cut point
      activeSource = nextCut.source;
      const src = CUTLIST.sources[activeSource];
      const offsetTime = Math.max(0, nextCut.time - (src?.offset || 0));
      switchTo(activeSource);
      const newVid = VIDEOS[activeSource];
      if (newVid) {
        newVid.currentTime = offsetTime;
        newVid.play().catch(() => {});
      }
      flashCutIndicator(nextCut);
    }

    // Update UI
    updateTimeDisplay(t);
    updateSeekBar(t);
    updateSegmentInfo(t);
  }, 100);
}

function findNextCut(currentTime) {
  if (!CUTLIST) return null;
  for (const cut of CUTLIST.cuts) {
    if (cut.time > currentTime + 0.05 && cut.source !== activeSource) {
      return cut;
    }
  }
  return null;
}

function flashCutIndicator(cut) {
  const el = document.getElementById('cut-indicator');
  const reason = cut?.reason || 'cut';
  el.textContent = `▶ ${cut?.source || ''} [${reason}]`;
  el.style.opacity = '1';
  setTimeout(() => { el.style.opacity = '0'; }, 800);
}

function seekTo(fraction) {
  const time = fraction * (CUTLIST?.duration || 0);
  // Find which cut we're in at this time
  let source = activeSource;
  if (CUTLIST) {
    for (let i = CUTLIST.cuts.length - 1; i >= 0; i--) {
      if (time >= CUTLIST.cuts[i].time) {
        source = CUTLIST.cuts[i].source;
        break;
      }
    }
  }

  if (source !== activeSource) switchTo(source);

  const src = CUTLIST?.sources[source];
  const offsetTime = Math.max(0, time - (src?.offset || 0));
  for (const [name, vid] of Object.entries(VIDEOS)) {
    const s = CUTLIST?.sources[name];
    vid.currentTime = Math.max(0, time - (s?.offset || 0));
  }

  updateTimeDisplay(time);
  updateSeekBar(time);
  updateSegmentInfo(time);

  if (!playing) {
    // Show a frame at the seek position
    const activeVid = VIDEOS[activeSource];
    if (activeVid) activeVid.currentTime = offsetTime;
  }
}

function updateTimeDisplay(t) {
  const total = CUTLIST?.duration || 0;
  document.getElementById('time-display').textContent =
    `${fmtTime(t)} / ${fmtTime(total)}`;
}

function updateSeekBar(t) {
  const bar = document.getElementById('seek-bar');
  if (CUTLIST) bar.value = Math.round((t / CUTLIST.duration) * 100);
}

function updateSourceBadge() {
  const badge = document.getElementById('current-source');
  if (activeSource && CUTLIST) {
    const src = CUTLIST.sources[activeSource];
    badge.textContent = src?.label || activeSource;
    const cls = src?.role === 'wide' ? 'badge-green' : src?.role === 'primary' ? 'badge-purple' : 'badge-yellow';
    badge.className = `badge ${cls}`;
  }
}

function updateSegmentInfo(t) {
  if (!CUTLIST) return;
  const div = document.getElementById('segment-info');
  const currentCut = CUTLIST.cuts.find(c => t >= c.time && t < c.time + c.duration);
  if (currentCut) {
    div.innerHTML = `
      <strong>${currentCut.source}</strong> at ${fmtTime(currentCut.time)}
      <span style="margin-left:8px;font-size:11px">${currentCut.reason}</span>
      ${currentCut.note ? `<br><span style="color:#666">${currentCut.note}</span>` : ''}
    `;
  }
}

// ── Timeline Canvas ─────────────────────────────────────────────────────────

function drawTimeline() {
  const canvas = document.getElementById('timeline-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width = canvas.offsetWidth * (window.devicePixelRatio || 1);
  const H = canvas.height = 40 * (window.devicePixelRatio || 1);
  ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
  const w = canvas.offsetWidth;
  const h = 40;

  ctx.clearRect(0, 0, w, h);

  if (!CUTLIST || CUTLIST.duration === 0) {
    ctx.fillStyle = '#444';
    ctx.fillRect(0, h/2-1, w, 2);
    return;
  }

  const dur = CUTLIST.duration;
  const scale = w / dur;
  const colors = { angle: '#7c5cfc', wide: '#4ade80', primary: '#60a5fa' };

  // Draw cut segments as colored blocks
  for (const cut of CUTLIST.cuts) {
    const x = cut.time * scale;
    const segW = Math.max(1, cut.duration * scale);
    const src = CUTLIST.sources[cut.source];
    ctx.fillStyle = colors[src?.role] || '#7c5cfc';
    ctx.fillRect(x, 0, segW, h);
  }

  // Draw waveform overlay
  if (document.getElementById('show-waveform')?.checked && WAVEFORM?.waveform) {
    ctx.strokeStyle = 'rgba(255,255,255,0.4)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    const wf = WAVEFORM.waveform;
    for (let i = 0; i < wf.length; i++) {
      const x = (i / wf.length) * w;
      const y = h / 2 - (wf[i] * (h / 2 - 4));
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Speaker bands
    if (document.getElementById('show-speakers')?.checked && WAVEFORM.speaker_bands) {
      const spkColors = ['rgba(124,92,252,0.6)', 'rgba(250,176,5,0.6)', 'rgba(96,165,250,0.6)'];
      let si = 0;
      for (const [spk, band] of Object.entries(WAVEFORM.speaker_bands)) {
        ctx.fillStyle = spkColors[si % spkColors.length];
        for (let i = 0; i < band.length; i++) {
          if (band[i] > 0.1) {
            const x = (i / band.length) * w;
            ctx.fillRect(x, h - 8 - band[i] * 6, Math.max(1, w / band.length), band[i] * 6);
          }
        }
        si++;
      }
    }
  }

  // Flag markers
  if (document.getElementById('show-flags')?.checked && CUTLIST.flags?.length) {
    for (const flag of CUTLIST.flags) {
      const x1 = flag.start * scale;
      const x2 = flag.end * scale;
      ctx.fillStyle = 'rgba(248,113,113,0.4)';
      ctx.fillRect(x1, 0, Math.max(2, x2 - x1), 4);
    }
  }

  // Cut markers (thin lines)
  for (const cut of CUTLIST.cuts) {
    const x = cut.time * scale;
    ctx.strokeStyle = 'rgba(255,255,255,0.5)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }

  // Playhead
  const playhead = parseFloat(document.getElementById('seek-bar')?.value || 0) / 100;
  const px = playhead * w;
  ctx.fillStyle = '#fff';
  ctx.beginPath();
  ctx.moveTo(px, h);
  ctx.lineTo(px - 5, h - 8);
  ctx.lineTo(px + 5, h - 8);
  ctx.closePath();
  ctx.fill();
}

function canvasSeek(e) {
  const canvas = document.getElementById('timeline-canvas');
  const rect = canvas.getBoundingClientRect();
  const frac = (e.clientX - rect.left) / rect.width;
  seekTo(Math.max(0, Math.min(1, frac)));
}

// ── Tab activation — load player when Preview tab is clicked ────────────────

const origTabClick = document.querySelector('[data-tab="preview"]')?.onclick;
document.querySelector('[data-tab="preview"]')?.addEventListener('click', () => {
  if (CUTLIST === null) initPlayer();
});

// ── Keep timeline updated during playback ────────────────────────────────────

setInterval(() => {
  if (CUTLIST && document.getElementById('tab-preview').classList.contains('active')) {
    drawTimeline();
  }
}, 250);

function fmtTime(s) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, '0')}`;
}

// ── LUT (WebGL color grading) ──────────────────────────────────────────────

let LUT_DATA = {};     // name → {size, data: Float32Array}
let LUT_NAMES = [];
let currentLUT = '';
let glContext = null;
let lutTexture = null;
let lutCanvas = null;

async function loadLUTs() {
  if (!PROJECT_ID) return;
  try {
    const data = await api(`/api/projects/${PROJECT_ID}/luts`);
    const sel = document.getElementById('lut-selector');
    sel.innerHTML = '<option value="">No LUT</option>';
    LUT_NAMES = [];
    LUT_DATA = {};

    for (const lut of data.luts) {
      sel.innerHTML += `<option value="${lut.name}">${lut.title}</option>`;
      LUT_NAMES.push(lut.name);
      // Parse the .cube file into a 3D texture array
      try {
        const resp = await fetch(`/api/projects/${PROJECT_ID}/lut/${lut.name}`);
        const text = await resp.text();
        LUT_DATA[lut.name] = parseCubeFile(text, lut.lut_size);
      } catch(e) { console.warn('LUT parse failed:', lut.name, e); }
    }

    // Set default
    if (data.default && LUT_NAMES.includes(data.default)) {
      sel.value = data.default;
      applyLUT(data.default);
    }

    renderLUTList(data.luts, data.default);
  } catch(e) { /* no LUTs yet */ }
}

function parseCubeFile(text, size) {
  // Parse .cube LUT into a flat Float32Array of RGB values
  // .cube format: header lines, then R G B values, one per line
  const lines = text.split('\n');
  const values = [];
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#') || trimmed.startsWith('TITLE')
        || trimmed.startsWith('LUT_3D_SIZE') || trimmed.startsWith('DOMAIN')) continue;
    const parts = trimmed.split(/\s+/).map(Number);
    if (parts.length >= 3 && !isNaN(parts[0])) {
      values.push(parts[0], parts[1], parts[2]);
    }
  }
  return { size, data: new Float32Array(values) };
}

function applyLUT(name) {
  currentLUT = name;
  if (!name || !LUT_DATA[name]) {
    removeLUTOverlay();
    return;
  }
  initLUTOverlay(LUT_DATA[name]);
}

function initLUTOverlay(lutInfo) {
  // Create or reuse a WebGL canvas overlaid on the video
  const container = document.getElementById('player-container');
  if (!lutCanvas) {
    lutCanvas = document.createElement('canvas');
    lutCanvas.id = 'lut-canvas';
    lutCanvas.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:2';
    container.appendChild(lutCanvas);
  }
  lutCanvas.style.display = 'block';

  const gl = lutCanvas.getContext('webgl', { preserveDrawingBuffer: true });
  if (!gl) { console.warn('WebGL not available'); return; }
  glContext = gl;

  const size = lutInfo.size;
  const data = lutInfo.data;

  // Create 3D texture
  gl.deleteTexture(lutTexture);
  lutTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_3D, lutTexture);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
  gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGB, size, size, size, 0, gl.RGB, gl.FLOAT, data);

  // Compile shaders
  if (!gl._lutProgram) {
    gl._lutProgram = createLUTShader(gl);
  }

  startLUTRenderLoop();
}

function createLUTShader(gl) {
  const vs = gl.createShader(gl.VERTEX_SHADER);
  gl.shaderSource(vs, 'attribute vec2 a_pos; varying vec2 v_uv; void main() { v_uv = a_pos * 0.5 + 0.5; gl_Position = vec4(a_pos, 0, 1); }');
  gl.compileShader(vs);

  const fs = gl.createShader(gl.FRAGMENT_SHADER);
  gl.shaderSource(fs, `
    precision mediump float;
    varying vec2 v_uv;
    uniform sampler2D u_video;
    uniform sampler3D u_lut;
    uniform float u_lutSize;
    void main() {
      vec4 color = texture2D(u_video, v_uv);
      // Map from [0,1] to LUT coordinate space
      float s = (u_lutSize - 1.0) / u_lutSize;
      float o = 0.5 / u_lutSize;
      vec3 lutCoord = color.rgb * s + o;
      vec3 graded = texture3D(u_lut, lutCoord).rgb;
      gl_FragColor = vec4(graded, color.a);
    }
  `);
  gl.compileShader(fs);

  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  return prog;
}

function startLUTRenderLoop() {
  const render = () => {
    if (!currentLUT || !glContext || !lutCanvas) return;
    const gl = glContext;
    const activeVid = VIDEOS[activeSource];
    if (!activeVid || activeVid.readyState < 2) { requestAnimationFrame(render); return; }

    // Match canvas size to video
    const vw = activeVid.videoWidth || 1920;
    const vh = activeVid.videoHeight || 1080;
    lutCanvas.width = vw;
    lutCanvas.height = vh;

    gl.viewport(0, 0, vw, vh);

    // Create video texture
    const vidTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, vidTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, activeVid);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);

    // Draw fullscreen quad
    const prog = gl._lutProgram;
    gl.useProgram(prog);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, vidTex);
    gl.uniform1i(gl.getUniformLocation(prog, 'u_video'), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_3D, lutTexture);
    gl.uniform1i(gl.getUniformLocation(prog, 'u_lut'), 1);
    gl.uniform1f(gl.getUniformLocation(prog, 'u_lutSize'), LUT_DATA[currentLUT]?.size || 32);

    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
    const apos = gl.getAttribLocation(prog, 'a_pos');
    gl.enableVertexAttribArray(apos);
    gl.vertexAttribPointer(apos, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    gl.deleteTexture(vidTex);
    gl.deleteBuffer(buf);

    if (currentLUT) requestAnimationFrame(render);
  };
  requestAnimationFrame(render);
}

function removeLUTOverlay() {
  if (lutCanvas) lutCanvas.style.display = 'none';
  currentLUT = '';
}

async function uploadLUT() {
  if (!PROJECT_ID) return;
  const file = document.getElementById('lut-file-input').files[0];
  if (!file) return;

  const form = new FormData();
  form.append('name', file.name);
  form.append('file', file);

  try {
    await api(`/api/projects/${PROJECT_ID}/lut`, { method: 'POST', body: form });
    document.getElementById('lut-file-input').value = '';
    loadLUTs();
  } catch(e) { alert('Upload failed: ' + e.message); }
}

async function renderLUTList(luts, defaultName) {
  const div = document.getElementById('lut-list');
  if (!luts.length) { div.innerHTML = 'No LUTs uploaded'; return; }
  div.innerHTML = luts.map(l => `
    <div style="display:flex;align-items:center;justify-content:space-between;padding:4px 0;border-bottom:1px solid #2a2a2f">
      <span>${l.title} <span style="color:#555">(${l.lut_size}³)</span>
        ${l.name === defaultName ? '<span class="badge badge-green" style="margin-left:4px">default</span>' : ''}
      </span>
      <span>
        ${l.name !== defaultName ? `<button class="btn btn-secondary btn-sm" onclick="setDefaultLUT('${l.name}')">Set default</button>` : ''}
        <button class="btn btn-danger btn-sm" onclick="deleteLUT('${l.name}')">×</button>
      </span>
    </div>
  `).join('');
}

async function setDefaultLUT(name) {
  const form = new FormData(); form.append('name', name);
  await api(`/api/projects/${PROJECT_ID}/lut/default`, { method: 'POST', body: form });
  loadLUTs();
}

async function deleteLUT(name) {
  await api(`/api/projects/${PROJECT_ID}/lut/${name}`, { method: 'DELETE' });
  if (currentLUT === name) applyLUT('');
  loadLUTs();
}

// ── Comments (Frame.io-style review) ───────────────────────────────────────

let COMMENTS = [];

async function loadComments() {
  if (!PROJECT_ID) return;
  try {
    const data = await api(`/api/projects/${PROJECT_ID}/comments`);
    COMMENTS = data.comments || [];
    renderComments();
    drawTimeline();
  } catch(e) { COMMENTS = []; }
}

function renderComments() {
  const div = document.getElementById('comments-list');
  const count = document.getElementById('comment-count');
  const unresolved = COMMENTS.filter(c => !c.resolved).length;
  count.textContent = `(${COMMENTS.length} total, ${unresolved} open)`;

  if (!COMMENTS.length) {
    div.innerHTML = '<p style="color:#666;font-size:12px">No comments yet. Click 💬 or use the "Add" button.</p>';
    return;
  }

  div.innerHTML = COMMENTS.map(c => `
    <div style="border-left:3px solid ${c.color};padding:6px 10px;margin:6px 0;background:#1a1a1f;border-radius:0 4px 4px 0;
      ${c.resolved ? 'opacity:0.5' : ''}">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <span style="font-size:12px;font-weight:500">${c.author}</span>
        <span style="font-size:11px;color:#888">
          <a href="#" onclick="seekTo(${c.timestamp} / (CUTLIST?.duration || 1));return false" style="color:#7c5cfc">${fmtTime(c.timestamp)}</a>
        </span>
      </div>
      <p style="margin:4px 0;font-size:13px">${c.text}</p>
      ${(c.replies || []).map(r => `
        <div style="margin-left:12px;padding:4px 8px;border-left:2px solid #3a3a3f;margin-top:4px;font-size:12px">
          <strong>${r.author}:</strong> ${r.text}
        </div>
      `).join('')}
      <div style="display:flex;gap:8px;margin-top:4px">
        <button class="btn btn-secondary btn-sm" onclick="replyToComment('${c.id}')">Reply</button>
        <button class="btn btn-secondary btn-sm" onclick="resolveComment('${c.id}')">${c.resolved ? '↩ Reopen' : '✓ Resolve'}</button>
        <button class="btn btn-danger btn-sm" onclick="deleteComment('${c.id}')">×</button>
      </div>
      <div id="reply-form-${c.id}" style="display:none;margin-top:4px">
        <input type="text" id="reply-author-${c.id}" placeholder="Your name" value="Peter" style="margin-bottom:2px;font-size:12px" />
        <textarea id="reply-text-${c.id}" placeholder="Reply..." rows="1" style="margin-bottom:2px;font-size:12px"></textarea>
        <button class="btn btn-primary btn-sm" onclick="submitReply('${c.id}')">Reply</button>
        <button class="btn btn-secondary btn-sm" onclick="document.getElementById('reply-form-${c.id}').style.display='none'">Cancel</button>
      </div>
    </div>
  `).join('');
}

function addCommentAtCurrentTime() {
  const bar = document.getElementById('seek-bar');
  const t = (parseFloat(bar.value) / 100) * (CUTLIST?.duration || 0);
  document.getElementById('comment-timestamp').value = t;
  document.getElementById('comment-form').style.display = 'block';
  document.getElementById('comment-text').focus();
  document.getElementById('comment-text').placeholder = `Note at ${fmtTime(t)}...`;
}

async function submitComment() {
  const timestamp = parseFloat(document.getElementById('comment-timestamp').value);
  const text = document.getElementById('comment-text').value.trim();
  const author = document.getElementById('comment-author').value.trim() || 'Anonymous';
  if (!text) return;

  try {
    await api(`/api/projects/${PROJECT_ID}/comments`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ timestamp, text, author })
    });
    document.getElementById('comment-form').style.display = 'none';
    document.getElementById('comment-text').value = '';
    loadComments();
  } catch(e) { alert('Failed: ' + e.message); }
}

function cancelComment() {
  document.getElementById('comment-form').style.display = 'none';
  document.getElementById('comment-text').value = '';
}

function replyToComment(id) {
  document.getElementById('reply-form-' + id).style.display = 'block';
  document.getElementById('reply-text-' + id).focus();
}

async function submitReply(id) {
  const text = document.getElementById('reply-text-' + id).value.trim();
  const author = document.getElementById('reply-author-' + id).value.trim() || 'Anonymous';
  if (!text) return;
  await api(`/api/projects/${PROJECT_ID}/comments/${id}/reply`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, author })
  });
  loadComments();
}

async function resolveComment(id) {
  await api(`/api/projects/${PROJECT_ID}/comments/${id}/resolve`, { method: 'POST' });
  loadComments();
}

async function deleteComment(id) {
  await api(`/api/projects/${PROJECT_ID}/comments/${id}`, { method: 'DELETE' });
  loadComments();
}

// ── Update initPlayer to load LUTs and comments ─────────────────────────────

const _origInitPlayer = initPlayer;
initPlayer = async function() {
  await _origInitPlayer();
  loadLUTs();
  loadComments();
};

// ── Update drawTimeline to show comment markers ─────────────────────────────

const _origDrawTimeline = drawTimeline;
drawTimeline = function() {
  _origDrawTimeline();
  if (!CUTLIST || !COMMENTS.length) return;

  const canvas = document.getElementById('timeline-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.offsetWidth;
  const h = 40;
  const dur = CUTLIST.duration;
  const scale = w / dur;

  // Comment markers as colored diamonds on the timeline
  for (const c of COMMENTS) {
    const x = c.timestamp * scale;
    ctx.fillStyle = c.resolved ? '#555' : (c.color || '#7c5cfc');
    ctx.beginPath();
    ctx.moveTo(x, h - 2);
    ctx.lineTo(x - 5, h - 10);
    ctx.lineTo(x, h - 16);
    ctx.lineTo(x + 5, h - 10);
    ctx.closePath();
    ctx.fill();
    // Hover tooltip area
    ctx.fillStyle = c.resolved ? '#444' : c.color + '88';
    ctx.beginPath();
    ctx.arc(x, h - 9, 8, 0, Math.PI * 2);
    ctx.fill();
  }
};
</script>
</body>
</html>"""
