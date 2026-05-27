"""
FCPXML generator — produces DaVinci Resolve-compatible timeline files.

Resolve imports FCPXML via: File → Import → Timeline → FCPXML

The generated FCPXML includes:
- Multicam clips with sync offsets
- Cut decisions as a main storyline with camera-angle references
- Markers for flags, clip suggestions, and review notes
- Optional: separate timelines for each social-media clip
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET
from xml.dom import minidom

from .models import (
    CameraSource, ClipExtract, EditDecision, EditResult,
    ProjectConfig, Segment, Transition,
)


FCPXML_NS = "http://www.apple.com/finalcutpro/fcpxml/1.9"


def generate_fcpxml(
    edit_result: EditResult,
    config: ProjectConfig,
    output_path: str,
    embed_media: bool = False,
) -> str:
    """
    Generate an FCPXML file that Resolve can import.

    Structure:
      resources/ — all source media referenced by ID
      library/event/project/sequence/spine/ — the edited timeline
    """
    fcpxml = ET.Element("fcpxml", {"version": "1.9"})

    # ── Resources ─────────────────────────────────────────────────────────
    resources = ET.SubElement(fcpxml, "resources")
    _build_resources(resources, config)

    # ── Library → Event → Project → Sequence ──────────────────────────────
    library = ET.SubElement(fcpxml, "library")
    event = ET.SubElement(library, "event", {"name": config.name})
    project = ET.SubElement(event, "project", {"name": f"{config.name} — Auto-Edit"})

    # ── Main timeline: multi-cam edit ─────────────────────────────────────
    sequence_main = _build_sequence(project, edit_result, config, is_multicam=len(config.sources) > 1)
    spine_main = ET.SubElement(sequence_main, "spine")

    _build_edit_timeline(spine_main, edit_result.decisions, config)

    # ── Social clip timelines ─────────────────────────────────────────────
    if edit_result.clips:
        for clip in edit_result.clips:
            clip_project = ET.SubElement(event, "project", {"name": f"Clip: {clip.title[:60]}"})
            clip_seq = _build_sequence(clip_project, edit_result, config, is_multicam=False)
            clip_spine = ET.SubElement(clip_seq, "spine")
            _build_clip_range(clip_spine, clip, config)

    # ── Flags as a marker-only timeline ───────────────────────────────────
    if edit_result.flags:
        flag_project = ET.SubElement(event, "project", {"name": f"{config.name} — Flags & Notes"})
        flag_seq = _build_sequence(flag_project, edit_result, config, is_multicam=False)
        flag_spine = ET.SubElement(flag_seq, "spine")
        _build_flags_timeline(flag_spine, edit_result.flags, config)

    # ── Serialize ─────────────────────────────────────────────────────────
    rough = ET.tostring(fcpxml, encoding="unicode")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ")

    # Strip the XML declaration if present (minidom adds it), then add our own
    lines = pretty.splitlines()
    if lines and lines[0].startswith("<?xml"):
        lines = lines[1:]
    output = "\n".join(lines)

    output_path = Path(output_path)
    output_path.write_text(output)

    return str(output_path)


def _build_resources(resources: ET.Element, config: ProjectConfig) -> None:
    """Add <asset> entries for each source video."""
    for src in config.sources:
        asset_id = _safe_id(src.name)
        # Resolve needs the format to be specified
        format_id = f"r1_{asset_id}"

        # Determine format from video
        format_el = ET.SubElement(resources, "format", {
            "id": format_id,
            "name": f"FFVideoFormat{src.name}",
            "frameDuration": "1001/30000s",  # 29.97 fps default
            "width": "1920",
            "height": "1080",
        })

        asset = ET.SubElement(resources, "asset", {
            "id": asset_id,
            "name": src.name,
            "src": f"file://{Path(src.file_path).absolute()}",
            "duration": f"{_media_duration(src.file_path)}/1s",
            "format": format_id,
        })


def _build_sequence(
    project: ET.Element,
    edit_result: EditResult,
    config: ProjectConfig,
    is_multicam: bool = False,
) -> ET.Element:
    """Build a <sequence> element."""
    # Get total duration from decisions
    total_dur = 0
    if edit_result.decisions:
        total_dur = edit_result.decisions[-1].time + 5

    attrs = {
        "format": f"r1_{_safe_id(config.sources[0].name)}",
        "duration": f"{int(total_dur * 24000 / 1001)}/24000s",  # 24fps timebase
    }
    return ET.SubElement(project, "sequence", attrs)


def _build_edit_timeline(
    spine: ET.Element,
    decisions: list[EditDecision],
    config: ProjectConfig,
) -> None:
    """
    Build the main edit timeline from cut decisions.
    Each decision = a clip from a specific camera angle.
    """
    # Filter to camera-switch decisions only (skip marker-only decisions)
    camera_cuts = [d for d in decisions if d.source and d.source != ""]

    if not camera_cuts:
        return

    # First decision sets the starting camera
    prev = camera_cuts[0]

    for i, curr in enumerate(camera_cuts):
        start_time = prev.time if i > 0 else 0.0
        end_time = curr.time

        # Only emit if there's meaningful duration
        if end_time - start_time < 0.01:
            prev = curr
            continue

        duration_sec = end_time - start_time

        # Convert to 24fps ticks (FCPXML uses fraction strings)
        start_ticks = int(start_time * 24000 / 1001)
        dur_ticks = int(duration_sec * 24000 / 1001)

        # Source offset accounts for sync offset
        src = _get_source(prev.source, config)
        offset_samples = int(
            (max(0, start_time) + src.offset_seconds) * 24000 / 1001
        )

        asset_id = _safe_id(prev.source)

        clip_attrs = {
            "ref": asset_id,
            "offset": f"{offset_samples}/24000s",
            "duration": f"{dur_ticks}/24000s",
            "name": f"{prev.source} — {prev.reason.value}",
        }

        if i == 0:
            clip_attrs["start"] = f"0/24000s"

        asset_clip = ET.SubElement(spine, "asset-clip", clip_attrs)

        # Add transition if applicable
        if prev.transition == Transition.DISSOLVE and i > 0:
            # Cross dissolve at the edit point
            trans_dur = int(prev.transition_duration * 24000 / 1001)
            if trans_dur > 0:
                ET.SubElement(spine, "transition", {
                    "name": "Cross Dissolve",
                    "duration": f"{trans_dur}/24000s",
                })

        # Add markers for decisions with notes
        if prev.note:
            note_time = int((start_time - prev.time) * 24000 / 1001)
            if note_time >= 0:
                ET.SubElement(asset_clip, "marker", {
                    "start": f"{note_time}/24000s",
                    "duration": "1/24000s",
                    "value": prev.note,
                })

        prev = curr

    # Final segment: from last cut to end
    if camera_cuts:
        last = camera_cuts[-1]
        total_dur_sec = decisions[-1].time + 5  # pad end
        final_start = last.time
        final_dur = max(0, total_dur_sec - final_start)

        if final_dur > 0.1:
            dur_ticks = int(final_dur * 24000 / 1001)
            offset_samples = int(
                (final_start + _get_source(last.source, config).offset_seconds) * 24000 / 1001
            )

            ET.SubElement(spine, "asset-clip", {
                "ref": _safe_id(last.source),
                "offset": f"{offset_samples}/24000s",
                "duration": f"{dur_ticks}/24000s",
                "name": f"{last.source} — end",
            })


def _build_clip_range(
    spine: ET.Element,
    clip: ClipExtract,
    config: ProjectConfig,
) -> None:
    """Build a single clip's timeline — the entire segment from one source."""
    if not config.sources:
        return

    src = config.sources[0]  # use primary source for clips
    dur_sec = clip.end - clip.start
    dur_ticks = int(dur_sec * 24000 / 1001)

    asset_clip = ET.SubElement(spine, "asset-clip", {
        "ref": _safe_id(src.name),
        "offset": f"{int(clip.start * 24000 / 1001)}/24000s",
        "duration": f"{dur_ticks}/24000s",
        "name": clip.title[:80],
    })

    if clip.description:
        ET.SubElement(asset_clip, "marker", {
            "start": "0/24000s",
            "duration": "1/24000s",
            "value": clip.description,
        })


def _build_flags_timeline(
    spine: ET.Element,
    flags: list[Segment],
    config: ProjectConfig,
) -> None:
    """Build a marker-only timeline showing flagged segments."""
    if not config.sources or not flags:
        return

    src = config.sources[0]
    total_dur = flags[-1].end
    dur_ticks = int(total_dur * 24000 / 1001)

    # Full asset clip spanning the entire recording
    asset_clip = ET.SubElement(spine, "asset-clip", {
        "ref": _safe_id(src.name),
        "offset": "0/24000s",
        "duration": f"{dur_ticks}/24000s",
        "name": "Flag Review",
    })

    for seg in flags:
        if not seg.flags:
            continue

        start_ticks = int(seg.start * 24000 / 1001)
        dur = int((seg.end - seg.start) * 24000 / 1001)

        flag_labels = [f.value for f in seg.flags]
        marker_text = f"{', '.join(flag_labels)} | {seg.transcript[:100]}..."
        if seg.repeats_segment:
            marker_text += f"\nRepeats: {seg.repeats_segment}"

        ET.SubElement(asset_clip, "marker", {
            "start": f"{start_ticks}/24000s",
            "duration": f"{dur}/24000s",
            "value": marker_text,
        })

        # Add a keyword marker at the start for quick navigation
        for flag in seg.flags:
            ET.SubElement(asset_clip, "keyword-marker", {
                "start": f"{start_ticks}/24000s",
                "duration": "1/24000s",
                "value": flag.value,
            })


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_id(name: str) -> str:
    """Convert a name to a safe XML ID."""
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_")


def _get_source(name: str, config: ProjectConfig) -> CameraSource:
    for s in config.sources:
        if s.name == name:
            return s
    return config.sources[0] if config.sources else CameraSource(
        name="fallback", file_path="", role="angle"
    )


def _media_duration(file_path: str) -> int:
    """Get media duration in integer seconds via ffprobe."""
    import subprocess
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
             file_path],
            capture_output=True, text=True, timeout=30,
        )
        return int(float(result.stdout.strip()))
    except Exception:
        return 3600  # fallback: assume 1 hour
