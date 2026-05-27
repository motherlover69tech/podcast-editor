"""
podcast-edit CLI — AI-powered podcast/video editor.

Workflow:
  1. init     — scan video folder, create project config
  2. run      — full pipeline: sync → analyze → decide → FCPXML
  3. review   — apply user review actions, regenerate FCPXML
  4. status   — show current project state
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="podcast-edit",
    help="AI-powered podcast/video editor — exports DaVinci Resolve timelines",
    no_args_is_help=True,
)

console = Console()


@app.command()
def init(
    input_dir: str = typer.Argument(..., help="Folder containing video files"),
    name: str = typer.Option("", "--name", "-n", help="Project name (default: folder name)"),
    wide: str = typer.Option("", "--wide", "-w", help="Name of wide-angle camera (auto-detected if omitted)"),
    speakers: Optional[str] = typer.Option(None, "--speakers", "-s",
        help="Speaker mapping as JSON: '{\"SPEAKER_00\": \"Alice\", \"SPEAKER_01\": \"Bob\"}'"),
    output_dir: str = typer.Option("", "--output", "-o", help="Output directory (default: <input>/.podcast-editor/)"),
):
    """
    Initialize a new project by scanning a folder of video files.

    Detects video sources, attempts to identify camera roles (wide, primary, angle),
    and creates a project configuration.
    """
    from .pipeline import create_project

    input_path = Path(input_dir)
    if not name:
        name = input_path.name

    speaker_dict = {}
    if speakers:
        import json
        speaker_dict = json.loads(speakers)

    config = create_project(
        name=name,
        input_dir=input_dir,
        output_dir=output_dir,
        wide_camera=wide,
        speakers=speaker_dict,
    )

    console.print(f"\n[bold green]✓[/bold green] Project '{name}' created")
    console.print(f"  Sources: {len(config.sources)} cameras")

    table = Table(title="Camera Sources")
    table.add_column("Name", style="cyan")
    table.add_column("Role", style="green")
    table.add_column("File", style="dim")

    for src in config.sources:
        table.add_row(src.name, src.role, Path(src.file_path).name)

    console.print(table)

    console.print(f"\n[bold]Next:[/bold] podcast-edit run '{input_dir}'")
    if config.wide_camera:
        console.print(f"  Wide camera: [cyan]{config.wide_camera}[/cyan] (auto-detected)")
    else:
        console.print("  [yellow]⚠ No wide camera detected. Add --wide <name> for variety breaks.[/yellow]")


@app.command()
def run(
    project_dir: str = typer.Argument(..., help="Project directory (with .podcast-editor/ subfolder)"),
    whisper_method: str = typer.Option("faster-whisper", "--whisper", help="faster-whisper (CPU, default) or whisperx (NVIDIA GPU)"),
    whisper_model: str = typer.Option("large-v3", "--model", "-m", help="Whisper model size"),
    device: str = typer.Option("cpu", "--device", "-d", help="Device: cpu or cuda"),
    hf_token: str = typer.Option("", "--hf-token", help="HuggingFace token for diarization (pyannote)"),
    skip_ingest: bool = typer.Option(False, "--skip-ingest", help="Skip audio extraction/sync"),
    skip_analyze: bool = typer.Option(False, "--skip-analyze", help="Use existing analysis.json"),
):
    """
    Run the full editing pipeline: ingest → sync → analyze → edit → FCPXML.

    This will:
    1. Extract and sync audio from all cameras
    2. Transcribe and diarize speakers
    3. Generate camera-switching edit decisions
    4. Score and suggest social-media clips
    5. Flag repeated content and low-density sections
    6. Export an FCPXML file for DaVinci Resolve
    """
    from .pipeline import load_project, run_pipeline

    config, pd = load_project(project_dir)

    console.print(f"\n[bold]Running pipeline for '{config.name}'...[/bold]")
    console.print(f"  Sources: {len(config.sources)} cameras")
    console.print(f"  Transcription: {whisper_method}/{whisper_model} on {device}")

    try:
        analysis, edit_result, fcpxml_path = run_pipeline(
            config=config,
            project_dir=pd,
            whisper_method=whisper_method,
            whisper_model=whisper_model,
            device=device,
            hf_token=hf_token,
            skip_ingest=skip_ingest,
            skip_analyze=skip_analyze,
        )
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("[yellow]Make sure ffmpeg is installed and video files exist.[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    # Summary
    total_cuts = len([d for d in edit_result.decisions if d.source])
    console.print(f"\n[bold green]✓ Pipeline complete[/bold green]")
    console.print(f"  Duration: {analysis.duration / 60:.1f} minutes")
    console.print(f"  Transcription: {len(analysis.words)} words, {len(analysis.segments)} topic segments")
    console.print(f"  Edit decisions: {total_cuts} camera cuts")
    console.print(f"  Flagged segments: {len(edit_result.flags)}")
    console.print(f"  Social clips: {len(edit_result.clips)}")

    console.print(f"\n[bold]Outputs:[/bold]")
    console.print(f"  FCPXML: [cyan]{fcpxml_path}[/cyan]")
    console.print(f"  Analysis: [dim]{pd / 'analysis.json'}[/dim]")
    console.print(f"  Review manifest: [dim]{pd / 'review_manifest.md'}[/dim]")

    # File sizes
    for path in [fcpxml_path, pd / "analysis.json", pd / "review_manifest.md"]:
        if path.exists():
            size_kb = path.stat().st_size / 1024
            console.print(f"    ({size_kb:.1f} KB)")

    # Flags summary
    if edit_result.flags:
        console.print(f"\n[yellow]⚠ {len(edit_result.flags)} segments flagged for review:[/yellow]")
        for seg in edit_result.flags[:5]:
            flag_str = ", ".join(f.value for f in seg.flags)
            start_m = seg.start / 60
            console.print(f"  [{flag_str}] at {start_m:.1f}min: \"{seg.transcript[:80]}...\"")

    # Clips
    if edit_result.clips:
        console.print(f"\n[bold cyan]🎬 Suggested social clips:[/bold cyan]")
        for clip in edit_result.clips:
            console.print(f"  [{clip.score:.2f}] {clip.title} ({clip.end - clip.start:.0f}s)")

    console.print(f"\n[bold]Next:[/bold]")
    console.print(f"  1. Open {fcpxml_path.name} in DaVinci Resolve (File → Import → Timeline → FCPXML)")
    console.print(f"  2. Review the manifest: [cyan]cat {pd / 'review_manifest.md'}[/cyan]")
    console.print(f"  3. Submit changes: podcast-edit review '{project_dir}' --actions review_actions.json")


@app.command()
def review(
    project_dir: str = typer.Argument(..., help="Project directory"),
    actions_file: str = typer.Option("review_actions.json", "--actions", "-a",
        help="Path to JSON file with review actions"),
):
    """
    Apply review actions and regenerate the FCPXML.

    Create a review_actions.json file with your changes, then run this command.
    The updated timeline is saved as <project>_auto_edit_v2.fcpxml.
    """
    from .pipeline import load_project, apply_review_actions

    config, pd = load_project(project_dir)

    actions_path = Path(actions_file)
    if not actions_path.exists():
        # Check inside project dir
        actions_path = pd / actions_file
    if not actions_path.exists():
        console.print(f"[red]Actions file not found:[/red] {actions_file}")
        console.print("Create a JSON file with review actions, e.g.:")
        console.print("""[
  {"action": "cut_section", "target_start": 120.0, "target_end": 135.0},
  {"action": "change_camera", "target_start": 300.0, "target_end": 320.0, "params": {"camera": "Wide"}}
]""")
        raise typer.Exit(1)

    edit_result = apply_review_actions(str(actions_path), config, pd)

    console.print(f"[bold green]✓[/bold green] Review applied — {len(edit_result.decisions)} decisions")
    fcpxml_path = pd / f"{config.name}_auto_edit_v2.fcpxml"
    console.print(f"  Updated FCPXML: [cyan]{fcpxml_path}[/cyan]")


@app.command()
def status(
    project_dir: str = typer.Argument(..., help="Project directory"),
):
    """Show current project state and analysis summary."""
    from .pipeline import load_project

    try:
        config, pd = load_project(project_dir)
    except FileNotFoundError:
        console.print("[red]No project found.[/red] Run 'podcast-edit init' first.")
        raise typer.Exit(1)

    console.print(f"\n[bold]{config.name}[/bold]")

    # Sources
    table = Table(title="Camera Sources")
    table.add_column("Name", style="cyan")
    table.add_column("Role")
    table.add_column("Offset", style="dim")
    table.add_column("File", style="dim")
    for src in config.sources:
        table.add_row(
            src.name, src.role, f"{src.offset_seconds:.3f}s",
            Path(src.file_path).name
        )
    console.print(table)

    # Analysis
    analysis_path = pd / "analysis.json"
    if analysis_path.exists():
        from .models import AnalysisResult
        analysis = AnalysisResult.model_validate_json(analysis_path.read_text())
        console.print(f"\nDuration: {analysis.duration / 60:.1f} min")
        console.print(f"Words: {len(analysis.words)}")
        console.print(f"Topic segments: {len(analysis.segments)}")
        speakers = set(w.speaker for w in analysis.words if w.speaker)
        if speakers:
            console.print(f"Speakers: {', '.join(sorted(speakers))}")

        # Top clips
        top_clips = sorted(analysis.segments, key=lambda s: s.clip_score, reverse=True)[:3]
        if top_clips:
            console.print(f"\nTop clip candidates:")
            for seg in top_clips:
                console.print(f"  [{seg.clip_score:.2f}] {seg.transcript[:80]}...")
    else:
        console.print("\n[yellow]No analysis yet. Run 'podcast-edit run'.[/yellow]")

    # Outputs
    fcpxml = pd / f"{config.name}_auto_edit.fcpxml"
    if fcpxml.exists():
        console.print(f"\nFCPXML: [cyan]{fcpxml}[/cyan] ({fcpxml.stat().st_size / 1024:.1f} KB)")

    review = pd / "review_manifest.md"
    if review.exists():
        console.print(f"Review manifest: [dim]{review}[/dim]")


@app.command()
def suggest(
    project_dir: str = typer.Argument(..., help="Project directory"),
    query: str = typer.Argument(..., help="What to search for, e.g. 'discussion about pricing'"),
    top_k: int = typer.Option(5, "--top", "-k", help="Number of results"),
):
    """
    Search the transcript for specific topics or moments.

    Uses semantic search over the transcribed segments.
    """
    from .pipeline import load_project

    config, pd = load_project(project_dir)
    analysis_path = pd / "analysis.json"

    if not analysis_path.exists():
        console.print("[red]No analysis found. Run 'podcast-edit run' first.[/red]")
        raise typer.Exit(1)

    from .models import AnalysisResult
    analysis = AnalysisResult.model_validate_json(analysis_path.read_text())

    # Semantic search
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        model = SentenceTransformer("all-MiniLM-L6-v2")
        segment_texts = [s.transcript for s in analysis.segments]
        embeddings = model.encode(segment_texts)
        query_emb = model.encode([query])

        similarities = cosine_similarity(query_emb, embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        console.print(f"\n[bold]Results for:[/bold] \"{query}\"")
        for rank, idx in enumerate(top_indices):
            seg = analysis.segments[idx]
            sim = similarities[idx]
            color = "green" if sim > 0.5 else "yellow" if sim > 0.3 else "dim"
            console.print(f"\n[{color}]#{rank+1}  [{_fmt_time(seg.start)}–{_fmt_time(seg.end)}]  sim={sim:.3f}[/{color}]")
            console.print(f"  {seg.transcript[:200]}...")
    except ImportError:
        console.print("[red]sentence-transformers not installed. Run: pip install sentence-transformers[/red]")
        raise typer.Exit(1)


def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


if __name__ == "__main__":
    app()
