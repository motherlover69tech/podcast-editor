"""
Time-constrained edit algorithm.

Given a target duration (e.g., 10 minutes), selects the best segments
grouped by theme to produce a cohesive edit that fits within the limit.

Strategy:
1. Group segments by theme
2. Score each theme by average clip_score × duration
3. Fill time budget with highest-scoring segments, respecting theme grouping
4. Drop entire themes if they're too weak to fit
5. Apply smooth transitions between themes (brief gap or transition marker)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .models import CutReason, EditDecision, Segment, Transition


@dataclass
class TimedEdit:
    """Result of a time-constrained edit."""
    selected_segments: list[Segment]
    dropped_segments: list[Segment]
    dropped_themes: list[str]
    total_duration: float
    target_duration: float
    fill_pct: float
    decisions: list[EditDecision]  # for FCPXML integration


def time_constrained_edit(
    segments: list[Segment],
    target_duration: float,
    exclude_themes: Optional[list[str]] = None,
    transition_padding: float = 2.0,  # seconds between theme transitions
) -> TimedEdit:
    """
    Select the best segments to fit within target_duration seconds,
    grouped by theme to avoid topic-hopping.

    Algorithm:
    1. Filter out excluded themes and flagged low-density segments
    2. Group remaining by theme
    3. Per theme: select highest-scoring contiguous run that fits the budget
    4. Allocate budget proportionally to theme quality, then fill gaps

    Returns a TimedEdit with selected/dropped segments and FCPXML decisions.
    """
    exclude = set(exclude_themes or [])

    # 1. Filter — drop excluded themes and low-density segments
    candidates = [
        s for s in segments
        if s.topic not in exclude
        and "low_density" not in [f.value for f in s.flags]
    ]

    if not candidates:
        # Nothing to select — return everything
        return TimedEdit(
            selected_segments=segments,
            dropped_segments=[],
            dropped_themes=[],
            total_duration=sum(s.end - s.start for s in segments),
            target_duration=target_duration,
            fill_pct=100.0,
            decisions=[],
        )

    # 2. Group by theme, preserving temporal order
    themes: dict[str, list[Segment]] = {}
    for seg in candidates:
        theme = seg.topic or "Uncategorized"
        themes.setdefault(theme, []).append(seg)

    # Sort segments within each theme by time
    for theme_segs in themes.values():
        theme_segs.sort(key=lambda s: s.start)

    # 3. Score each theme
    theme_scores: list[tuple[str, float, float]] = []  # (name, score, total_duration)
    for theme, theme_segs in themes.items():
        total_dur = sum(s.end - s.start for s in theme_segs)
        avg_score = sum(s.clip_score for s in theme_segs) / max(len(theme_segs), 1)
        # Score = quality × duration (longer good themes get more budget)
        score = avg_score * total_dur
        theme_scores.append((theme, score, total_dur))

    theme_scores.sort(key=lambda x: x[1], reverse=True)

    # 4. Allocate budget
    # First pass: allocate proportional to score, minimum 10% per theme
    total_score = sum(s for _, s, _ in theme_scores)
    min_theme_budget = target_duration * 0.05  # 5% minimum per theme

    allocations: dict[str, float] = {}
    remaining = target_duration - len(theme_scores) * transition_padding

    for theme, score, total_dur in theme_scores:
        if total_score > 0:
            alloc = (score / total_score) * remaining
        else:
            alloc = remaining / max(len(theme_scores), 1)

        # Floor: give at least min_theme_budget or the theme's actual duration
        alloc = max(min(alloc, total_dur), min(min_theme_budget, total_dur))
        allocations[theme] = alloc

    # Normalize to fit within budget
    total_allocated = sum(allocations.values()) + len(theme_scores) * transition_padding
    if total_allocated > target_duration:
        scale = target_duration / total_allocated
        for theme in allocations:
            allocations[theme] *= scale

    # 5. Select segments per theme
    selected: list[Segment] = []
    dropped: list[Segment] = []
    dropped_themes: list[str] = []

    for theme, _, _ in theme_scores:
        budget = allocations.get(theme, 0)
        theme_segs = themes[theme]

        # If budget is very small, drop the whole theme
        if budget < 10:  # less than 10 seconds = not worth it
            dropped.extend(theme_segs)
            dropped_themes.append(theme)
            continue

        # Select highest-scoring contiguous run within budget
        best_run = _select_best_run(theme_segs, budget)

        if not best_run:
            dropped.extend(theme_segs)
            dropped_themes.append(theme)
        else:
            selected.extend(best_run)
            dropped.extend([s for s in theme_segs if s not in best_run])

    # 6. Sort selected by time for continuity
    selected.sort(key=lambda s: s.start)

    # 7. Check if we're over budget and trim
    total_selected = sum(s.end - s.start for s in selected) + max(0, len(set(s.topic for s in selected)) - 1) * transition_padding
    if total_selected > target_duration * 1.05:
        # Trim lowest-scoring segments until we fit
        selected.sort(key=lambda s: s.clip_score)
        while total_selected > target_duration * 1.05 and selected:
            removed = selected.pop(0)
            dropped.append(removed)
            total_selected = sum(s.end - s.start for s in selected) + max(0, len(set(s.topic for s in selected)) - 1) * transition_padding

        selected.sort(key=lambda s: s.start)  # re-sort by time

    # 8. Generate edit decisions for FCPXML
    decisions = _generate_time_edit_decisions(selected, transition_padding)

    return TimedEdit(
        selected_segments=selected,
        dropped_segments=dropped,
        dropped_themes=dropped_themes,
        total_duration=total_selected,
        target_duration=target_duration,
        fill_pct=round(total_selected / target_duration * 100, 1) if target_duration else 100,
        decisions=decisions,
    )


def _select_best_run(
    segments: list[Segment],
    budget: float,
) -> list[Segment]:
    """
    Select the highest-scoring contiguous sub-sequence that fits within budget.
    Uses sliding window with score maximization.

    If no contiguous run fits, takes the single best segment.
    """
    if not segments:
        return []

    # Find all contiguous runs that fit within budget
    best_run: list[Segment] = []
    best_score = 0.0

    for i in range(len(segments)):
        dur = 0.0
        total_score = 0.0
        run: list[Segment] = []

        for j in range(i, len(segments)):
            seg_dur = segments[j].end - segments[j].start
            if dur + seg_dur > budget:
                break
            run.append(segments[j])
            dur += seg_dur
            total_score += segments[j].clip_score * seg_dur

        avg_score = total_score / max(dur, 0.1)
        if avg_score > best_score and len(run) >= 1:
            best_score = avg_score
            best_run = list(run)

    # If nothing fits, take single best segment
    if not best_run:
        best_seg = max(segments, key=lambda s: s.clip_score)
        seg_dur = best_seg.end - best_seg.start
        if seg_dur <= budget * 1.5:  # allow slight overflow
            best_run = [best_seg]

    return best_run


def _generate_time_edit_decisions(
    selected: list[Segment],
    padding: float,
) -> list[EditDecision]:
    """
    Generate edit decisions for a time-constrained edit.
    Includes theme boundary markers.
    """
    decisions: list[EditDecision] = []
    prev_theme = ""

    for i, seg in enumerate(selected):
        seg_start = seg.start
        seg_end = seg.end

        # Theme boundary
        if seg.topic != prev_theme and prev_theme:
            decisions.append(EditDecision(
                time=seg_start,
                source="",  # marker only
                transition=Transition.DISSOLVE,
                transition_duration=min(padding, 2.0),
                reason=CutReason.TOPIC_BOUNDARY,
                note=f"→ Theme: {seg.topic}",
            ))

        decisions.append(EditDecision(
            time=seg_start,
            source="main",  # placeholder — replaced by actual camera decisions later
            reason=CutReason.TOPIC_BOUNDARY,
            note=f"Include: {seg.topic or 'segment'} [{seg.clip_score:.2f}]",
        ))

        prev_theme = seg.topic or prev_theme

    return decisions


def format_time_edit_summary(edit: TimedEdit) -> str:
    """Human-readable summary of a time-constrained edit."""
    lines = []
    lines.append(f"Target: {edit.target_duration / 60:.1f} min")
    lines.append(f"Actual: {edit.total_duration / 60:.1f} min ({edit.fill_pct:.0f}% fill)")
    lines.append(f"Selected: {len(edit.selected_segments)} segments")
    lines.append(f"Dropped: {len(edit.dropped_segments)} segments")
    if edit.dropped_themes:
        lines.append(f"Dropped themes: {', '.join(edit.dropped_themes)}")

    # Per-theme breakdown
    themes: dict[str, list[Segment]] = {}
    for seg in edit.selected_segments:
        themes.setdefault(seg.topic or "Other", []).append(seg)

    lines.append("\nBy theme:")
    for theme, segs in sorted(themes.items(), key=lambda x: -sum(s.clip_score for s in x[1])):
        dur = sum(s.end - s.start for s in segs)
        avg = sum(s.clip_score for s in segs) / max(len(segs), 1)
        lines.append(f"  {theme}: {dur / 60:.1f}min ({len(segs)} segments, avg score {avg:.2f})")

    return "\n".join(lines)
