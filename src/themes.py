"""
Theme extraction, content analysis, and YouTube trend-based title suggestions.

Uses:
- Sentence embeddings for topic clustering
- LLM (optional) for theme labeling and title generation
- YouTube Data API or web scraping for trend analysis
"""

from __future__ import annotations

import json
import re
from typing import Optional

from .models import Segment


# ── Theme extraction ─────────────────────────────────────────────────────────

def extract_themes(
    segments: list[Segment],
    n_themes: int = 5,
) -> dict[str, list[Segment]]:
    """
    Cluster segments into themes using embedding similarity.
    Returns {theme_label: [segments]}.

    If sentence-transformers is available, uses semantic clustering.
    Otherwise falls back to keyword-based grouping.
    """
    if len(segments) <= 1:
        return {"Full Episode": segments}

    try:
        return _semantic_themes(segments, n_themes)
    except ImportError:
        return _keyword_themes(segments, n_themes)


def _semantic_themes(
    segments: list[Segment],
    n_themes: int,
) -> dict[str, list[Segment]]:
    """Cluster segments by embedding similarity (agglomerative)."""
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity

    texts = [s.transcript for s in segments]
    if len(texts) < 3:
        return {"Full Episode": segments}

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts)

    # Agglomerative clustering
    n_clusters = min(n_themes, len(segments) // 2, len(segments) - 1)
    n_clusters = max(2, n_clusters)

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(embeddings)

    # Group segments by cluster
    clusters: dict[int, list[Segment]] = {}
    for seg, label in zip(segments, labels):
        clusters.setdefault(int(label), []).append(seg)

    # Name each cluster by its most representative segment
    themed: dict[str, list[Segment]] = {}
    for label, cluster_segs in clusters.items():
        # Find segment closest to cluster centroid
        cluster_embs = embeddings[[i for i, s in enumerate(segments) if s in cluster_segs]]
        centroid = np.mean(cluster_embs, axis=0).reshape(1, -1)
        sims = cosine_similarity(centroid, cluster_embs)[0]
        best_idx = int(np.argmax(sims))
        representative = cluster_segs[best_idx].transcript[:80]

        # Generate a short label from the representative text
        theme_name = _summarize_to_label(representative, max_words=5)

        # Assign theme to segments
        for seg in cluster_segs:
            seg.topic = theme_name
        themed[theme_name] = cluster_segs

    # Assign themes back to segments
    for theme_name, cluster_segs in themed.items():
        for seg in cluster_segs:
            seg.topic = theme_name

    return themed


def _keyword_themes(
    segments: list[Segment],
    n_themes: int,
) -> dict[str, list[Segment]]:
    """Fallback: keyword-based theme grouping."""
    from collections import Counter

    # Common stopwords to ignore
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
        "us", "them", "my", "your", "his", "its", "our", "their",
        "this", "that", "these", "those", "and", "or", "but", "if",
        "in", "on", "at", "to", "for", "of", "with", "from", "by",
        "about", "as", "into", "through", "during", "before", "after",
        "so", "just", "then", "now", "also", "very", "really", "like",
        "would", "could", "should", "will", "can", "may", "might",
        "do", "does", "did", "have", "has", "had", "been", "being",
    }

    # Collect significant keywords per segment
    segment_keywords: list[set[str]] = []
    all_keywords: Counter = Counter()

    for seg in segments:
        words = re.findall(r'\b[a-z]{4,}\b', seg.transcript.lower())
        keywords = set(w for w in words if w not in stopwords)
        segment_keywords.append(keywords)
        all_keywords.update(keywords)

    # Top keywords become theme labels
    top_keywords = [kw for kw, _ in all_keywords.most_common(n_themes * 3)]

    themed: dict[str, list[Segment]] = {}

    # Assign each segment to the theme whose keywords it shares most
    for seg, seg_kws in zip(segments, segment_keywords):
        best_theme = "General"
        best_overlap = 0
        for kw in top_keywords:
            if kw in seg_kws:
                count = sum(1 for k in seg_kws if k == kw)
                if count > best_overlap:
                    best_overlap = count
                    best_theme = kw.title()

        themed.setdefault(best_theme, []).append(seg)
        seg.topic = best_theme

    return themed


def _summarize_to_label(text: str, max_words: int = 5) -> str:
    """Extract a short label from text."""
    # Remove filler and take first few meaningful words
    words = text.split()
    label_words = []
    for w in words:
        clean = w.strip(".,!?;:\"'()[]{}").lower()
        if clean and clean not in {"um", "uh", "er", "ah", "like", "you", "know", "so", "yeah", "okay", "just", "really", "actually", "basically"}:
            label_words.append(clean)
            if len(label_words) >= max_words:
                break
    return " ".join(label_words).title() if label_words else "Discussion"


# ── Episode summary ──────────────────────────────────────────────────────────

def generate_summary(
    segments: list[Segment],
    total_duration: float,
) -> dict:
    """Generate an episode overview: topics, pacing, recommendations."""
    themes = extract_themes(segments)

    # Per-theme stats
    theme_stats = {}
    for theme, theme_segs in themes.items():
        dur = sum(s.end - s.start for s in theme_segs)
        avg_score = sum(s.clip_score for s in theme_segs) / max(len(theme_segs), 1)
        theme_stats[theme] = {
            "duration_sec": round(dur, 1),
            "duration_pct": round(dur / total_duration * 100, 1) if total_duration else 0,
            "segment_count": len(theme_segs),
            "avg_clip_score": round(avg_score, 3),
            "first_at": round(theme_segs[0].start, 1),
        }

    # Overall pacing
    speaking_changes = 0
    prev_speaker = ""
    for seg in segments:
        if seg.speaker and seg.speaker != prev_speaker:
            speaking_changes += 1
        prev_speaker = seg.speaker

    flagged_count = sum(1 for s in segments if s.flags)

    return {
        "total_duration_sec": round(total_duration, 1),
        "total_duration_min": round(total_duration / 60, 1),
        "segment_count": len(segments),
        "theme_count": len(themes),
        "themes": theme_stats,
        "speaking_turns": speaking_changes,
        "flagged_segments": flagged_count,
        "top_themes": sorted(theme_stats.keys(),
            key=lambda t: theme_stats[t]["duration_pct"], reverse=True)[:3],
    }


# ── YouTube trend-based title suggestions ────────────────────────────────────

def suggest_titles(
    segments: list[Segment],
    themes: dict[str, list[Segment]],
    style: str = "balanced",  # "clickbait", "educational", "balanced", "interview"
    n: int = 5,
) -> list[dict]:
    """
    Generate episode title suggestions based on content themes
    and YouTube-trending title patterns.

    Uses known effective YouTube title formats for each style.
    Future: can integrate with YouTube Data API for real trend data.
    """
    # Collect key phrases from top themes
    top_themes = sorted(themes.keys(),
        key=lambda t: sum(s.clip_score for s in themes[t]) / max(len(themes[t]), 1),
        reverse=True)

    theme_phrases = []
    for theme in top_themes[:3]:
        segs = themes[theme]
        # Get most impactful sentences
        scored_sentences = []
        for seg in segs:
            for sent in seg.transcript.split("."):
                sent = sent.strip()
                if 5 < len(sent.split()) < 15:
                    scored_sentences.append((sent, seg.clip_score))
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        for phrase, _ in scored_sentences[:2]:
            if phrase not in theme_phrases:
                theme_phrases.append(_clean_phrase(phrase))

    # YouTube-proven title patterns by style
    patterns = {
        "clickbait": [
            "The SHOCKING Truth About {topic}",
            "You Won't Believe What {topic} Revealed",
            "{topic} Changed EVERYTHING — Here's Why",
            "The {topic} SECRET Nobody Talks About",
            "I Can't Believe {topic} Actually Works",
        ],
        "educational": [
            "Why {topic} Matters More Than You Think",
            "The Complete Guide to {topic}",
            "How {topic} Actually Works — Explained",
            "{topic}: What They Don't Tell You",
            "Understanding {topic} in {minutes} Minutes",
        ],
        "interview": [
            "{guest} on {topic} — Full Conversation",
            "{guest} Reveals: {quote}",
            "Inside {topic} with {guest}",
            "{guest}: \"{quote}\" — Exclusive Interview",
            "The {topic} Discussion Everyone's Talking About",
        ],
        "balanced": [
            "What We Got Wrong About {topic}",
            "{topic} — A Deep Dive",
            "The Real Story Behind {topic}",
            "Rethinking {topic}: {quote}",
            "Everything We Know About {topic} (So Far)",
        ],
    }

    chosen = patterns.get(style, patterns["balanced"])
    titles: list[dict] = []

    for i, pattern in enumerate(chosen[:n]):
        # Fill in template with available phrases
        title = pattern
        phrase_idx = i % max(len(theme_phrases), 1)

        if theme_phrases:
            title = title.replace("{topic}", theme_phrases[phrase_idx])
            title = title.replace("{quote}", f"\"{theme_phrases[phrase_idx]}\"")
            title = title.replace("{guest}", "Our Guest")
        else:
            title = title.replace("{topic}", "This Episode")
            title = title.replace("{quote}", "\"Fascinating Discussion\"")
            title = title.replace("{guest}", "Our Guest")

        # Estimate duration
        total_dur = sum(s.end - s.start for s in segments)
        title = title.replace("{minutes}", f"{int(total_dur / 60)}")

        # Score based on pattern match quality
        score = 0.8 if theme_phrases else 0.4

        titles.append({
            "title": title,
            "style": style,
            "score": round(score, 2),
        })

    return titles


def suggest_youtube_titles_from_summary(
    summary: dict,
    phrases: list[str],
    style: str = "balanced",
) -> list[dict]:
    """
    Generate titles from a pre-computed summary + key phrases.
    Used by the web backend to avoid re-loading the transcript.
    """
    # Create lightweight fake segments
    fake_segs = []
    for theme, stats in summary.get("themes", {}).items():
        fake_segs.append(Segment(
            id=f"fake_{theme}",
            start=stats.get("first_at", 0),
            end=stats.get("first_at", 0) + stats.get("duration_sec", 60),
            speaker="",
            transcript=theme,
            topic=theme,
            clip_score=stats.get("avg_clip_score", 0.5),
        ))

    fake_themes = {theme: [s] for theme, s in zip(summary.get("themes", {}).keys(), fake_segs)
                   if theme in summary.get("themes", {})}

    return suggest_titles(fake_segs, fake_themes or {"Full Episode": fake_segs}, style=style)


def _clean_phrase(text: str) -> str:
    """Clean and capitalize a phrase for title use."""
    text = text.strip().rstrip(".,;")
    # Remove common filler prefixes
    text = re.sub(r'^(um |uh |so |yeah |okay |well |i mean |you know |like )+', '', text, flags=re.IGNORECASE)
    if len(text) > 3:
        text = text[0].upper() + text[1:]
    return text or "This Topic"


# ── YouTube trend lookup (stub — ready for real API) ─────────────────────────

def lookup_youtube_trends(
    keywords: list[str],
    api_key: str = "",
    max_results: int = 10,
) -> list[dict]:
    """
    Search YouTube for trending/popular videos matching keywords.
    Requires a YouTube Data API key.

    Without an API key, returns known effective title patterns
    as "trend data" for the given keywords.
    """
    if not api_key:
        return _simulate_trends(keywords)

    # Real YouTube Data API call
    import urllib.request
    import urllib.parse

    results = []
    for keyword in keywords[:3]:
        params = urllib.parse.urlencode({
            "part": "snippet",
            "q": keyword,
            "type": "video",
            "maxResults": min(max_results, 5),
            "order": "viewCount",
            "relevanceLanguage": "en",
            "key": api_key,
        })
        url = f"https://www.googleapis.com/youtube/v3/search?{params}"

        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read())
                for item in data.get("items", []):
                    results.append({
                        "title": item["snippet"]["title"],
                        "channel": item["snippet"]["channelTitle"],
                        "video_id": item["id"].get("videoId", ""),
                    })
        except Exception:
            pass

    return results[:max_results]


def _simulate_trends(keywords: list[str]) -> list[dict]:
    """Return known effective title patterns when API key is unavailable."""
    patterns = []
    for kw in keywords[:3]:
        patterns.extend([
            {"title": f"Why {kw.title()} Is The Future", "channel": "TrendSignal", "video_id": ""},
            {"title": f"I Tested {kw.title()} For 30 Days", "channel": "TrendSignal", "video_id": ""},
            {"title": f"The Truth About {kw.title()}", "channel": "TrendSignal", "video_id": ""},
        ])
    return patterns[:10]
