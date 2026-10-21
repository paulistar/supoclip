"""Fase 4 — candidate review helpers (ETA, previews, selection)."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

# Rough CPU-only vertical+captions estimate (worker Contabo-class).
RENDER_SECONDS_PER_CLIP_BASE = 40
RENDER_SECONDS_PER_SOURCE_SECOND = 1.6


def parse_mmss_seconds(label: str | None) -> float | None:
    if not label or not isinstance(label, str):
        return None
    parts = label.strip().split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    if len(nums) == 2:
        return float(nums[0] * 60 + nums[1])
    if len(nums) == 3:
        return float(nums[0] * 3600 + nums[1] * 60 + nums[2])
    return None


def segment_duration_seconds(segment: Mapping[str, Any]) -> float:
    start = parse_mmss_seconds(str(segment.get("start_time") or ""))
    end = parse_mmss_seconds(str(segment.get("end_time") or ""))
    if start is None or end is None or end <= start:
        return 45.0
    return float(end - start)


def estimate_render_seconds(duration_seconds: float) -> int:
    """Gross wall-clock estimate to render one clip on CPU-only worker."""
    duration = max(1.0, float(duration_seconds or 45.0))
    return max(
        RENDER_SECONDS_PER_CLIP_BASE,
        int(round(duration * RENDER_SECONDS_PER_SOURCE_SECOND)),
    )


def text_preview(text: str | None, *, max_chars: int = 160) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1].rstrip() + "…"


def build_candidate_rows(
    excellent: Sequence[Mapping[str, Any]],
    *,
    rendered_starts: Iterable[str],
    skipped_starts: Iterable[str],
) -> list[dict[str, Any]]:
    rendered = {str(s).strip() for s in rendered_starts if s}
    skipped = {str(s).strip() for s in skipped_starts if s}
    rows: list[dict[str, Any]] = []
    for index, segment in enumerate(excellent):
        start = str(segment.get("start_time") or "").strip()
        end = str(segment.get("end_time") or "").strip()
        duration = segment_duration_seconds(segment)
        eta = estimate_render_seconds(duration)
        rows.append(
            {
                "index": index,
                "start_time": start,
                "end_time": end,
                "duration_seconds": round(duration, 1),
                "hook_title": segment.get("hook_title"),
                "text_preview": text_preview(segment.get("text")),
                "virality_score": int(segment.get("virality_score") or 0),
                "hook_score": int(segment.get("hook_score") or 0),
                "rendered": start in rendered,
                "skipped": start in skipped,
                "eta_seconds": eta,
            }
        )
    return rows


def select_candidates_for_render(
    excellent: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    start_times: Sequence[str] | None = None,
    count: int | None = None,
    rendered_starts: Iterable[str] = (),
    skipped_starts: Iterable[str] = (),
    ceiling: int | None = None,
) -> list[Mapping[str, Any]]:
    """Pick segments to render: selected | top | all (respecting skip/already rendered)."""
    rendered = {str(s).strip() for s in rendered_starts if s}
    skipped = {str(s).strip() for s in skipped_starts if s}
    available = [
        seg
        for seg in excellent
        if str(seg.get("start_time") or "").strip() not in rendered
        and str(seg.get("start_time") or "").strip() not in skipped
    ]
    mode_norm = (mode or "selected").strip().lower()
    if mode_norm == "selected":
        wanted = {str(s).strip() for s in (start_times or []) if s}
        picked = [
            seg
            for seg in available
            if str(seg.get("start_time") or "").strip() in wanted
        ]
    elif mode_norm == "all":
        picked = list(available)
    else:  # top
        limit = max(1, int(count or 15))
        picked = list(available[:limit])

    if ceiling is not None:
        picked = picked[: max(1, int(ceiling))]
    return picked


def append_evitar_phrases(brief: str, phrases: Sequence[str]) -> str:
    """Append permanent skips into an EVITAR list in the user brief."""
    cleaned = [ " ".join(str(p).split()).strip() for p in phrases if p ]
    cleaned = [p for p in cleaned if p]
    if not cleaned:
        return brief or ""
    base = (brief or "").rstrip()
    block = "; ".join(cleaned)
    if "EVITAR" in base.upper() or "EVITAR" in base:
        if not base.endswith((";", ",", ".")):
            base = f"{base};"
        return f"{base} {block}".strip()
    if base:
        return f"{base}\nEVITAR: {block}"
    return f"EVITAR: {block}"
