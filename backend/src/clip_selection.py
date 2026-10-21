"""Clip duration presets, quantity budget, and user brief helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Mapping

# Soft safety cap only — never reject with 400; truncate instead.
CLIP_BRIEF_MAX_CHARS = 50_000
VALID_DURATION_PRESETS = (
    "auto",
    "short",
    "long",
    "viral",
    "monetization",
    "podcast",
)
VALID_MIN_CLIPS = (5, 10, 15, 20)
# Auto quantity: no fill-to-N. Render at most this many excellent clips.
AUTO_RENDER_DEFAULT = 15
# Server safety cap for discovery (not a product target). Env MAX_CLIPS may be lower.
DEFAULT_DISCOVERY_SAFETY_CAP = 60

# Spacing multiplies ideal midpoint for legacy density estimates / UI hints.
# Longer formats leave more room between clips.
DURATION_PRESETS: dict[str, dict[str, Any]] = {
    "auto": {
        "label": "Auto (30s–2min)",
        "min_seconds": 30,
        "max_seconds": 120,
        "ideal_min": 30,
        "ideal_max": 90,
        "spacing": 1.2,
        "suggest_min_clips": "auto",
    },
    "short": {
        "label": "Curtos (15-30s)",
        "min_seconds": 15,
        "max_seconds": 30,
        "ideal_min": 20,
        "ideal_max": 28,
        "spacing": 1.0,
        "suggest_min_clips": 10,
    },
    "long": {
        "label": "Longos (45-90s)",
        "min_seconds": 45,
        "max_seconds": 90,
        "ideal_min": 50,
        "ideal_max": 75,
        "spacing": 1.3,
        "suggest_min_clips": 10,
    },
    "viral": {
        "label": "Virais (2-3 min)",
        "min_seconds": 120,
        "max_seconds": 180,
        "ideal_min": 135,
        "ideal_max": 165,
        "spacing": 1.6,
        "suggest_min_clips": 5,
    },
    "monetization": {
        "label": "Monetização (3-5 min)",
        "min_seconds": 180,
        "max_seconds": 300,
        "ideal_min": 210,
        "ideal_max": 270,
        "spacing": 1.8,
        "suggest_min_clips": 5,
    },
    "podcast": {
        "label": "Podcast (9-16 min)",
        "min_seconds": 540,
        "max_seconds": 960,
        "ideal_min": 600,
        "ideal_max": 840,
        "spacing": 2.0,
        "suggest_min_clips": 5,
    },
}


@dataclass(frozen=True)
class ClipSelectionPrefs:
    duration_preset: str = "auto"
    min_clips: int | None = None  # None = Auto quantity
    brief: str = ""

    @property
    def preset(self) -> dict[str, Any]:
        return DURATION_PRESETS.get(self.duration_preset, DURATION_PRESETS["auto"])

    @property
    def min_seconds(self) -> int:
        return int(self.preset["min_seconds"])

    @property
    def max_seconds(self) -> int:
        return int(self.preset["max_seconds"])

    @property
    def ideal_min(self) -> int:
        return int(self.preset["ideal_min"])

    @property
    def ideal_max(self) -> int:
        return int(self.preset["ideal_max"])

    @property
    def ideal_mid(self) -> float:
        return (self.ideal_min + self.ideal_max) / 2.0

    @property
    def spacing(self) -> float:
        return float(self.preset["spacing"])

    def to_metadata(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "clip_duration_preset": self.duration_preset,
            "min_clips": self.min_clips if self.min_clips is not None else "auto",
        }
        if self.brief:
            payload["clip_brief"] = self.brief
        return payload

    def cache_token(self) -> str:
        brief_token = self.brief.strip().lower()
        min_token = "auto" if self.min_clips is None else str(self.min_clips)
        return f"{self.duration_preset}|{min_token}|{brief_token}"


def normalize_clip_brief(raw: Any) -> str:
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raise ValueError("clip_brief must be a string")
    brief = " ".join(raw.split()).strip()
    if len(brief) > CLIP_BRIEF_MAX_CHARS:
        return brief[:CLIP_BRIEF_MAX_CHARS]
    return brief


def normalize_duration_preset(raw: Any) -> str:
    if raw is None or raw == "":
        return "auto"
    if not isinstance(raw, str):
        raise ValueError("clip_duration_preset must be a string")
    preset = raw.strip().lower()
    if preset not in DURATION_PRESETS:
        raise ValueError(
            f"Invalid clip_duration_preset. Use one of: {', '.join(VALID_DURATION_PRESETS)}"
        )
    return preset


def normalize_min_clips(raw: Any, *, strict: bool = True) -> int | None:
    """Return an explicit render ceiling (Até N), or None for Auto.

    ``strict=True`` (API input) only allows the UI choices 5/10/15/20.
    ``strict=False`` (stored metadata) allows any positive int up to the
    discovery safety cap after physical clamping on create.
    """
    if raw is None or raw == "" or raw == "auto":
        return None
    if isinstance(raw, bool):
        raise ValueError("min_clips must be auto or one of 5, 10, 15, 20")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("min_clips must be auto or one of 5, 10, 15, 20") from exc
    if strict and value not in VALID_MIN_CLIPS:
        raise ValueError("min_clips must be auto or one of 5, 10, 15, 20")
    if not strict and not 1 <= value <= DEFAULT_DISCOVERY_SAFETY_CAP:
        raise ValueError(
            f"min_clips must be between 1 and {DEFAULT_DISCOVERY_SAFETY_CAP}"
        )
    return value


def parse_clip_selection_prefs(
    data: Mapping[str, Any] | None = None,
    *,
    duration_preset: Any = None,
    min_clips: Any = None,
    clip_brief: Any = None,
    strict_min_clips: bool = True,
) -> ClipSelectionPrefs:
    payload = dict(data or {})
    if duration_preset is not None:
        payload["clip_duration_preset"] = duration_preset
    if min_clips is not None:
        payload["min_clips"] = min_clips
    if clip_brief is not None:
        payload["clip_brief"] = clip_brief

    # Metadata reloads may contain physically clamped counts outside the UI set.
    if data is not None and duration_preset is None and min_clips is None:
        strict_min_clips = False

    return ClipSelectionPrefs(
        duration_preset=normalize_duration_preset(payload.get("clip_duration_preset")),
        min_clips=normalize_min_clips(
            payload.get("min_clips"), strict=strict_min_clips
        ),
        brief=normalize_clip_brief(payload.get("clip_brief")),
    )


def physical_max_clips(source_seconds: float, prefs: ClipSelectionPrefs) -> int:
    """How many non-overlapping clips of at least min_seconds fit in the source."""
    if source_seconds <= 0 or prefs.min_seconds <= 0:
        return 1
    return max(1, int(source_seconds // prefs.min_seconds))


def discovery_ceiling(
    source_seconds: float,
    prefs: ClipSelectionPrefs,
    *,
    max_clips: int,
    absolute_ceiling: int = DEFAULT_DISCOVERY_SAFETY_CAP,
) -> int:
    """Upper bound on how many excellent clips we may harvest (not a fill target)."""
    return max(
        1,
        min(
            absolute_ceiling,
            max_clips,
            physical_max_clips(source_seconds, prefs),
        ),
    )


def render_ceiling(
    prefs: ClipSelectionPrefs,
    *,
    discovery: int,
    auto_default: int = AUTO_RENDER_DEFAULT,
) -> int:
    """How many excellent clips to render (CPU cost). Explicit Até N or Auto default."""
    discovery = max(1, discovery)
    if prefs.min_clips is not None:
        return max(1, min(discovery, prefs.min_clips))
    return max(1, min(discovery, auto_default))


def auto_clip_count(source_seconds: float, prefs: ClipSelectionPrefs, ceiling: int) -> int:
    """Legacy density estimate — kept for tests/hints; Auto quantity no longer pads to this."""
    ceiling = max(1, ceiling)
    if source_seconds <= 0:
        return min(1, ceiling)
    if prefs.duration_preset == "auto":
        step = 5 * 60
    else:
        step = max(prefs.min_seconds, prefs.ideal_mid * prefs.spacing)
    estimated = ceil(max(0.0, source_seconds) / step)
    return max(1, min(ceiling, estimated))


def resolve_clip_budget(
    source_seconds: float,
    prefs: ClipSelectionPrefs,
    *,
    max_clips: int,
    absolute_ceiling: int = DEFAULT_DISCOVERY_SAFETY_CAP,
) -> int:
    """Render ceiling for this run (Até N or Auto default), clamped by discovery."""
    discovery = discovery_ceiling(
        source_seconds,
        prefs,
        max_clips=max_clips,
        absolute_ceiling=absolute_ceiling,
    )
    return render_ceiling(prefs, discovery=discovery)


def preset_is_feasible(source_seconds: float | None, prefs: ClipSelectionPrefs) -> bool:
    if source_seconds is None or source_seconds <= 0:
        return True
    return source_seconds >= prefs.min_seconds


def window_segment_target(clip_budget: int, *, windowed: bool) -> str:
    """How many excellent candidates to ask for in one analysis pass/window.

    ``clip_budget`` is the discovery ceiling — harvest enough for the gate/rerank,
    without asking the model to pad to a quota.
    """
    if not windowed:
        low = max(3, min(6, clip_budget))
        high = max(low, min(10, max(clip_budget, low + 2)))
        return f"{low}-{high}"
    # Windowed: harvest local excellents so global gate/rerank has material.
    per_window = max(4, min(10, ceil(clip_budget / 3) + 2))
    return f"{per_window}-{per_window + 3}"


def suggested_min_clips_for_preset(duration_preset: str) -> int | None:
    preset = DURATION_PRESETS.get(duration_preset, DURATION_PRESETS["auto"])
    suggestion = preset.get("suggest_min_clips", "auto")
    if suggestion == "auto":
        return None
    return int(suggestion)
