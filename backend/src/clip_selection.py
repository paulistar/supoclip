"""Clip duration presets, quantity budget, and user brief helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Mapping

CLIP_BRIEF_MAX_CHARS = 800
VALID_DURATION_PRESETS = (
    "auto",
    "short",
    "long",
    "viral",
    "monetization",
    "podcast",
)
VALID_MIN_CLIPS = (5, 10, 15, 20)

# Spacing multiplies ideal midpoint for Auto quantity density.
# Longer formats leave more room between clips.
DURATION_PRESETS: dict[str, dict[str, Any]] = {
    "auto": {
        "label": "Auto (20-60s)",
        "min_seconds": 20,
        "max_seconds": 60,
        "ideal_min": 25,
        "ideal_max": 50,
        "spacing": 1.1,
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
        raise ValueError(f"clip_brief must be at most {CLIP_BRIEF_MAX_CHARS} characters")
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
    """Return an explicit minimum clip count, or None for Auto.

    ``strict=True`` (API input) only allows the UI choices 5/10/15/20.
    ``strict=False`` (stored metadata) allows any positive int up to 20 after
    physical clamping on create.
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
    if not strict and not 1 <= value <= 20:
        raise ValueError("min_clips must be between 1 and 20")
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


def auto_clip_count(source_seconds: float, prefs: ClipSelectionPrefs, ceiling: int) -> int:
    """Density-aware Auto quantity for the selected duration preset."""
    ceiling = max(1, ceiling)
    if source_seconds <= 0:
        return min(5, ceiling)
    # Keep the historical ~1 clip / 5 minutes for the Auto duration preset so
    # Auto/Auto stays close to prior behavior; other presets use ideal spacing.
    if prefs.duration_preset == "auto":
        step = 5 * 60
        floor = min(5, ceiling)
    else:
        step = max(prefs.min_seconds, prefs.ideal_mid * prefs.spacing)
        floor = (
            1
            if prefs.duration_preset in {"viral", "monetization", "podcast"}
            else min(5, ceiling)
        )
    estimated = ceil(max(0.0, source_seconds) / step)
    return max(floor, min(ceiling, estimated))


def resolve_clip_budget(
    source_seconds: float,
    prefs: ClipSelectionPrefs,
    *,
    max_clips: int,
    absolute_ceiling: int = 20,
) -> int:
    ceiling = max(1, min(absolute_ceiling, max_clips, physical_max_clips(source_seconds, prefs)))
    if prefs.min_clips is not None:
        return max(1, min(ceiling, prefs.min_clips))
    return auto_clip_count(source_seconds, prefs, ceiling)


def preset_is_feasible(source_seconds: float | None, prefs: ClipSelectionPrefs) -> bool:
    if source_seconds is None or source_seconds <= 0:
        return True
    return source_seconds >= prefs.min_seconds


def window_segment_target(clip_budget: int, *, windowed: bool) -> str:
    """How many candidates to ask the model for in one analysis pass/window."""
    if not windowed:
        low = max(2, min(5, clip_budget))
        high = max(low, min(8, max(clip_budget, low + 1)))
        return f"{low}-{high}"
    # Windowed: ask for enough local candidates so the global rerank can fill budget.
    per_window = max(3, min(8, ceil(clip_budget / 2)))
    return f"{per_window}-{per_window + 2}"


def suggested_min_clips_for_preset(duration_preset: str) -> int | None:
    preset = DURATION_PRESETS.get(duration_preset, DURATION_PRESETS["auto"])
    suggestion = preset.get("suggest_min_clips", "auto")
    if suggestion == "auto":
        return None
    return int(suggestion)
