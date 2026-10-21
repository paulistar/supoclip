from src.clip_selection import (
    AUTO_RENDER_DEFAULT,
    CLIP_BRIEF_MAX_CHARS,
    ClipSelectionPrefs,
    auto_clip_count,
    discovery_ceiling,
    normalize_clip_brief,
    parse_clip_selection_prefs,
    physical_max_clips,
    render_ceiling,
    resolve_clip_budget,
    window_segment_target,
)


def test_parse_defaults_and_brief():
    prefs = parse_clip_selection_prefs({})
    assert prefs.duration_preset == "auto"
    assert prefs.min_clips is None
    assert prefs.brief == ""
    assert prefs.min_seconds == 30
    assert prefs.max_seconds == 120
    assert prefs.ideal_min == 30
    assert prefs.ideal_max == 90
    assert prefs.spacing == 1.2
    assert prefs.preset["label"] == "Auto (30s–2min)"

    prefs = parse_clip_selection_prefs(
        duration_preset="long",
        min_clips=15,
        clip_brief="  Hook forte  ",
    )
    assert prefs.duration_preset == "long"
    assert prefs.min_clips == 15
    assert prefs.brief == "Hook forte"


def test_clip_brief_accepts_over_800_and_soft_truncates():
    long_brief = "x" * 801
    assert normalize_clip_brief(long_brief) == long_brief
    prefs = parse_clip_selection_prefs(clip_brief=long_brief)
    assert prefs.brief == long_brief

    oversized = "y" * (CLIP_BRIEF_MAX_CHARS + 50)
    truncated = normalize_clip_brief(oversized)
    assert len(truncated) == CLIP_BRIEF_MAX_CHARS
    assert truncated == "y" * CLIP_BRIEF_MAX_CHARS


def test_auto_discovery_is_physical_not_density_floor():
    prefs = ClipSelectionPrefs(duration_preset="auto")
    # 1h source → physical max at 30s min = 120, safety 60 → discovery 60
    assert discovery_ceiling(60 * 60, prefs, max_clips=60) == 60
    # Render Auto default 15, not a fill quota
    assert render_ceiling(prefs, discovery=60) == AUTO_RENDER_DEFAULT
    assert resolve_clip_budget(60 * 60, prefs, max_clips=60) == AUTO_RENDER_DEFAULT
    # No density floor of 5 for short sources
    assert discovery_ceiling(5 * 60, prefs, max_clips=60) == physical_max_clips(
        5 * 60, prefs
    )
    assert resolve_clip_budget(5 * 60, prefs, max_clips=60) == min(
        AUTO_RENDER_DEFAULT, physical_max_clips(5 * 60, prefs)
    )


def test_legacy_auto_clip_count_estimate_has_no_pad_floor():
    prefs = ClipSelectionPrefs(duration_preset="auto")
    assert auto_clip_count(5 * 60, prefs, 20) == 1
    assert auto_clip_count(45 * 60, prefs, 20) == 9


def test_podcast_discovery_is_physically_clamped():
    prefs = ClipSelectionPrefs(duration_preset="podcast")
    assert physical_max_clips(60 * 60, prefs) == 6
    assert discovery_ceiling(60 * 60, prefs, max_clips=60) == 6
    assert resolve_clip_budget(60 * 60, prefs, max_clips=60) == 6


def test_explicit_ate_n_is_render_ceiling_until_physical_cap():
    prefs = ClipSelectionPrefs(duration_preset="short", min_clips=15)
    assert resolve_clip_budget(60 * 60, prefs, max_clips=60) == 15
    assert resolve_clip_budget(3 * 60, prefs, max_clips=60) == physical_max_clips(
        3 * 60, prefs
    )


def test_window_segment_target_harvests_without_quota_language():
    assert window_segment_target(5, windowed=False) == "5-7"
    assert window_segment_target(15, windowed=True).startswith("7-")
