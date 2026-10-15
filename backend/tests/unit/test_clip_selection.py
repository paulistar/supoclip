from src.clip_selection import (
    ClipSelectionPrefs,
    auto_clip_count,
    parse_clip_selection_prefs,
    physical_max_clips,
    resolve_clip_budget,
    window_segment_target,
)


def test_parse_defaults_and_brief():
    prefs = parse_clip_selection_prefs({})
    assert prefs.duration_preset == "auto"
    assert prefs.min_clips is None
    assert prefs.brief == ""

    prefs = parse_clip_selection_prefs(
        duration_preset="long",
        min_clips=15,
        clip_brief="  Hook forte  ",
    )
    assert prefs.duration_preset == "long"
    assert prefs.min_clips == 15
    assert prefs.brief == "Hook forte"


def test_auto_quantity_keeps_legacy_density_for_auto_preset():
    prefs = ClipSelectionPrefs(duration_preset="auto")
    assert auto_clip_count(5 * 60, prefs, 20) == 5
    assert auto_clip_count(45 * 60, prefs, 20) == 9
    assert resolve_clip_budget(90 * 60, prefs, max_clips=10) == 10


def test_podcast_auto_is_sparser_and_physically_clamped():
    prefs = ClipSelectionPrefs(duration_preset="podcast")
    assert physical_max_clips(60 * 60, prefs) == 6
    assert resolve_clip_budget(60 * 60, prefs, max_clips=20) <= 6


def test_explicit_min_clips_is_floor_until_physical_cap():
    prefs = ClipSelectionPrefs(duration_preset="short", min_clips=15)
    assert resolve_clip_budget(60 * 60, prefs, max_clips=20) == 15
    assert resolve_clip_budget(3 * 60, prefs, max_clips=20) == physical_max_clips(3 * 60, prefs)


def test_window_segment_target_scales_with_budget():
    assert window_segment_target(5, windowed=False) == "5-6"
    assert window_segment_target(15, windowed=True).startswith("8-")
