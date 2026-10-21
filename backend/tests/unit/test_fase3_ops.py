"""Fase 3 — generate-more selection + drop telemetry shape."""

from src.ai import apply_quality_gate
from src.services.video_service import VideoService
from tests.unit.test_quality_gate import _seg


def test_drop_log_reasons_are_structured():
    strong = _seg("02:00", "03:00", total=90, hook=22)
    weak = _seg("08:00", "08:40", total=50, hook=8)
    near = _seg("02:10", "03:00", total=88, hook=21)
    _kept, drops = apply_quality_gate([strong, weak, near])
    reasons = {d["reason"] for d in drops}
    assert "low_total" in reasons or "low_hook" in reasons
    assert "too_close_in_time" in reasons
    assert all("start_time" in d and "reason" in d for d in drops)


def test_segment_payload_preserves_scores_for_cache():
    payload = VideoService._segment_to_payload(
        _seg("01:00", "02:00", total=88, hook=20, title="Claim forte")
    )
    assert payload["hook_title"] == "Claim forte"
    assert payload["virality_score"] == 88
    assert payload["start_time"] == "01:00"


def test_generate_more_picks_unrendered_excellents():
    excellent = [
        {"start_time": "01:00", "end_time": "02:00"},
        {"start_time": "10:00", "end_time": "11:00"},
        {"start_time": "20:00", "end_time": "21:00"},
    ]
    rendered_starts = {"01:00"}
    remaining = [
        seg
        for seg in excellent
        if str(seg.get("start_time") or "").strip() not in rendered_starts
    ]
    assert [s["start_time"] for s in remaining[:2]] == ["10:00", "20:00"]
