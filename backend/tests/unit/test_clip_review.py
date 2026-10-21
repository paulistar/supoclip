from src.clip_review import (
    append_evitar_phrases,
    build_candidate_rows,
    estimate_render_seconds,
    select_candidates_for_render,
)


def test_estimate_render_seconds_has_floor():
    assert estimate_render_seconds(5) >= 40
    assert estimate_render_seconds(120) > estimate_render_seconds(30)


def test_build_candidate_rows_marks_rendered_and_skipped():
    excellent = [
        {
            "start_time": "01:00",
            "end_time": "02:00",
            "text": "Claim concreto com payoff claro aqui.",
            "hook_title": "Claim concreto",
            "virality_score": 88,
            "hook_score": 20,
        },
        {
            "start_time": "10:00",
            "end_time": "11:00",
            "text": "Outro momento excelente.",
            "hook_title": "Outro momento",
            "virality_score": 80,
            "hook_score": 18,
        },
    ]
    rows = build_candidate_rows(
        excellent,
        rendered_starts=["01:00"],
        skipped_starts=["10:00"],
    )
    assert rows[0]["rendered"] is True
    assert rows[1]["skipped"] is True
    assert rows[0]["eta_seconds"] >= 40
    assert "Claim" in rows[0]["text_preview"]


def test_select_modes():
    excellent = [
        {"start_time": "01:00"},
        {"start_time": "10:00"},
        {"start_time": "20:00"},
    ]
    top = select_candidates_for_render(
        excellent, mode="top", count=2, rendered_starts=[], skipped_starts=[]
    )
    assert [s["start_time"] for s in top] == ["01:00", "10:00"]

    selected = select_candidates_for_render(
        excellent,
        mode="selected",
        start_times=["20:00", "01:00"],
        rendered_starts=["01:00"],
        skipped_starts=[],
    )
    assert [s["start_time"] for s in selected] == ["20:00"]

    all_pending = select_candidates_for_render(
        excellent, mode="all", rendered_starts=["01:00"], skipped_starts=["10:00"]
    )
    assert [s["start_time"] for s in all_pending] == ["20:00"]


def test_append_evitar_phrases():
    assert "EVITAR" in append_evitar_phrases("", ["Balde"])
    brief = append_evitar_phrases("Clips bons.", ["Tema X", "Tema Y"])
    assert "EVITAR" in brief
    assert "Tema X" in brief
    extended = append_evitar_phrases("EVITAR: Antigo", ["Novo"])
    assert "Antigo" in extended and "Novo" in extended
