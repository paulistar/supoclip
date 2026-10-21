"""Golden-set / prompt regression for MartClip excellence criteria (Fase 1b)."""

from __future__ import annotations

import json
from pathlib import Path

from src.ai import (
    TRANSCRIPT_ANALYSIS_CACHE_VERSION,
    TranscriptSegment,
    ViralityAnalysis,
    _extract_forbidden_phrases,
    _filter_segments_by_brief,
    apply_quality_gate,
    build_transcript_analysis_prompt,
    is_meta_hook_title,
    sanitize_hook_title,
    transcript_analysis_system_prompt,
)
from src.clip_selection import ClipSelectionPrefs

GOLDEN_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "golden_clips" / "cases.json"
)


def _load_cases() -> list[dict]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _scored(
    start: str,
    end: str,
    text: str,
    *,
    total: int,
    hook: int,
    title: str | None = None,
) -> TranscriptSegment:
    rest = max(0, total - hook)
    a = rest // 3
    b = rest // 3
    c = rest - a - b
    return TranscriptSegment(
        start_time=start,
        end_time=end,
        text=text,
        hook_title=title,
        virality=ViralityAnalysis(
            hook_score=hook,
            engagement_score=min(25, a),
            value_score=min(25, b),
            shareability_score=min(25, c),
            total_score=total,
        ),
    )


def test_cache_version_tracks_quality_count():
    assert "v11" in TRANSCRIPT_ANALYSIS_CACHE_VERSION
    assert "quality-count" in TRANSCRIPT_ANALYSIS_CACHE_VERSION


def test_system_prompt_is_quality_only_not_quota():
    assert "never pad toward a count" in transcript_analysis_system_prompt.lower()
    assert "Prefer meeting the requested count" not in transcript_analysis_system_prompt
    assert "Do NOT pad" in build_transcript_analysis_prompt(
        transcript="[00:00 - 00:30] Claim.",
        prefs=ClipSelectionPrefs(),
        clip_budget=15,
    )
    assert "quality-only" in build_transcript_analysis_prompt(
        transcript="[00:00 - 00:30] Claim.",
        clip_budget=15,
    )


def test_golden_strong_hooks_gate():
    case = next(c for c in _load_cases() if c["id"] == "strong_hooks_pt")
    # Simulate LLM harvest with inflated scores on greetings/CTA too.
    candidates = [
        _scored(
            "00:00",
            "00:18",
            "Olá pessoal, bem-vindos ao vídeo de hoje. Então vamos começar.",
            total=92,
            hook=22,
            title="Introdução do episódio",
        ),
        _scored(
            "02:00",
            "03:05",
            "Eu perdi R$40 mil em um único erro de precificação. "
            "A regra: nunca negociar preço antes de travar o escopo.",
            total=90,
            hook=22,
            title="O erro de R$40 mil",
        ),
        _scored(
            "08:00",
            "08:20",
            "Inscreva-se no canal e deixe o like se gostou.",
            total=50,
            hook=8,
            title="Se inscreva",
        ),
        _scored(
            "12:00",
            "13:10",
            "Ninguém te conta que funil sem oferta clara só queima mídia. "
            "Só o ângulo com preço no gancho converteu o dobro.",
            total=88,
            hook=21,
            title="Funil sem oferta queima mídia",
        ),
    ]
    kept, drops = apply_quality_gate(candidates)
    starts = {s.start_time for s in kept}
    assert "02:00" in starts
    assert "12:00" in starts
    assert "00:00" not in starts  # weak opener
    assert "08:00" not in starts  # low score CTA
    assert any(d["reason"] == "weak_opener" for d in drops)


def test_golden_brief_balde_still_hard_filtered():
    case = next(c for c in _load_cases() if c["id"] == "brief_balde_skip")
    phrases = _extract_forbidden_phrases(case["brief"])
    assert "nao adianta aumentar o balde" in phrases

    balde = _scored(
        "05:00",
        "05:50",
        "Não adianta aumentar o balde se a torneira está fechada.",
        total=95,
        hook=24,
        title="Não adianta aumentar o balde",
    )
    allowed = _scored(
        "20:00",
        "20:55",
        "O desejo do cliente define o ângulo; CPA caiu 37% em sete dias.",
        total=90,
        hook=22,
        title="CPA caiu 37% em sete dias",
    )
    filtered = _filter_segments_by_brief(
        [balde, allowed], case["brief"], clip_budget=10
    )
    assert [s.start_time for s in filtered] == ["20:00"]
    kept, _drops = apply_quality_gate(filtered)
    assert [s.start_time for s in kept] == ["20:00"]


def test_golden_near_duplicate_gap():
    candidates = [
        _scored(
            "30:00",
            "30:50",
            "Três métricas matam qualquer campanha: CTR baixo, CPM alto e oferta fraca.",
            total=90,
            hook=22,
        ),
        _scored(
            "30:40",
            "31:10",
            "Repito: oferta fraca mata campanha mesmo com CTR bonito.",
            total=88,
            hook=21,
        ),
        _scored(
            "45:00",
            "45:55",
            "Eu demiti o funil de 12 etapas e fechei em três.",
            total=86,
            hook=20,
        ),
    ]
    kept, drops = apply_quality_gate(candidates)
    assert [s.start_time for s in kept] == ["30:00", "45:00"]
    assert any(d["reason"] == "too_close_in_time" for d in drops)


def test_golden_meta_title_rejected_claim_kept():
    case = next(c for c in _load_cases() if c["id"] == "meta_title_vs_claim")
    meta = next(lab for lab in case["labels"] if lab.get("expect") == "drop_title")
    claim = next(lab for lab in case["labels"] if lab.get("expect") == "keep_title")
    assert is_meta_hook_title(meta["meta_title"])
    assert sanitize_hook_title(meta["meta_title"]) is None
    assert sanitize_hook_title(claim["claim_title"]) is not None
