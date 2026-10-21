"""Unit tests for the excellence quality gate (Fase 1)."""

from src.ai import (
    TranscriptSegment,
    ViralityAnalysis,
    apply_quality_gate,
    is_meta_hook_title,
    sanitize_hook_title,
)


def _seg(
    start: str,
    end: str,
    *,
    total: int = 84,
    hook: int = 20,
    text: str = "Concrete claim with a sharp payoff here.",
    title: str | None = "CPA caiu 37% em sete dias",
) -> TranscriptSegment:
    rest = max(0, total - hook)
    thirds = [rest // 3, rest // 3, rest - 2 * (rest // 3)]
    return TranscriptSegment(
        start_time=start,
        end_time=end,
        text=text,
        hook_title=title,
        virality=ViralityAnalysis(
            hook_score=hook,
            engagement_score=thirds[0],
            value_score=thirds[1],
            shareability_score=thirds[2],
            total_score=total,
        ),
    )


def test_quality_gate_keeps_strong_and_drops_weak():
    strong = _seg("02:00", "03:00", total=88, hook=22)
    weak = _seg("08:00", "08:40", total=55, hook=10)
    kept, drops = apply_quality_gate([strong, weak])
    assert kept == [strong]
    assert any(d["reason"] == "low_total" for d in drops)


def test_quality_gate_drops_far_from_top():
    top = _seg("01:00", "02:00", total=95, hook=24)
    tail = _seg("20:00", "21:00", total=72, hook=18)  # 95-15=80 floor → drop
    kept, drops = apply_quality_gate([top, tail])
    assert kept == [top]
    assert any(d["reason"] == "far_from_top" for d in drops)


def test_quality_gate_drops_weak_opener_greeting():
    greeting = _seg(
        "00:00",
        "00:40",
        total=90,
        hook=22,
        text="Olá pessoal, bem-vindos ao vídeo de hoje vamos falar de marketing.",
    )
    strong = _seg("12:00", "13:00", total=88, hook=20)
    kept, drops = apply_quality_gate([greeting, strong])
    assert kept == [strong]
    assert any(d["reason"] == "weak_opener" for d in drops)


def test_quality_gate_enforces_temporal_gap():
    first = _seg("30:00", "30:50", total=90, hook=22)
    near = _seg("30:20", "31:00", total=88, hook=21)
    far = _seg("45:00", "45:50", total=86, hook=20)
    kept, drops = apply_quality_gate([first, near, far])
    assert [s.start_time for s in kept] == ["30:00", "45:00"]
    assert any(d["reason"] == "too_close_in_time" for d in drops)


def test_quality_gate_fallback_keeps_single_best():
    only_weak = _seg("00:00", "00:40", total=40, hook=8)
    kept, _drops = apply_quality_gate([only_weak])
    assert kept == [only_weak]


def test_meta_hook_titles_are_rejected():
    assert is_meta_hook_title("A pergunta sobre escala de mídia")
    assert is_meta_hook_title("Everything you need to know about ads")
    assert sanitize_hook_title("A pergunta sobre escala de mídia") is None
    assert sanitize_hook_title("Escalar budget sem oferta queima caixa") is not None
