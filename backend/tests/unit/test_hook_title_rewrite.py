"""Fase 2 — dedicated hook_title rewrite pass."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src import ai


def _seg(text: str, title: str | None = None) -> ai.TranscriptSegment:
    return ai.TranscriptSegment(
        start_time="01:00",
        end_time="02:00",
        text=text,
        hook_title=title,
        virality=ai.ViralityAnalysis(
            hook_score=20,
            engagement_score=20,
            value_score=20,
            shareability_score=20,
            total_score=80,
        ),
    )


def test_hook_title_rewrite_prompt_lists_segments():
    segments = [
        _seg("Eu perdi R$40 mil em precificação.", "A pergunta sobre preço"),
        _seg("CPA caiu 37% em sete dias.", None),
    ]
    prompt = ai.build_hook_title_rewrite_prompt(segments)
    assert "segment_index: 0" in prompt
    assert "segment_index: 1" in prompt
    assert "A pergunta sobre preço" in prompt
    assert "CPA caiu 37%" in prompt


async def test_rewrite_hook_titles_applies_batch(monkeypatch):
    segments = [
        _seg("Eu perdi R$40 mil em precificação.", "A pergunta sobre preço"),
        _seg("Ninguém te conta que funil sem oferta queima mídia.", "Let's talk about funnels"),
    ]
    batch = ai.HookTitleBatch(
        titles=[
            ai.HookTitleItem(segment_index=0, hook_title="O erro de R$40 mil"),
            ai.HookTitleItem(
                segment_index=1, hook_title="Funil sem oferta queima mídia"
            ),
        ]
    )
    agent = SimpleNamespace(
        run=AsyncMock(return_value=SimpleNamespace(output=batch))
    )
    monkeypatch.setattr(ai, "get_hook_title_agent", lambda: agent)

    result = await ai.rewrite_hook_titles(segments)

    assert result[0].hook_title == "O erro de R$40 mil"
    assert result[1].hook_title == "Funil sem oferta queima mídia"
    agent.run.assert_awaited_once()


async def test_rewrite_hook_titles_keeps_draft_on_failure(monkeypatch):
    segments = [_seg("Claim concreto com payoff.", "CPA caiu 37% em sete dias")]
    monkeypatch.setattr(
        ai,
        "get_hook_title_agent",
        lambda: (_ for _ in ()).throw(RuntimeError("no key")),
    )

    result = await ai.rewrite_hook_titles(segments)

    assert result[0].hook_title == "CPA caiu 37% em sete dias"


def test_hook_title_system_prompt_has_formulas_and_anti_meta():
    assert "Curiosity gap" in ai.hook_title_rewrite_system_prompt
    assert "Anti-meta" in ai.hook_title_rewrite_system_prompt
    assert "A pergunta sobre" in ai.hook_title_rewrite_system_prompt
