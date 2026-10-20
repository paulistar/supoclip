from types import SimpleNamespace

from pydantic_ai.models.ollama import OllamaModel

from src.ai import (
    IDEAL_CLIP_MAX_SECONDS,
    IDEAL_CLIP_MIN_SECONDS,
    MAX_ACCEPTED_CLIP_SECONDS,
    MIN_ACCEPTED_CLIP_SECONDS,
    TranscriptSegment,
    _build_transcript_model,
    _choose_repaired_bounds,
    _extract_transcript_text,
    _format_transcript_timestamp,
    _get_missing_llm_key_error,
    _parse_transcript_spans,
    _parse_transcript_timestamp_seconds,
    build_transcript_analysis_prompt,
    transcript_analysis_system_prompt,
)
from src.clip_selection import ClipSelectionPrefs


def test_system_prompt_enforces_grounding_rules():
    assert "extraction and ranking, not creative rewriting" in (
        transcript_analysis_system_prompt
    )
    assert "Never invent facts, tone, context, or transitions" in (
        transcript_analysis_system_prompt
    )
    assert "Each selected segment must map to one contiguous range" in (
        transcript_analysis_system_prompt
    )
    assert "Do not judge, moralize, or downgrade a segment" in (
        transcript_analysis_system_prompt
    )
    assert "Duration bounds come from the task prompt" in transcript_analysis_system_prompt
    assert "no fixed 30/60/90-second targets" in transcript_analysis_system_prompt
    assert "Default clip length is 20-60" not in transcript_analysis_system_prompt
    assert "Prefer 25-50" not in transcript_analysis_system_prompt
    assert "Bad picks include intros" in transcript_analysis_system_prompt
    assert "Return valid JSON only" in transcript_analysis_system_prompt
    assert "Do not use \"segment\" as an output field. Use \"text\"." in (
        transcript_analysis_system_prompt
    )
    assert "SHORT-FORM RETENTION (WATCH + hook-writer)" in (
        transcript_analysis_system_prompt
    )
    assert "Cash the promise" in transcript_analysis_system_prompt
    assert "Win the first ~3 seconds" in transcript_analysis_system_prompt
    assert "Hard gate" in transcript_analysis_system_prompt


def test_build_transcript_analysis_prompt_includes_watch_brief_guidance():
    prompt = build_transcript_analysis_prompt(
        transcript="[00:12 - 00:21] A strong opening line",
        prefs=ClipSelectionPrefs(brief="Prioritize curiosity hooks for paid traffic"),
    )
    assert "User brief" in prompt
    assert "Prioritize curiosity hooks for paid traffic" in prompt
    assert "cash the promise" in prompt
    assert "Win the first ~3s" in prompt or "win the first ~3s" in prompt


def test_build_transcript_analysis_prompt_auto_uses_adaptive_duration_language():
    prompt = build_transcript_analysis_prompt(
        transcript="[00:12 - 00:21] A strong opening line",
        prefs=ClipSelectionPrefs(duration_preset="auto"),
    )
    assert "Continuous natural duration per clip" in prompt
    assert "do not aim for fixed 30/60/90-second lengths" in prompt
    assert "Expand contiguous same-topic lines" in prompt
    assert "cite the opening line" in prompt
    assert f"Most selected clips should be {IDEAL_CLIP_MIN_SECONDS}-{IDEAL_CLIP_MAX_SECONDS} seconds." not in prompt
    assert f"Accepted duration range for this run: {MIN_ACCEPTED_CLIP_SECONDS}-{MAX_ACCEPTED_CLIP_SECONDS} seconds." in prompt


def test_build_transcript_analysis_prompt_short_keeps_ideal_guidance():
    prompt = build_transcript_analysis_prompt(
        transcript="[00:12 - 00:21] A strong opening line",
        prefs=ClipSelectionPrefs(duration_preset="short"),
    )
    assert "Most selected clips should be 20-28 seconds." in prompt
    assert "Continuous natural duration per clip" not in prompt
    assert "do not aim for fixed 30/60/90-second lengths" not in prompt


def test_build_transcript_analysis_prompt_requires_transcript_fidelity():
    prompt = build_transcript_analysis_prompt(
        transcript="[00:12 - 00:21] A strong opening line"
    )

    assert "Do not fabricate or embellish content." in prompt
    assert "Do not merge separate non-contiguous moments into one segment." in prompt
    assert "If there is a tradeoff between \"viral\" and \"accurate\", choose accuracy." in prompt
    assert "Do not reject or penalize a segment simply because of the subject matter" in prompt
    assert "Continuous natural duration per clip" in prompt
    assert "viewer would understand and care without seeing the rest" in prompt
    assert "Return one valid JSON object and nothing else." in prompt
    assert "No Markdown, headings, bullets, code fences" in prompt
    assert "[00:12 - 00:21] A strong opening line" in prompt


def test_build_transcript_analysis_prompt_mentions_broll_only_when_enabled():
    without_broll = build_transcript_analysis_prompt(
        transcript="[00:12 - 00:21] A strong opening line"
    )
    with_broll = build_transcript_analysis_prompt(
        transcript="[00:12 - 00:21] A strong opening line",
        include_broll=True,
    )

    assert "B-roll opportunities" not in without_broll
    assert "B-roll opportunities" in with_broll


def test_ollama_llm_builds_native_ollama_model():
    runtime_config = SimpleNamespace(
        llm="ollama:gpt-oss:20b",
        ollama_api_key=None,
        resolve_ollama_base_url=lambda: "http://ollama.example/v1",
    )

    model = _build_transcript_model(runtime_config)

    assert isinstance(model, OllamaModel)
    assert model.model_name == "gpt-oss:20b"
    assert model.base_url == "http://ollama.example/v1/"


def test_parse_transcript_timestamp_supports_minute_and_hour_formats():
    assert _parse_transcript_timestamp_seconds("02:35") == 155
    assert _parse_transcript_timestamp_seconds("01:02:35") == 3755
    assert _format_transcript_timestamp(155) == "02:35"
    assert _format_transcript_timestamp(3755) == "01:02:35"
    assert MIN_ACCEPTED_CLIP_SECONDS == 30
    assert MAX_ACCEPTED_CLIP_SECONDS == 120
    assert IDEAL_CLIP_MIN_SECONDS == 30
    assert IDEAL_CLIP_MAX_SECONDS == 90


def test_transcript_span_helpers_repair_near_miss_durations():
    spans = _parse_transcript_spans(
        "\n".join(
            [
                "[00:00 - 00:10] Setup",
                "[00:10 - 00:24] Short highlight",
                "[00:24 - 00:36] Payoff",
                "[00:36 - 01:20] Too much background",
            ]
        )
    )

    assert _extract_transcript_text(spans, 10, 36) == "Short highlight Payoff"
    # 14s < min 30 → expand to contiguous payoff ending at 36s
    assert _choose_repaired_bounds(spans, 10, 24) == (10, 36)
    # 80s is within auto max 120 → no trim needed
    assert _choose_repaired_bounds(spans, 0, 80) is None


def test_choose_repaired_bounds_trims_to_max_not_ideal():
    # Single long span: no intermediate end inside the cap → fall back to start+max_s
    spans = _parse_transcript_spans("[00:00 - 03:00] One continuous overlong segment")
    # 180s > max 120 → trim to start+max_s (120), not ideal_max (90)
    assert _choose_repaired_bounds(spans, 0, 180) == (0, 120)


def test_choose_repaired_bounds_auto_allows_duration_outside_ideal_band():
    spans = _parse_transcript_spans(
        "\n".join(
            [
                "[01:40 - 01:50] Short core",
                "[01:50 - 03:20] Continues past ideal_max",
            ]
        )
    )
    auto = ClipSelectionPrefs(duration_preset="auto")
    # 10s core → expand to 100s (> ideal_max 90); auto must not reject for mid-band pull
    repaired = _choose_repaired_bounds(spans, 100, 110, prefs=auto)
    assert repaired == (100, 200)
    assert repaired[1] - repaired[0] > auto.ideal_max


def test_choose_repaired_bounds_non_auto_still_uses_ideal_penalty():
    spans = _parse_transcript_spans(
        "\n".join(
            [
                "[00:10 - 00:20] Short core",
                "[00:20 - 00:25] Tiny add",
                "[00:25 - 00:30] Ideal landing",
                "[00:30 - 00:40] Max edge",
            ]
        )
    )
    short = ClipSelectionPrefs(duration_preset="short")
    # 10s < min 15 → expand; ideal_penalty should prefer ~20-28s over bare minimum
    repaired = _choose_repaired_bounds(spans, 10, 20, prefs=short)
    assert repaired is not None
    duration = repaired[1] - repaired[0]
    assert short.min_seconds <= duration <= short.max_seconds
    assert duration >= short.ideal_min


def test_transcript_segment_normalizes_percent_relevance_score():
    segment = TranscriptSegment(
        start_time="00:00",
        end_time="00:30",
        text="A complete standalone moment with useful context.",
        relevance_score=100,
    )

    assert segment.relevance_score == 1.0


def test_llm_validation_rejects_unsupported_or_incomplete_model_names():
    runtime_config = SimpleNamespace(
        google_api_key=None,
        openai_api_key=None,
        anthropic_api_key=None,
    )

    assert "Unsupported LLM provider" in _get_missing_llm_key_error(
        "local:model", runtime_config
    )
    assert "missing a model name" in _get_missing_llm_key_error(
        "ollama:", runtime_config
    )
    assert _get_missing_llm_key_error("ollama:gpt-oss:20b", runtime_config) is None
