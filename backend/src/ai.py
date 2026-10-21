"""
AI-related functions for transcript analysis with enhanced precision and virality scoring.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
import asyncio
import logging
import math
import random
import re
import unicodedata

import httpx
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models import Model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic import AliasChoices, BaseModel, Field, PrivateAttr, field_validator

from .clip_selection import (
    AUTO_RENDER_DEFAULT,
    CLIP_BRIEF_MAX_CHARS,
    DEFAULT_DISCOVERY_SAFETY_CAP,
    ClipSelectionPrefs,
    discovery_ceiling,
    parse_clip_selection_prefs,
    render_ceiling,
    window_segment_target,
)
from .config import Config, get_config
from .runtime_settings import apply_settings_to_process_env

logger = logging.getLogger(__name__)

# Defaults match the "auto" duration preset (used when prefs are omitted).
# ideal_* is soft metadata only — not a hard generation target for auto.
IDEAL_CLIP_MIN_SECONDS = 30
IDEAL_CLIP_MAX_SECONDS = 90
MIN_ACCEPTED_CLIP_SECONDS = 30
MAX_ACCEPTED_CLIP_SECONDS = 120
TRANSCRIPT_ANALYSIS_CACHE_VERSION = "windowed-rerank-v11-quality-count"
TRANSCRIPT_ANALYSIS_MAX_ATTEMPTS = 3
TRANSCRIPT_ANALYSIS_TIMEOUT_SECONDS = 600
# Long transcripts are analysed in overlapping windows so every part of the
# source gets the model's full attention; candidates are then reranked together.
ANALYSIS_WINDOW_SECONDS = 12 * 60
ANALYSIS_WINDOW_OVERLAP_SECONDS = 2 * 60
ANALYSIS_SINGLE_WINDOW_MAX_SECONDS = 15 * 60
ANALYSIS_WINDOW_CONCURRENCY = 3
ANALYSIS_WINDOW_SEGMENT_TARGET = "4-7"
# Wall-clock budget for all windows together. Each window additionally keeps
# its own TRANSCRIPT_ANALYSIS_TIMEOUT_SECONDS budget once it starts running.
ANALYSIS_TOTAL_TIMEOUT_SECONDS = 1800
RERANK_TIMEOUT_SECONDS = 240
CANDIDATE_DEDUPE_OVERLAP_RATIO = 0.5
# How far a model timestamp may sit from a transcript line boundary and still be
# snapped onto it instead of the whole clip being dropped.
SEGMENT_SNAP_TOLERANCE_SECONDS = 8
# Discovery safety + quality gate (quantity Auto is quality-first, not pad-to-N).
MIN_FINAL_CLIPS = 1
MAX_FINAL_CLIPS = DEFAULT_DISCOVERY_SAFETY_CAP
SECONDS_PER_FINAL_CLIP = 5 * 60
QUALITY_GATE_MIN_TOTAL = 70
QUALITY_GATE_MIN_HOOK = 16
QUALITY_GATE_TOP_GAP = 15
QUALITY_GATE_MIN_START_GAP_SECONDS = 45
TRANSIENT_MODEL_STATUS_CODES = {408, 429, 500, 502, 503, 504, 529}
HOOK_TITLE_MAX_CHARS = 64
HOOK_TITLE_MAX_WORDS = 10
HOOK_TITLE_REWRITE_TIMEOUT_SECONDS = 120
# The transcript formatter does not wrap minutes into hours, so sources of 100+
# minutes produce labels such as "100:05".
TRANSCRIPT_SPAN_RE = re.compile(
    r"^\[(?P<start>\d{1,3}:\d{2}(?::\d{2})?)\s*-\s*"
    r"(?P<end>\d{1,3}:\d{2}(?::\d{2})?)\]\s*(?P<text>.*)$"
)
TRANSCRIPT_LINE_PATTERN = re.compile(
    r"^\[(\d{1,3}:\d{2}(?::\d{2})?)\s*-\s*(\d{1,3}:\d{2}(?::\d{2})?)\]\s*(.*)$"
)
SIGNAL_LINE_PATTERN = re.compile(r"^- \[(?P<start>\d{1,3}:\d{2}(?::\d{2})?)\s*-")
SPEAKER_PREFIX_PATTERN = re.compile(r"^Speaker [^:]+:\s*")


def _normalize_transcript_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9']+", " ", value.lower())).strip()


def _parse_transcript_lines(transcript: str) -> List[Dict[str, Any]]:
    lines: List[Dict[str, Any]] = []
    for raw_line in transcript.splitlines():
        match = TRANSCRIPT_LINE_PATTERN.match(raw_line.strip())
        if not match:
            continue
        line_text = SPEAKER_PREFIX_PATTERN.sub("", match.group(3).strip())
        lines.append(
            {
                "start_time": match.group(1),
                "end_time": match.group(2),
                "start_seconds": _parse_transcript_timestamp_seconds(match.group(1)),
                "end_seconds": _parse_transcript_timestamp_seconds(match.group(2)),
                "text": line_text,
            }
        )
    return lines


def _snap_bounds_to_transcript_lines(
    transcript_lines: List[Dict[str, Any]],
    start_seconds: int,
    end_seconds: int,
    tolerance_seconds: int = SEGMENT_SNAP_TOLERANCE_SECONDS,
) -> Optional[tuple[Dict[str, Any], Dict[str, Any]]]:
    """Snap model timestamps onto the nearest transcript line start and end.

    Models often pick a second that sits inside a line rather than on its
    boundary. Grounding needs exact line labels, so move each bound to the
    closest boundary within the tolerance instead of discarding the clip.
    """
    if not transcript_lines:
        return None

    start_line = min(
        transcript_lines,
        key=lambda line: (abs(line["start_seconds"] - start_seconds), line["start_seconds"]),
    )
    if abs(start_line["start_seconds"] - start_seconds) > tolerance_seconds:
        return None

    end_candidates = [
        line
        for line in transcript_lines
        if line["end_seconds"] > start_line["start_seconds"]
    ]
    if not end_candidates:
        return None
    end_line = min(
        end_candidates,
        key=lambda line: (abs(line["end_seconds"] - end_seconds), -line["end_seconds"]),
    )
    if abs(end_line["end_seconds"] - end_seconds) > tolerance_seconds:
        return None
    return start_line, end_line


def _extract_transcript_text_for_segment(
    transcript_lines: List[Dict[str, str]],
    start_time: str,
    end_time: str,
) -> Optional[str]:
    for start_index, line in enumerate(transcript_lines):
        if line["start_time"] != start_time:
            continue

        collected_parts = [line["text"]] if line["text"] else []
        if line["end_time"] == end_time:
            return " ".join(part for part in collected_parts if part).strip()

        for next_line in transcript_lines[start_index + 1 :]:
            if next_line["text"]:
                collected_parts.append(next_line["text"])
            if next_line["end_time"] == end_time:
                return " ".join(part for part in collected_parts if part).strip()

        return None

    return None


class ViralityAnalysis(BaseModel):
    """Detailed virality breakdown for a segment."""

    hook_score: int = Field(
        default=15,
        description="How strong is the opening hook (0-25)",
        ge=0,
        le=25,
    )
    engagement_score: int = Field(
        default=15,
        description="How engaging/entertaining is the content (0-25)",
        ge=0,
        le=25,
    )
    value_score: int = Field(
        default=15,
        description="Educational/informational value (0-25)",
        ge=0,
        le=25,
    )
    shareability_score: int = Field(
        default=15,
        description="Likelihood of being shared (0-25)",
        ge=0,
        le=25,
    )
    total_score: int = Field(
        default=60,
        description="Combined virality score (0-100)",
        ge=0,
        le=100,
    )
    hook_type: Optional[
        Literal["question", "statement", "statistic", "story", "contrast", "none"]
    ] = Field(
        default="none",
        description="Type of hook: question, statement, statistic, story, contrast, or none",
    )
    virality_reasoning: str = Field(
        default="The model did not provide a detailed virality breakdown.",
        description="Explanation of the virality score",
    )


def _default_virality_analysis() -> ViralityAnalysis:
    return ViralityAnalysis()


class TranscriptSegment(BaseModel):
    """Represents a relevant segment of transcript with precise timing and virality analysis."""

    start_time: str = Field(description="Start timestamp in MM:SS format")
    end_time: str = Field(description="End timestamp in MM:SS format")
    text: str = Field(
        validation_alias=AliasChoices("text", "segment"),
        description=(
            "Transcript text taken only from the selected timestamp range. "
            "Keep it verbatim or near-verbatim, and do not paraphrase or merge non-contiguous lines."
        )
    )
    relevance_score: float = Field(
        default=0.75,
        description="Relevance score from 0.0 to 1.0", ge=0.0, le=1.0
    )
    reasoning: str = Field(
        default="Selected by the AI model as a clip candidate.",
        description=(
            "Brief factual explanation of why this exact segment works as a clip. "
            "Base it only on the provided transcript content."
        )
    )
    virality: ViralityAnalysis = Field(
        default_factory=_default_virality_analysis,
        description="Detailed virality score breakdown",
    )
    hook_title: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("hook_title", "title", "headline"),
        description=(
            "Short punchy on-screen title for the clip (3-9 words). Grounded in "
            "the segment content, no hashtags, no emojis, no surrounding quotes."
        ),
    )

    hook_start_time: Optional[str] = Field(
        default=None,
        description=(
            "Optional MM:SS start of the single strongest 2-5 second line inside "
            "the segment, replayed as a cold-open teaser. Null when the segment "
            "already opens on its best line."
        ),
    )
    hook_end_time: Optional[str] = Field(
        default=None,
        description="Optional MM:SS end of that hook line (null with hook_start_time).",
    )

    @field_validator("relevance_score", mode="before")
    @classmethod
    def _coerce_percent_relevance_score(cls, value: Any) -> Any:
        if value is None:
            return value
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return value
        if numeric_value > 1 and numeric_value <= 100:
            return numeric_value / 100
        return value


class BRollOpportunity(BaseModel):
    """Identifies an opportunity to insert B-roll footage."""

    timestamp: str = Field(
        default="00:00",
        validation_alias=AliasChoices("timestamp", "segment_start_time", "start_time"),
        description="When to insert B-roll (MM:SS format)",
    )
    duration: float = Field(
        default=3.0,
        description="How long to show B-roll (2-5 seconds)",
        ge=2.0,
        le=5.0,
    )
    search_term: str = Field(
        default="related visual",
        validation_alias=AliasChoices("search_term", "broll", "visual", "query"),
        description="Keyword to search for B-roll footage",
    )
    context: str = Field(
        default="Suggested B-roll opportunity from the model.",
        validation_alias=AliasChoices("context", "description"),
        description="What's being discussed at this point",
    )

    @field_validator("search_term", "context", mode="before")
    @classmethod
    def _coerce_textish_value(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            return ", ".join(str(item) for item in value if item is not None)
        return str(value)


class TranscriptAnalysis(BaseModel):
    """Analysis result for transcript segments with virality and B-roll opportunities."""

    most_relevant_segments: List[TranscriptSegment]
    summary: str = Field(description="Brief summary of the video content")
    key_topics: List[str] = Field(description="List of main topics discussed")
    broll_opportunities: Optional[List[BRollOpportunity]] = Field(
        default=None, description="Opportunities to insert B-roll footage"
    )
    # Set post-analysis (not from the LLM). Excellent = gate-passed before render slice.
    _excellent_clip_count: int = PrivateAttr(default=0)
    _render_ceiling: int = PrivateAttr(default=AUTO_RENDER_DEFAULT)
    _discovery_ceiling: int = PrivateAttr(default=MAX_FINAL_CLIPS)
    _excellent_segments: List[TranscriptSegment] = PrivateAttr(default_factory=list)
    _drop_log: List[Dict[str, Any]] = PrivateAttr(default_factory=list)


# Enhanced system prompt with virality scoring and B-roll detection
transcript_analysis_system_prompt = """You are an expert transcript analyst for short-form video editing.

Your job is extraction and ranking, not creative rewriting. You must stay fully grounded in the transcript and choose the best clip candidates that already exist in the source material.

OUTPUT CONTRACT:
- Return valid JSON only. Do not output Markdown, headings, bullets, prose, code fences, explanations, or commentary outside the JSON object.
- The top-level JSON object must include: "most_relevant_segments", "summary", and "key_topics".
- Set "broll_opportunities" to null when B-roll was not requested.
- Each item in "most_relevant_segments" must include: "start_time", "end_time", "text", "relevance_score", "reasoning", "virality", and "hook_title".
- Do not use "segment" as an output field. Use "text".
- "virality" must include: "hook_score", "engagement_score", "value_score", "shareability_score", "total_score", "hook_type", and "virality_reasoning".
- Duration bounds come from the task prompt (min/max). Prefer continuous natural duration for each clip — do not aim for fixed 30/60/90-second targets.

CORE OBJECTIVES:
1. Identify segments that would be compelling on social media platforms
2. Focus on complete thoughts, insights, or entertaining moments
3. Prioritize content with hooks, emotional moments, or valuable information
4. Each segment should be engaging and worth watching
5. Score each segment's viral potential with detailed breakdown

GROUNDING RULES:
1. Use only the provided transcript lines and timestamps
2. Never invent facts, tone, context, or transitions that are not present
3. Treat this as span selection over a timestamped transcript, not open-ended summarization
4. Each selected segment must map to one contiguous range in the transcript
5. segment.text must match the chosen span closely and must not include content from outside the chosen range
6. Do not stitch together distant moments into one clip
7. If a speaker label appears, use it only if it is part of the spoken content and helps clarity

CONTENT NEUTRALITY RULES:
1. This is clipping software for legitimate editing workflows
2. Do not judge, moralize, or downgrade a segment just because the topic is controversial, sensitive, adult, political, criminal, medical, or otherwise intense
3. Evaluate segments only on clip quality: clarity, self-contained value, hook strength, emotional impact, specificity, and shareability
4. Do not refuse analysis just because the speaker describes risky, offensive, or uncomfortable subject matter
5. Only downgrade a segment when the transcript itself is weak, confusing, repetitive, unusable, or a poor standalone clip

SEGMENT SELECTION CRITERIA:
1. STRONG HOOKS: Attention-grabbing opening lines
2. VALUABLE CONTENT: Tips, insights, interesting facts, stories
3. EMOTIONAL MOMENTS: Excitement, surprise, humor, inspiration
4. COMPLETE THOUGHTS: Self-contained ideas that make sense alone
5. ENTERTAINING: Content people would want to share
6. HIGH SIGNAL: Prefer specific, concrete language over vague discussion
7. LOW FILLER: Avoid greetings, sponsor reads, repeated setup, throat-clearing, and housekeeping unless they are unusually compelling

WHAT A GOOD CLIP FEELS LIKE:
- A viewer should understand and care without the original title, thumbnail, or previous context
- Prefer a complete mini-story or argument: setup, tension or claim, specific detail, and payoff
- Expand a great short moment to nearby contiguous lines when that adds needed setup, stakes, or payoff
- Strong picks include contrarian claims, mistakes or lessons, concrete examples, before/after moments, frameworks, surprising results, emotionally charged reactions, and complete answers to interesting questions
- Bad picks include intros, sponsor or CTA sections, vague setup, contextless quote fragments, repeated points, definitions without payoff, meandering background, and answer fragments that require unseen context

VIRALITY SCORING (0-100 total, from four 0-25 subscores):
For each segment, provide a detailed virality breakdown:

1. HOOK STRENGTH (0-25):
   - 20-25: Immediately grabs attention (surprising fact, bold claim, intriguing question)
   - 15-19: Good opener that creates curiosity
   - 10-14: Decent start but could be stronger
   - 0-9: Weak or no hook

2. ENGAGEMENT (0-25):
   - 20-25: Highly entertaining, emotional, or dramatic
   - 15-19: Interesting and holds attention
   - 10-14: Moderately engaging
   - 0-9: Flat or boring delivery

3. VALUE (0-25):
   - 20-25: Actionable insights, unique knowledge, or transformative ideas
   - 15-19: Useful information most people don't know
   - 10-14: Somewhat informative
   - 0-9: Common knowledge or filler content

4. SHAREABILITY (0-25):
   - 20-25: "I need to send this to someone" content
   - 15-19: Content worth bookmarking
   - 10-14: Nice but not share-worthy
   - 0-9: Generic content

HOOK TITLES ("hook_title" per segment):
- Write a short on-screen headline (3-9 words) that is burned into the top of the clip
- It must make a scrolling viewer stop: a bold claim, curiosity gap, number, or stakes taken directly from the segment
- Stay grounded: only promise what the clip actually delivers; never invent facts or numbers
- Do not simply repeat the first spoken words verbatim; reframe them as a headline
- Plain text only: no hashtags, no emojis, no quotes around the title
- Write it in the same language as the transcript
- Good examples: "The $40k mistake I keep seeing", "Why nobody tells you this about VC", "Do this before your next interview"

COLD-OPEN HOOK ("hook_start_time"/"hook_end_time" per segment, optional):
- If the segment's most gripping 2-5 second line (punchline, bold claim, shocking number, emotional peak) comes later in the clip, return its MM:SS range; it is replayed as a teaser before the clip starts
- Use timestamps from the transcript lines inside the segment, at least 3 seconds after start_time
- The line must make sense on its own and must not spoil the setup in a confusing way
- Set both to null when the clip already opens on its strongest line or no line qualifies

HOOK TYPES to identify:
- "question": Opens with a question that creates curiosity
- "statement": Bold claim or surprising statement
- "statistic": Uses compelling numbers or data
- "story": Starts with narrative/anecdote
- "contrast": Before/after or problem/solution framing
- "none": No clear hook pattern

B-ROLL OPPORTUNITIES:
Identify 2-4 moments in each segment where B-roll footage could enhance the video:
- When specific objects, places, or concepts are mentioned
- During explanations that could benefit from visual illustration
- At emotional peaks that could use supporting imagery
- Use simple, searchable keywords (e.g., "coffee shop", "laptop coding", "money stack")

TIMING GUIDELINES:
- Obey the min/max durations stated in each task prompt; duration bounds come from the task
- Prefer continuous natural duration per clip — no fixed 30/60/90-second targets
- CRITICAL: start_time MUST be different from end_time
- Focus on natural content boundaries rather than arbitrary time limits
- Include enough context for the segment to be understandable
- Start at the hook or the minimum setup needed to make the hook land, and end after the payoff
- If a highlight is only one good line, expand to include the surrounding setup and payoff rather than returning a tiny fragment
- Stop expanding when the topic drifts, the speaker repeats the same point, or the clip loses momentum
- Never start mid-thought or end abruptly mid-idea when contiguous transcript can complete the arc

TIMESTAMP REQUIREMENTS - EXTREMELY IMPORTANT:
- Use EXACT timestamps as they appear in the transcript
- Never modify timestamp format (keep MM:SS structure)
- start_time MUST be LESS THAN end_time (start_time < end_time)
- Respect the MINIMUM and MAXIMUM segment durations from the task prompt
- Look at transcript ranges like [02:25 - 02:35] and use different start/end times
- NEVER use the same timestamp for both start_time and end_time
- Example: start_time: "02:25", end_time: "02:35" (NOT "02:25" and "02:25")

SCORING AND OUTPUT RULES:
- relevance_score should reflect how well the segment works as a standalone short clip, not just whether the topic is generally important
- Penalize clips that are only quotable but not self-contained, too generic, missing setup, missing payoff, or padded with filler
- virality_reasoning and reasoning should cite what is actually present in the chosen span
- summary and key_topics must also stay grounded in the transcript and should not add outside interpretation
- When a user brief is provided in the task prompt, treat it as a HARD constraint: NEVER select spans, titles, or hooks that violate forbidden topics/phrases in the brief; skip to the next allowed moment while staying grounded

SHORT-FORM RETENTION (WATCH + hook-writer) — apply to every segment, with or without a user brief:
- W — Win the first ~3 seconds: the start of the span must open a real gap (curiosity, stakes, contrarian with receipts, emotional cold-open, or a specific concrete claim). Prefer spans whose opening lands the hook immediately — no warm-up intros. Reject clips whose first spoken seconds are throat-clearing, greetings, or soft setup.
- Hook honesty: extract the hook from what the transcript actually delivers; reject bait that the clip cannot pay off (over-promise → mid-clip retention cliff). hook_score and virality_reasoning must cite the opening line that stops the scroll.
- Specificity beats abstraction: a real number, named entity, or sharp claim scores higher than vague advice.
- A — Arc with open loops: setup → tension → payoff inside the same contiguous span; resolve the loop before the cut; never end mid-thought when contiguous lines can complete it.
- C — Cash the promise: the ending must close what the opening opened.
- Retention pacing: prefer moments with idea/emotion shifts across the clip, not a single quotable line with dead air around it.
- Watch-through over vanity: rank for un-skippable standalone retention (social + paid), not topic importance alone. Hard gate: if the opening cannot stop a scroll in ~3s, do not select the span.

Return only excellent, publishable standalone clips — never pad toward a count. If fewer excellent moments exist than the harvest range, return fewer. Every selected segment must be accurate, self-contained, have proper time ranges, and score high on virality metrics."""

# Lazy-loaded agent to avoid import-time failures when API keys aren't set
_transcript_agent: Optional[Agent[None, TranscriptAnalysis]] = None
_transcript_agent_signature: Optional[tuple[str | None, ...]] = None

SUPPORTED_LLM_PROVIDERS = {"google", "google-gla", "openai", "openrouter", "anthropic", "ollama"}


def _split_llm_name(model_name: str) -> tuple[str, str | None]:
    if ":" not in model_name:
        return model_name.strip().lower(), None

    provider, provider_model_name = model_name.split(":", 1)
    return provider.strip().lower(), provider_model_name.strip() or None


def _get_missing_llm_key_error(model_name: str, runtime_config: Config) -> Optional[str]:
    """Return a clear configuration error when the selected LLM key is missing."""
    provider, provider_model_name = _split_llm_name(model_name)

    if provider not in SUPPORTED_LLM_PROVIDERS:
        return (
            f"Unsupported LLM provider '{provider}'. "
            "Use google-gla:*, openai:*, openrouter:*, anthropic:*, or ollama:*."
        )

    if not provider_model_name:
        return (
            "Selected LLM is missing a model name. "
            "Use the format provider:model, for example ollama:gpt-oss:20b."
        )

    if provider in {"google", "google-gla"} and not runtime_config.google_api_key:
        return (
            "Selected LLM provider is Google, but GOOGLE_API_KEY is not set. "
            "Set GOOGLE_API_KEY or set LLM to openai:* / anthropic:* / ollama:* with the matching API key."
        )

    if provider == "openrouter" and not runtime_config.openrouter_api_key:
        return "Selected LLM provider is OpenRouter, but OPENROUTER_API_KEY is not set."

    if provider == "openrouter" and "/" not in provider_model_name:
        return "OpenRouter model must include its vendor, for example openrouter:anthropic/claude-sonnet-5.5."

    if provider == "openai" and not runtime_config.openai_api_key:
        return (
            "Selected LLM provider is OpenAI, but OPENAI_API_KEY is not set. "
            "Set OPENAI_API_KEY or choose another provider with a matching API key."
        )

    if provider == "anthropic" and not runtime_config.anthropic_api_key:
        return (
            "Selected LLM provider is Anthropic, but ANTHROPIC_API_KEY is not set. "
            "Set ANTHROPIC_API_KEY or choose another provider with a matching API key."
        )

    if provider == "ollama":
        # Ollama can run locally without an API key. OLLAMA_BASE_URL/OLLAMA_API_KEY
        # are optional and passed through as environment variables.
        return None

    return None


def _openrouter_keys(runtime_config: Config) -> list[str]:
    """Active OpenRouter key first, then any extras from OPENROUTER_API_KEYS."""
    keys = list(getattr(runtime_config, "openrouter_api_keys", None) or [])
    primary = runtime_config.openrouter_api_key
    if primary and primary not in keys:
        keys.insert(0, primary)
    return keys


def _log_prompt_cache_usage(result: Any, *, label: str) -> None:
    """Surface OpenRouter/Anthropic cache hits so we can track cost savings."""
    usage = getattr(result, "usage", None)
    if callable(usage):
        try:
            usage = usage()
        except Exception:
            usage = None
    if usage is None:
        return
    cache_read = int(getattr(usage, "cache_read_tokens", 0) or 0)
    cache_write = int(getattr(usage, "cache_write_tokens", 0) or 0)
    if cache_read or cache_write:
        logger.info(
            "Prompt cache %s: read=%s write=%s input=%s output=%s",
            label,
            cache_read,
            cache_write,
            getattr(usage, "input_tokens", None),
            getattr(usage, "output_tokens", None),
        )


class _OpenRouterCachedModel(OpenRouterModel):
    """OpenRouter chat model that marks Anthropic system prompts for caching."""

    async def _map_messages(self, messages, model_request_parameters):  # type: ignore[override]
        openai_messages = await super()._map_messages(messages, model_request_parameters)
        if not str(self.model_name).startswith("anthropic/"):
            return openai_messages
        mapped = []
        for message in openai_messages:
            role = message.get("role") if isinstance(message, dict) else None
            content = message.get("content") if isinstance(message, dict) else None
            if role == "system" and isinstance(content, str) and content.strip():
                mapped.append(
                    {
                        **message,
                        "content": [
                            {
                                "type": "text",
                                "text": content,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    }
                )
            else:
                mapped.append(message)
        return mapped


def _build_transcript_model(
    runtime_config: Config, model_name: str | None = None
) -> Model | str:
    model_name = model_name or runtime_config.llm
    provider, provider_model_name = _split_llm_name(model_name)
    if provider == "openrouter":
        fallback = runtime_config.openrouter_fallback_model
        if not provider_model_name or "/" not in provider_model_name or "/" not in fallback:
            raise RuntimeError("OpenRouter primary and fallback models must use vendor/model IDs.")
        # The installed SDK predates Sonnet 5.5. Both selected models advertise
        # native JSON schema output; require supporting endpoints when routing.
        profile = OpenRouterProvider.model_profile(provider_model_name).update(
            OpenAIModelProfile(supports_json_schema_output=True)
        )
        settings = {
            "openrouter_models": [fallback] if fallback != provider_model_name else [],
            "openrouter_provider": {"allow_fallbacks": True, "require_parameters": True, "data_collection": "deny"},
            "openrouter_reasoning": {"effort": "low"},
            "max_tokens": 8192,
            # Stable key so multi-window analysis reuses the same system prefix.
            "openai_prompt_cache_key": f"supoclip:{TRANSCRIPT_ANALYSIS_CACHE_VERSION}",
        }
        # One model per key, in priority order. FallbackModel moves to the next
        # key on an API error (401/402/429 etc.), so a drained key never stops a job.
        models = [
            _OpenRouterCachedModel(
                provider_model_name,
                provider=OpenRouterProvider(api_key=key),
                profile=profile,
                settings=settings,
            )
            for key in _openrouter_keys(runtime_config)
        ]
        return models[0] if len(models) == 1 else FallbackModel(*models)
    if provider != "ollama":
        return model_name

    if not provider_model_name:
        raise RuntimeError(
            "Selected LLM provider is Ollama, but no model name was provided. "
            "Use the format ollama:<model>, for example ollama:gpt-oss:20b."
        )

    return OllamaModel(
        provider_model_name,
        provider=OllamaProvider(
            base_url=runtime_config.resolve_ollama_base_url(),
            api_key=runtime_config.ollama_api_key,
        ),
    )


def get_transcript_agent() -> Agent[None, TranscriptAnalysis]:
    """Get or create the transcript analysis agent (lazy initialization)."""
    global _transcript_agent, _transcript_agent_signature
    runtime_config = get_config()
    provider, _ = _split_llm_name(runtime_config.llm)
    signature = (
        runtime_config.llm,
        runtime_config.openai_api_key,
        *_openrouter_keys(runtime_config),
        runtime_config.openrouter_fallback_model,
        runtime_config.google_api_key,
        runtime_config.anthropic_api_key,
        runtime_config.ollama_base_url,
        runtime_config.ollama_api_key,
    )
    if _transcript_agent is None or _transcript_agent_signature != signature:
        apply_settings_to_process_env(runtime_config.as_runtime_settings())
        config_error = _get_missing_llm_key_error(runtime_config.llm, runtime_config)
        if config_error:
            raise RuntimeError(config_error)

        _transcript_agent = Agent[None, TranscriptAnalysis](
            model=_build_transcript_model(runtime_config),
            output_type=NativeOutput(TranscriptAnalysis, strict=True) if provider == "openrouter" else TranscriptAnalysis,
            system_prompt=transcript_analysis_system_prompt,
            # Some local Ollama/OpenAI-compatible endpoints can return formatted
            # prose before settling on schema-valid JSON. Keep retries limited
            # while still allowing enough repair attempts for local models.
            output_retries=2 if provider == "ollama" else 2,
        )
        _transcript_agent_signature = signature
    return _transcript_agent


class RankedCandidate(BaseModel):
    """One candidate's place in the comparative rerank."""

    candidate_id: int = Field(description="The id of the candidate being ranked")
    score: int = Field(
        description="Comparative virality score (0-100) relative to the other candidates",
        ge=0,
        le=100,
    )
    reason: str = Field(
        default="",
        description="One short sentence on why it ranks here versus the others",
    )


class CandidateRanking(BaseModel):
    """Best-first ranking of clip candidates from one source video."""

    ranking: List[RankedCandidate] = Field(
        description="Every candidate, ordered from strongest to weakest clip"
    )


candidate_rerank_system_prompt = """You are a senior short-form video editor choosing which clips from one long video to publish.

You receive candidate clips that were picked from different parts of the same video. Each one was scored in isolation, so those scores are not comparable. Your job is to compare the candidates against each other and produce one calibrated, best-first ranking.

Judge each candidate only on its own transcript text:
1. Hook: do the first one or two sentences make a scrolling viewer stop?
2. Standalone: would a viewer understand and care without the rest of the video?
3. Payoff: does it land a clear point, punchline, lesson, or story beat before it ends?
4. Specificity and energy: concrete details, numbers, strong opinions, emotion, or humor beat vague talk.
5. Shareability: would someone send this to a friend or save it?

Rules:
- Stay content-neutral: never downgrade a candidate just because its topic is sensitive, controversial, or adult.
- Rank every candidate exactly once, using the given candidate_id values.
- Scores are comparative: spread them out so better clips clearly score higher. The best candidate should usually land between 80 and 95; weak, contextless, or filler candidates belong below 50.
- Return valid JSON only, with a "ranking" list ordered from best to worst."""

_rerank_agent: Optional[Agent[None, CandidateRanking]] = None
_rerank_agent_signature: Optional[tuple[str | None, ...]] = None

_hook_title_agent: Optional[Agent[None, "HookTitleBatch"]] = None
_hook_title_agent_signature: Optional[tuple[str | None, ...]] = None


class HookTitleItem(BaseModel):
    """One rewritten on-screen headline for a final clip."""

    segment_index: int = Field(description="0-based index into the segments list")
    hook_title: str = Field(description="Punchy 3-9 word on-screen headline")


class HookTitleBatch(BaseModel):
    """Dedicated hook_title rewrite pass over final segments."""

    titles: List[HookTitleItem] = Field(
        description="One rewritten hook_title per segment_index"
    )


hook_title_rewrite_system_prompt = """You write short on-screen headlines for short-form video clips.

Input: a list of final clip segments with transcript text and an optional draft title.
Output: JSON only — {"titles":[{"segment_index":0,"hook_title":"..."}, ...]}.

Formulas (pick the best fit per clip; stay in the transcript's language):
1. Curiosity gap — incomplete but honest promise the clip pays off
2. Stakes — risk, cost, consequence, or urgency grounded in the text
3. Number / statistic — a concrete figure from the segment
4. Contrast — before/after, myth vs reality, old vs new
5. Direct question — only if the answer is inside the clip

Hard rules:
- 3-9 words, plain text, no hashtags, emojis, or surrounding quotes
- Grounded only in the segment text — never invent facts or numbers
- Anti-meta: NEVER write titles like "A pergunta sobre…", "O que você precisa saber sobre…",
  "Por que isso importa", "The truth about…", "Everything you need to know…",
  "Let's talk about…", "Uma reflexão sobre…", or topic labels that describe the category
- Prefer a claim, number, stakes, or curiosity the scroll-stopper can feel in one glance
- Sound human: short words, punchy rhythm; no corporate or academic phrasing
- One title per segment_index; cover every index provided"""


def get_hook_title_agent() -> Agent[None, HookTitleBatch]:
    """Get or create the dedicated hook_title rewrite agent."""
    global _hook_title_agent, _hook_title_agent_signature
    runtime_config = get_config()
    provider, _ = _split_llm_name(runtime_config.llm)
    model_name = runtime_config.llm
    signature = (
        model_name,
        runtime_config.openai_api_key,
        *_openrouter_keys(runtime_config),
        runtime_config.openrouter_fallback_model,
        runtime_config.google_api_key,
        runtime_config.anthropic_api_key,
        runtime_config.ollama_base_url,
        runtime_config.ollama_api_key,
    )
    if _hook_title_agent is None or _hook_title_agent_signature != signature:
        apply_settings_to_process_env(runtime_config.as_runtime_settings())
        config_error = _get_missing_llm_key_error(model_name, runtime_config)
        if config_error:
            raise RuntimeError(config_error)

        _hook_title_agent = Agent[None, HookTitleBatch](
            model=_build_transcript_model(runtime_config, model_name),
            output_type=(
                NativeOutput(HookTitleBatch, strict=True)
                if provider == "openrouter"
                else HookTitleBatch
            ),
            system_prompt=hook_title_rewrite_system_prompt,
            output_retries=2,
        )
        _hook_title_agent_signature = signature
    return _hook_title_agent


def get_rerank_agent() -> Agent[None, CandidateRanking]:
    """Get or create the candidate rerank agent (lazy initialization)."""
    global _rerank_agent, _rerank_agent_signature
    runtime_config = get_config()
    model_name = getattr(runtime_config, "rerank_llm", None) or runtime_config.llm
    provider, _ = _split_llm_name(model_name)
    signature = (
        model_name,
        runtime_config.openai_api_key,
        *_openrouter_keys(runtime_config),
        runtime_config.openrouter_fallback_model,
        runtime_config.google_api_key,
        runtime_config.anthropic_api_key,
        runtime_config.ollama_base_url,
        runtime_config.ollama_api_key,
    )
    if _rerank_agent is None or _rerank_agent_signature != signature:
        apply_settings_to_process_env(runtime_config.as_runtime_settings())
        config_error = _get_missing_llm_key_error(model_name, runtime_config)
        if config_error:
            raise RuntimeError(config_error)

        _rerank_agent = Agent[None, CandidateRanking](
            model=_build_transcript_model(runtime_config, model_name),
            output_type=NativeOutput(CandidateRanking, strict=True) if provider == "openrouter" else CandidateRanking,
            system_prompt=candidate_rerank_system_prompt,
            output_retries=2,
        )
        _rerank_agent_signature = signature
    return _rerank_agent


def build_candidate_rerank_prompt(candidates: List[TranscriptSegment]) -> str:
    """Describe every candidate in full so the reranker can compare them."""
    blocks = []
    for candidate_id, segment in enumerate(candidates):
        duration = _parse_transcript_timestamp_seconds(
            segment.end_time
        ) - _parse_transcript_timestamp_seconds(segment.start_time)
        blocks.append(
            f"candidate_id: {candidate_id}\n"
            f"time: {segment.start_time} - {segment.end_time} ({duration}s)\n"
            f"hook_title: {segment.hook_title or '(none)'}\n"
            f"text: {segment.text}"
        )
    return (
        f"Rank these {len(candidates)} clip candidates from the same video, best first.\n\n"
        + "\n\n".join(blocks)
    )


def build_transcript_analysis_prompt(
    transcript: str,
    include_broll: bool = False,
    clip_signals: str | None = None,
    window: tuple[int, int] | None = None,
    prefs: ClipSelectionPrefs | None = None,
    clip_budget: int | None = None,
) -> str:
    """Build the grounded task prompt for transcript analysis.

    ``window`` marks the prompt as one (start, end) slice of a longer source.
    """
    selection = prefs or ClipSelectionPrefs()
    budget = max(1, int(clip_budget or 5))
    segment_target = window_segment_target(budget, windowed=window is not None)
    min_s = selection.min_seconds
    max_s = selection.max_seconds
    ideal_min = selection.ideal_min
    ideal_max = selection.ideal_max
    window_note = ""
    if window:
        window_note = (
            f"\nThis transcript is one window ({_format_minutes_label(window[0])} - "
            f"{_format_minutes_label(window[1])}) of a longer video. Other windows are "
            "analysed separately, so pick the best clips that exist inside this window only.\n"
        )
    broll_instruction = ""
    if include_broll:
        broll_instruction = (
            "\n5. Also identify B-roll opportunities for each chosen segment where stock footage could enhance the visual appeal."
        )
    signal_section = ""
    if clip_signals:
        signal_section = (
            "\n\nAdditional deterministic signals from transcript/audio analysis:\n"
            f"{clip_signals}\n\n"
            "Use these as hints only. They should influence ranking, but every final segment "
            "must still be a coherent contiguous transcript range."
        )
    brief_section = ""
    if selection.brief:
        truncated_note = (
            " (brief truncated)"
            if len(selection.brief) >= CLIP_BRIEF_MAX_CHARS
            else ""
        )
        brief_section = (
            "\n\nHARD CONSTRAINTS — User brief (NEVER violate; do not invent)"
            f"{truncated_note}:\n"
            f"{selection.brief}\n\n"
            "NEVER select any segment whose spoken text, hook_title, or hook violates this brief "
            "(forbidden topics, skip lists, or phrases marked to avoid). If the strongest viral "
            "moment violates the brief, skip it and choose the next allowed contiguous span. "
            "Stay fully grounded in the transcript. Still reinforce WATCH retention on allowed "
            "spans: win the first ~3s with an honest gap extracted from the content, cash the "
            "promise by the end of the span, keep a complete arc (no mid-thought cut), and prefer "
            "specific claims over vague advice — for scroll-stopping social/paid use."
        )

    if selection.duration_preset == "auto":
        duration_guidance = f"""- Accepted duration range for this run: {min_s}-{max_s} seconds.
- Continuous natural duration per clip: do not aim for fixed 30/60/90-second lengths.
- Expand contiguous same-topic lines until the narrative arc or payoff closes.
- Different clips may have different lengths within the accepted range.
- Stop when the promised hook pays off — do not pad after the payoff.
- The start must be a hook or pattern interrupt in the first ~3 seconds (WATCH).
- hook_score and virality_reasoning must cite the opening line that wins the scroll-stop."""
    else:
        duration_guidance = f"""- Most selected clips should be {ideal_min}-{ideal_max} seconds.
- Accepted duration range for this run: {min_s}-{max_s} seconds.
- Only choose a clip near the minimum when it already contains a full setup and payoff.
- If a strong moment is shorter than {ideal_min} seconds, first try expanding to nearby contiguous transcript lines that add useful context."""

    return f"""Analyze this video transcript and identify the most engaging segments for short-form content.
{window_note}
The transcript is formatted as one line per timestamped span, for example:
[00:12 - 00:21] Spoken text here
[00:21 - 00:35] More spoken text here

Follow this workflow:
1. Read the transcript as a sequence of timestamped spans.
2. Select only contiguous ranges that already exist in the transcript.
3. Prefer moments with a strong hook, clear payoff, emotional charge, or concrete value.
4. For each chosen segment, use the earliest timestamp in the selected range as start_time and the latest timestamp in the selected range as end_time.{broll_instruction}

Selection target (quality-only):
- Harvest up to {segment_target} excellent segments in this pass (discovery ceiling for the whole video ≈ {budget}).
- Return ONLY publishable clips: strong ~3s opener, standalone, payoff, specific, shareable, and inside the brief.
- Do NOT pad to hit a count. If only 2 excellent moments exist, return 2. Prefer empty over filler.
{duration_guidance}
- Skip weak standalone picks: intros, sponsor reads, CTAs, contextless quotes, repeated points, vague setup, greetings/warm-ups, and answer fragments that require prior context.
- Before returning a segment, ask whether a viewer would understand and care without seeing the rest of the source video.
- Do not start without a hook or end abruptly mid-idea when contiguous lines can complete the arc.
{brief_section}

Critical accuracy requirements:
- Do not fabricate or embellish content.
- Do not use timestamps that are not present in the transcript.
- Do not merge separate non-contiguous moments into one segment.
- segment.text must reflect only the spoken content inside the selected time range.
- If a span lacks enough context to stand alone, expand to nearby contiguous lines rather than guessing.
- If there is a tradeoff between "viral" and "accurate", choose accuracy.
- Do not reject or penalize a segment simply because of the subject matter; stay content-neutral and assess clip quality only.
{signal_section}

JSON-only output requirements:
- Return one valid JSON object and nothing else.
- No Markdown, headings, bullets, code fences, or explanatory text outside JSON.
- Top-level keys: "most_relevant_segments", "summary", "key_topics", "broll_opportunities".{' Set "broll_opportunities" to null.' if not include_broll else ''}
- Segment keys: "start_time", "end_time", "text", "relevance_score", "reasoning", "virality", "hook_title".
- Optional segment keys "hook_start_time"/"hook_end_time": MM:SS range of the strongest 2-5 second line later in the segment (cold-open teaser), or null.
- "hook_title" is a 3-9 word plain-text headline for the clip, in the transcript's language, grounded in the segment (no hashtags, emojis, or quotes).
- Virality keys: "hook_score", "engagement_score", "value_score", "shareability_score", "total_score", "hook_type", "virality_reasoning".
- Do not return segments shorter than {min_s} seconds or longer than {max_s} seconds.

Transcript:
{transcript}"""


_META_HOOK_TITLE_RE = re.compile(
    r"^\s*(?:"
    r"(?:a|uma)\s+pergunta\s+sobre\b|"
    r"o\s+que\s+voc[eê]\s+precisa\s+saber\s+sobre\b|"
    r"por\s+que\s+isso\s+importa\b|"
    r"uma\s+reflex[aã]o\s+sobre\b|"
    r"vamos\s+falar\s+sobre\b|"
    r"tudo\s+sobre\b|"
    r"the\s+truth\s+about\b|"
    r"everything\s+you\s+need\s+to\s+know\b|"
    r"let'?s\s+talk\s+about\b|"
    r"why\s+this\s+matters\b|"
    r"a\s+(?:quick\s+)?(?:note|thought|look)\s+on\b|"
    r"understanding\b|"
    r"exploring\b|"
    r"an\s+introduction\s+to\b"
    r")",
    re.IGNORECASE,
)


def is_meta_hook_title(title: Optional[str]) -> bool:
    """True when the title names the topic instead of making a scroll-stopping claim."""
    if not title:
        return True
    return bool(_META_HOOK_TITLE_RE.search(title.strip()))


def sanitize_hook_title(raw: Optional[str]) -> Optional[str]:
    """Normalize an AI-provided hook title for on-screen rendering.

    Strips wrapping quotes/markdown, collapses whitespace, drops hashtags, and
    trims to a word-boundary length cap. Returns None when nothing usable is
    left (including meta/topic-label titles) so callers can skip or rewrite.
    """
    if not raw:
        return None
    title = str(raw).strip()
    title = title.strip("\"'`“”‘’").strip()
    title = re.sub(r"#\w+", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    # Drop trailing sentence punctuation but keep ?/! (they carry the hook).
    title = title.rstrip(".,;:-–— ").strip()
    if not title:
        return None
    if is_meta_hook_title(title):
        return None

    words = title.split()
    if len(words) > HOOK_TITLE_MAX_WORDS:
        words = words[:HOOK_TITLE_MAX_WORDS]
        title = " ".join(words)
    if len(title) > HOOK_TITLE_MAX_CHARS:
        clipped = title[: HOOK_TITLE_MAX_CHARS + 1]
        cut = clipped.rfind(" ")
        title = (clipped[:cut] if cut > 20 else title[:HOOK_TITLE_MAX_CHARS]).rstrip(
            ".,;:-–— "
        )
    if not title or is_meta_hook_title(title):
        return None
    return title


def build_hook_title_rewrite_prompt(segments: List[TranscriptSegment]) -> str:
    blocks = []
    for index, segment in enumerate(segments):
        draft = segment.hook_title or "(none)"
        text = (segment.text or "").strip()
        if len(text) > 600:
            text = text[:600].rstrip() + "…"
        blocks.append(
            f"segment_index: {index}\n"
            f"time: {segment.start_time} - {segment.end_time}\n"
            f"draft_title: {draft}\n"
            f"text: {text}"
        )
    return (
        f"Rewrite hook_title for these {len(segments)} final clips.\n\n"
        + "\n\n".join(blocks)
    )


async def rewrite_hook_titles(
    segments: List[TranscriptSegment],
) -> List[TranscriptSegment]:
    """Second LLM pass: claim/number/question headlines; drop meta drafts."""
    if not segments:
        return segments
    try:
        agent = get_hook_title_agent()
        result = await asyncio.wait_for(
            agent.run(build_hook_title_rewrite_prompt(segments)),
            timeout=HOOK_TITLE_REWRITE_TIMEOUT_SECONDS,
        )
        batch = result.output
    except Exception as exc:
        logger.warning("Hook title rewrite failed; keeping drafts: %s", exc)
        for segment in segments:
            segment.hook_title = sanitize_hook_title(segment.hook_title)
        return segments

    by_index = {
        item.segment_index: sanitize_hook_title(item.hook_title)
        for item in (batch.titles or [])
        if 0 <= item.segment_index < len(segments)
    }
    for index, segment in enumerate(segments):
        rewritten = by_index.get(index)
        if rewritten:
            segment.hook_title = rewritten
        else:
            segment.hook_title = sanitize_hook_title(segment.hook_title)
        logger.debug(
            "Hook title [%s] %s-%s -> %r",
            index,
            segment.start_time,
            segment.end_time,
            segment.hook_title,
        )
    return segments


def _sanitize_hook_span(
    segment: TranscriptSegment, start_seconds: int, end_seconds: int
) -> None:
    """Drop a cold-open hook range that is malformed or outside the segment.

    Requires the teaser to start ≥2s after the body open (already the opener
    otherwise) and last 1–8s. Word-level snap happens at render time.
    """
    try:
        hook_start = _parse_transcript_timestamp_seconds(segment.hook_start_time or "")
        hook_end = _parse_transcript_timestamp_seconds(segment.hook_end_time or "")
    except ValueError:
        hook_start = hook_end = None
    duration = (hook_end - hook_start) if hook_start is not None and hook_end is not None else None
    if (
        hook_start is None
        or hook_end is None
        or duration is None
        or hook_start < start_seconds + 2
        or hook_end > end_seconds
        or hook_start >= hook_end
        or duration < 1
        or duration > 8
    ):
        segment.hook_start_time = None
        segment.hook_end_time = None


def _parse_transcript_timestamp_seconds(timestamp: str) -> int:
    """Parse MM:SS or HH:MM:SS transcript timestamps into seconds."""
    parts = [int(part) for part in timestamp.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"Unsupported timestamp format: {timestamp}")


def _format_transcript_timestamp(seconds: int) -> str:
    """Format seconds as a transcript timestamp."""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _format_minutes_label(seconds: int) -> str:
    """Format seconds the way transcript lines do: MM:SS with unbounded minutes."""
    minutes, seconds = divmod(max(0, int(seconds)), 60)
    return f"{minutes:02d}:{seconds:02d}"


def _parse_transcript_spans(transcript: str) -> list[dict[str, Any]]:
    """Parse timestamped transcript lines into spans."""
    spans = []
    for line in transcript.splitlines():
        match = TRANSCRIPT_SPAN_RE.match(line.strip())
        if not match:
            continue
        try:
            start_seconds = _parse_transcript_timestamp_seconds(match.group("start"))
            end_seconds = _parse_transcript_timestamp_seconds(match.group("end"))
        except ValueError:
            continue
        if end_seconds <= start_seconds:
            continue
        spans.append(
            {
                "start": start_seconds,
                "end": end_seconds,
                "text": match.group("text").strip(),
            }
        )
    return spans


def _extract_transcript_text(
    transcript_spans: list[dict[str, Any]], start_seconds: int, end_seconds: int
) -> str:
    """Return transcript text overlapping a selected time range."""
    selected_text = [
        span["text"]
        for span in transcript_spans
        if span["text"]
        and span["end"] > start_seconds
        and span["start"] < end_seconds
    ]
    return " ".join(selected_text).strip()


def _choose_repaired_bounds(
    transcript_spans: list[dict[str, Any]],
    start_seconds: int,
    end_seconds: int,
    prefs: ClipSelectionPrefs | None = None,
) -> tuple[int, int] | None:
    """Repair model-selected bounds to the nearest acceptable contiguous range."""
    if not transcript_spans:
        return None

    selection = prefs or ClipSelectionPrefs()
    min_s = selection.min_seconds
    max_s = selection.max_seconds
    ideal_min = selection.ideal_min
    ideal_max = selection.ideal_max

    starts = sorted({span["start"] for span in transcript_spans})
    ends = sorted({span["end"] for span in transcript_spans})
    current_duration = end_seconds - start_seconds

    if current_duration > max_s:
        # Trim to start+max_s (hard cap), not ideal_max — ideal is soft metadata only.
        target_end = start_seconds + max_s
        candidate_ends = [
            candidate
            for candidate in ends
            if start_seconds + min_s <= candidate <= min(target_end, end_seconds)
        ]
        if candidate_ends:
            return start_seconds, max(candidate_ends)
        if start_seconds + min_s <= target_end:
            return start_seconds, target_end
        return None

    if current_duration < min_s:
        candidate_ranges: list[tuple[int, int, int]] = []
        for candidate_start in starts:
            if candidate_start > start_seconds:
                continue
            for candidate_end in ends:
                if candidate_end < end_seconds:
                    continue
                duration = candidate_end - candidate_start
                if min_s <= duration <= max_s:
                    extra_context = (start_seconds - candidate_start) + (
                        candidate_end - end_seconds
                    )
                    # Auto: do not pull toward mid-band via ideal_penalty.
                    ideal_penalty = 0
                    if selection.duration_preset != "auto":
                        if duration < ideal_min:
                            ideal_penalty = ideal_min - duration
                        elif duration > ideal_max:
                            ideal_penalty = duration - ideal_max
                    candidate_ranges.append(
                        (ideal_penalty * 1000 + extra_context, candidate_start, candidate_end)
                    )
        if candidate_ranges:
            _, repaired_start, repaired_end = min(candidate_ranges)
            return repaired_start, repaired_end

    return None


def _repair_segment_bounds(
    segment: TranscriptSegment,
    transcript_spans: list[dict[str, Any]],
    start_seconds: int,
    end_seconds: int,
    prefs: ClipSelectionPrefs | None = None,
) -> tuple[int, int] | None:
    """Adjust near-miss model ranges to usable transcript-aligned bounds."""
    repaired_bounds = _choose_repaired_bounds(
        transcript_spans,
        start_seconds,
        end_seconds,
        prefs=prefs,
    )
    if not repaired_bounds:
        return None

    repaired_start, repaired_end = repaired_bounds
    segment.start_time = _format_transcript_timestamp(repaired_start)
    segment.end_time = _format_transcript_timestamp(repaired_end)
    repaired_text = _extract_transcript_text(
        transcript_spans,
        repaired_start,
        repaired_end,
    )
    if repaired_text:
        segment.text = repaired_text
    logger.info(
        "Repaired segment duration: %s-%s -> %s-%s",
        _format_transcript_timestamp(start_seconds),
        _format_transcript_timestamp(end_seconds),
        segment.start_time,
        segment.end_time,
    )
    return repaired_start, repaired_end


def _is_transient_model_error(error: Exception) -> bool:
    """Recognize provider overload and transport failures, including SDK wrappers."""
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ModelHTTPError):
            return current.status_code in TRANSIENT_MODEL_STATUS_CODES
        if isinstance(
            current,
            (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError, TimeoutError),
        ):
            return True
        current = current.__cause__
    return False


async def _run_transcript_analysis(
    agent: Agent,
    prompt: str,
    timeout_seconds: Optional[float] = None,
) -> Any:
    """Retry temporary failures on the configured model within one overall budget.

    Agent output retries handle schema repair, not HTTP failures. Retrying only
    the model step preserves completed downloads and transcription. SDK retries
    also count against the wall-clock budget, and cancellation passes through.
    """
    if timeout_seconds is None:
        timeout_seconds = TRANSCRIPT_ANALYSIS_TIMEOUT_SECONDS
    try:
        async with asyncio.timeout(timeout_seconds):
            for attempt in range(1, TRANSCRIPT_ANALYSIS_MAX_ATTEMPTS + 1):
                try:
                    result = await agent.run(prompt)
                    if hasattr(result, "response"):
                        logger.info("AI analysis completed using model %s", result.response.model_name)
                    _log_prompt_cache_usage(result, label="transcript-analysis")
                    return result
                except Exception as error:
                    if not _is_transient_model_error(error):
                        raise
                    if attempt == TRANSCRIPT_ANALYSIS_MAX_ATTEMPTS:
                        logger.error(
                            "AI analysis unavailable after %s attempts (%s, HTTP status %s)",
                            attempt,
                            type(error).__name__,
                            getattr(error, "status_code", None),
                        )
                        raise RuntimeError(
                            "The AI analysis service is temporarily unavailable after "
                            "several attempts. Please try again shortly."
                        ) from None

                    delay = 5 * 2 ** (attempt - 1) + random.uniform(0, 1)
                    # Do not log provider bodies: they may contain transcript data.
                    logger.warning(
                        "Temporary AI analysis failure (%s, HTTP status %s); "
                        "retrying attempt %s/%s in %.1fs",
                        type(error).__name__,
                        getattr(error, "status_code", None),
                        attempt + 1,
                        TRANSCRIPT_ANALYSIS_MAX_ATTEMPTS,
                        delay,
                    )
                    await asyncio.sleep(delay)
    except TimeoutError:
        logger.error("AI analysis exceeded the %ss time budget", timeout_seconds)
        raise RuntimeError(
            "The AI analysis service took too long to respond. Please try again shortly."
        ) from None


def split_transcript_into_windows(
    transcript: str,
    window_seconds: int = ANALYSIS_WINDOW_SECONDS,
    overlap_seconds: int = ANALYSIS_WINDOW_OVERLAP_SECONDS,
    single_window_max_seconds: int = ANALYSIS_SINGLE_WINDOW_MAX_SECONDS,
) -> List[Dict[str, Any]]:
    """Split a formatted transcript into overlapping time windows.

    Transcripts up to ``single_window_max_seconds`` stay one window holding the
    original text unchanged. A short trailing window is folded into the one
    before it so no window is analysed with too little material.
    """
    timed_lines: List[tuple[int, str]] = []
    total_seconds = 0
    for raw_line in transcript.splitlines():
        line = raw_line.strip()
        match = TRANSCRIPT_SPAN_RE.match(line)
        if not match:
            continue
        try:
            start_seconds = _parse_transcript_timestamp_seconds(match.group("start"))
            end_seconds = _parse_transcript_timestamp_seconds(match.group("end"))
        except ValueError:
            continue
        timed_lines.append((start_seconds, line))
        total_seconds = max(total_seconds, end_seconds, start_seconds)

    if not timed_lines or total_seconds <= single_window_max_seconds:
        return [{"start": 0, "end": total_seconds, "transcript": transcript}]

    step = max(1, window_seconds - overlap_seconds)
    bounds: List[List[int]] = []
    window_start = 0
    while True:
        window_end = window_start + window_seconds
        if window_end >= total_seconds:
            bounds.append([window_start, total_seconds])
            break
        bounds.append([window_start, window_end])
        window_start += step
    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < window_seconds // 2:
        bounds.pop()
        bounds[-1][1] = total_seconds

    windows: List[Dict[str, Any]] = []
    for index, (window_start, window_end) in enumerate(bounds):
        is_last = index == len(bounds) - 1
        lines = [
            line
            for start_seconds, line in timed_lines
            if window_start <= start_seconds
            and (start_seconds < window_end or (is_last and start_seconds <= window_end))
        ]
        if lines:
            windows.append(
                {"start": window_start, "end": window_end, "transcript": "\n".join(lines)}
            )
    return windows


def _filter_clip_signals_for_window(
    clip_signals: Optional[str], window_start: int, window_end: int
) -> Optional[str]:
    """Keep only the signal lines that fall inside one analysis window."""
    if not clip_signals:
        return None
    header: List[str] = []
    kept: List[str] = []
    for line in clip_signals.splitlines():
        match = SIGNAL_LINE_PATTERN.match(line.strip())
        if not match:
            if not kept:
                header.append(line)
            continue
        try:
            signal_start = _parse_transcript_timestamp_seconds(match.group("start"))
        except ValueError:
            continue
        if window_start <= signal_start < window_end:
            kept.append(line)
    if not kept:
        return None
    return "\n".join(header + kept)


def _segment_bounds_seconds(segment: TranscriptSegment) -> tuple[int, int]:
    return (
        _parse_transcript_timestamp_seconds(segment.start_time),
        _parse_transcript_timestamp_seconds(segment.end_time),
    )


def _segment_rank_key(segment: TranscriptSegment) -> tuple[int, float]:
    return (
        segment.virality.total_score if segment.virality else 0,
        segment.relevance_score,
    )


def _overlap_ratio(first: tuple[int, int], second: tuple[int, int]) -> float:
    """Overlap as a share of the shorter range (1.0 = one contains the other)."""
    shorter = min(first[1] - first[0], second[1] - second[0])
    if shorter <= 0:
        return 0.0
    overlap = min(first[1], second[1]) - max(first[0], second[0])
    return max(0, overlap) / shorter


def dedupe_overlapping_segments(
    segments: List[TranscriptSegment],
    max_overlap_ratio: float = CANDIDATE_DEDUPE_OVERLAP_RATIO,
) -> List[TranscriptSegment]:
    """Sort candidates best-first and drop ones that mostly repeat a stronger pick.

    Overlapping windows can surface the same moment twice; keep the higher score.
    """
    kept: List[tuple[TranscriptSegment, tuple[int, int]]] = []
    for segment in sorted(segments, key=_segment_rank_key, reverse=True):
        bounds = _segment_bounds_seconds(segment)
        if any(
            _overlap_ratio(bounds, kept_bounds) > max_overlap_ratio
            for _, kept_bounds in kept
        ):
            logger.info(
                "Dropping overlapping candidate %s-%s", segment.start_time, segment.end_time
            )
            continue
        kept.append((segment, bounds))
    return [segment for segment, _ in kept]


def max_final_clips_for_duration(
    source_seconds: float,
    max_clips: int = MAX_FINAL_CLIPS,
    *,
    prefs: ClipSelectionPrefs | None = None,
    min_clips: int | None = None,
    duration_preset: str | None = None,
) -> int:
    """Discovery ceiling: physical fit × server safety (not a fill-to-N target).

    With prefs, uses ``discovery_ceiling``. Without prefs, same physical/safety
    clamp using the Auto duration preset. Render limit is applied separately.
    """
    selection = prefs or parse_clip_selection_prefs(
        duration_preset=duration_preset or "auto",
        min_clips=min_clips,
    )
    if min_clips is not None and prefs is not None and prefs.min_clips is None:
        selection = ClipSelectionPrefs(
            duration_preset=selection.duration_preset,
            min_clips=min_clips,
            brief=selection.brief,
        )
    return discovery_ceiling(
        source_seconds,
        selection,
        max_clips=max_clips,
        absolute_ceiling=MAX_FINAL_CLIPS,
    )


def _rescale_virality(virality: ViralityAnalysis, new_total: int) -> None:
    """Set a new total while keeping the subscores' proportions and 0-25 bounds."""
    new_total = max(0, min(100, int(new_total)))
    subscores = [
        virality.hook_score,
        virality.engagement_score,
        virality.value_score,
        virality.shareability_score,
    ]
    old_total = sum(subscores)
    if old_total <= 0:
        scaled = [new_total / 4] * 4
    else:
        scaled = [score * new_total / old_total for score in subscores]
    rounded = [min(25, max(0, int(value))) for value in scaled]
    remainder = new_total - sum(rounded)
    by_fraction = sorted(range(4), key=lambda i: scaled[i] - int(scaled[i]), reverse=True)
    while remainder > 0:
        progressed = False
        for index in by_fraction:
            if remainder <= 0:
                break
            if rounded[index] < 25:
                rounded[index] += 1
                remainder -= 1
                progressed = True
        if not progressed:
            break
    (
        virality.hook_score,
        virality.engagement_score,
        virality.value_score,
        virality.shareability_score,
    ) = rounded
    virality.total_score = sum(rounded)


def apply_candidate_ranking(
    candidates: List[TranscriptSegment], ranking: CandidateRanking
) -> List[TranscriptSegment]:
    """Reorder candidates by the rerank and give them its calibrated scores.

    Scores are clamped to be non-increasing down the ranking. Candidates the
    reranker skipped keep their first-pass order after the ranked ones.
    """
    ordered: List[tuple[TranscriptSegment, int]] = []
    seen: set[int] = set()
    for item in ranking.ranking:
        candidate_id = item.candidate_id
        if candidate_id in seen or not 0 <= candidate_id < len(candidates):
            continue
        seen.add(candidate_id)
        ordered.append((candidates[candidate_id], item.score))
    if not ordered:
        raise ValueError("Rerank returned no usable candidate ids")

    reranked: List[TranscriptSegment] = []
    ceiling = 100
    for segment, score in ordered:
        ceiling = min(score, ceiling)
        _rescale_virality(segment.virality, ceiling)
        reranked.append(segment)
    for index, segment in enumerate(candidates):
        if index in seen:
            continue
        ceiling = min(segment.virality.total_score, ceiling)
        _rescale_virality(segment.virality, ceiling)
        reranked.append(segment)
    return reranked


async def rerank_candidates(candidates: List[TranscriptSegment]) -> List[TranscriptSegment]:
    """Compare all candidates in one call; keep first-pass order if that fails."""
    if len(candidates) < 2:
        return candidates
    try:
        agent = get_rerank_agent()
        result = await _run_transcript_analysis(
            agent,
            build_candidate_rerank_prompt(candidates),
            timeout_seconds=RERANK_TIMEOUT_SECONDS,
        )
        reranked = apply_candidate_ranking(candidates, result.output)
    except Exception as error:
        # Do not log the error text: provider bodies may contain transcript data.
        logger.warning(
            "Candidate rerank failed (%s); keeping first-pass order", type(error).__name__
        )
        return candidates
    logger.info("Reranked %d clip candidates", len(reranked))
    return reranked


async def _analyze_transcript_windows(
    agent: Agent,
    windows: List[Dict[str, Any]],
    include_broll: bool,
    clip_signals: Optional[str],
    prefs: ClipSelectionPrefs | None = None,
    clip_budget: int | None = None,
) -> List[TranscriptAnalysis]:
    """Analyse windows with bounded concurrency, tolerating some failures.

    Raises the first window error only when no window produced an analysis.
    """
    semaphore = asyncio.Semaphore(ANALYSIS_WINDOW_CONCURRENCY)
    selection = prefs or ClipSelectionPrefs()

    async def run_window(window: Dict[str, Any]) -> TranscriptAnalysis:
        async with semaphore:
            prompt = build_transcript_analysis_prompt(
                transcript=window["transcript"],
                include_broll=include_broll,
                clip_signals=_filter_clip_signals_for_window(
                    clip_signals, window["start"], window["end"]
                ),
                window=(window["start"], window["end"]),
                prefs=selection,
                clip_budget=clip_budget,
            )
            result = await _run_transcript_analysis(agent, prompt)
            return result.output

    tasks = [asyncio.create_task(run_window(window)) for window in windows]
    try:
        _, pending = await asyncio.wait(tasks, timeout=ANALYSIS_TOTAL_TIMEOUT_SECONDS)
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    if pending:
        logger.warning(
            "%d of %d analysis windows exceeded the %ss budget",
            len(pending),
            len(windows),
            ANALYSIS_TOTAL_TIMEOUT_SECONDS,
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    analyses: List[TranscriptAnalysis] = []
    errors: List[BaseException] = []
    for index, task in enumerate(tasks):
        if task in pending:
            continue
        error = task.exception()
        if error is not None:
            logger.warning(
                "Analysis window %d/%d failed (%s)", index + 1, len(windows), type(error).__name__
            )
            errors.append(error)
            continue
        analyses.append(task.result())

    if not analyses:
        if errors:
            raise errors[0]
        raise RuntimeError(
            "The AI analysis service took too long to respond. Please try again shortly."
        )
    return analyses


def _snap_segment_bounds(
    segment: TranscriptSegment,
    transcript_lines: List[Dict[str, Any]],
    start_seconds: int,
    end_seconds: int,
) -> tuple[int, int]:
    """Move a segment onto transcript line boundaries when they are close."""
    snapped = _snap_bounds_to_transcript_lines(transcript_lines, start_seconds, end_seconds)
    if not snapped:
        return start_seconds, end_seconds
    start_line, end_line = snapped
    if (start_line["start_time"], end_line["end_time"]) != (
        segment.start_time,
        segment.end_time,
    ):
        logger.info(
            "Snapped segment %s-%s to transcript lines %s-%s",
            segment.start_time,
            segment.end_time,
            start_line["start_time"],
            end_line["end_time"],
        )
        segment.start_time = start_line["start_time"]
        segment.end_time = end_line["end_time"]
    return start_line["start_seconds"], end_line["end_seconds"]


def _validate_segments(
    segments: List[TranscriptSegment],
    transcript_lines: List[Dict[str, Any]],
    transcript_spans: List[Dict[str, Any]],
    prefs: ClipSelectionPrefs | None = None,
) -> List[TranscriptSegment]:
    """Ground model segments onto real transcript lines and drop unusable ones."""
    selection = prefs or ClipSelectionPrefs()
    min_s = selection.min_seconds
    max_s = selection.max_seconds
    validated_segments: List[TranscriptSegment] = []
    for segment in segments:
        # Validate text content
        if not segment.text.strip() or len(segment.text.split()) < 3:
            logger.warning(
                f"Skipping segment with insufficient content: '{segment.text[:50]}...'"
            )
            continue

        # Validate timestamps - CRITICAL: start and end must be different
        if segment.start_time == segment.end_time:
            logger.warning(
                f"Skipping segment with identical start/end times: {segment.start_time}"
            )
            continue

        # Parse timestamps to validate duration
        try:
            start_seconds = _parse_transcript_timestamp_seconds(segment.start_time)
            end_seconds = _parse_transcript_timestamp_seconds(segment.end_time)
            start_seconds, end_seconds = _snap_segment_bounds(
                segment, transcript_lines, start_seconds, end_seconds
            )
            duration = end_seconds - start_seconds

            if duration < min_s or duration > max_s:
                repaired_bounds = _repair_segment_bounds(
                    segment,
                    transcript_spans,
                    start_seconds,
                    end_seconds,
                    prefs=selection,
                )
                if repaired_bounds:
                    # Repairs format hours as HH:MM:SS; snap back onto line labels.
                    start_seconds, end_seconds = _snap_segment_bounds(
                        segment, transcript_lines, *repaired_bounds
                    )
                    duration = end_seconds - start_seconds

            if duration <= 0:
                logger.warning(
                    f"Skipping segment with invalid duration: {segment.start_time} to {segment.end_time} = {duration}s"
                )
                continue

            if duration < min_s:
                logger.warning(
                    f"Skipping segment too short: {duration}s (min {min_s}s required)"
                )
                continue

            if duration > max_s:
                logger.warning(
                    f"Skipping segment too long: {duration}s (max {max_s}s allowed)"
                )
                continue

            grounded_text = _extract_transcript_text_for_segment(
                transcript_lines,
                segment.start_time,
                segment.end_time,
            )
            if grounded_text is None:
                logger.warning(
                    "Skipping segment with timestamps not aligned to transcript lines: %s-%s",
                    segment.start_time,
                    segment.end_time,
                )
                continue
            if _normalize_transcript_text(segment.text) != _normalize_transcript_text(
                grounded_text
            ):
                logger.warning(
                    "Adjusting segment text to transcript-backed span for %s-%s",
                    segment.start_time,
                    segment.end_time,
                )
                segment.text = grounded_text

            # Validate virality scores
            if segment.virality:
                # Ensure total score is sum of subscores
                calculated_total = (
                    segment.virality.hook_score
                    + segment.virality.engagement_score
                    + segment.virality.value_score
                    + segment.virality.shareability_score
                )
                if segment.virality.total_score != calculated_total:
                    logger.warning(
                        f"Correcting virality total: {segment.virality.total_score} -> {calculated_total}"
                    )
                    segment.virality.total_score = calculated_total

            segment.hook_title = sanitize_hook_title(segment.hook_title)
            # Bounds are final here (after snapping and repair): keep the teaser
            # only when it still sits inside the clip.
            _sanitize_hook_span(segment, start_seconds, end_seconds)

            validated_segments.append(segment)
            virality_info = (
                f", virality={segment.virality.total_score}"
                if segment.virality
                else ""
            )
            logger.info(
                f"Validated segment: {segment.start_time}-{segment.end_time} ({duration}s){virality_info}"
            )

        except (ValueError, IndexError) as e:
            logger.warning(
                f"Skipping segment with invalid timestamp format: {segment.start_time}-{segment.end_time}: {e}"
            )
            continue
    return validated_segments


# Prefer strong skip markers so soft "não mirar 30/60/90" guidance is not treated
# as a forbidden-topic list. Bare "não" only harvests quoted phrases.
_BRIEF_STRONG_EXCLUSION_RE = re.compile(
    r"(?i)\b("
    r"evitar|evite|skip|forbid|forbidden|proibid[oa]s?|"
    r"exclude|excluir|exclua|never|nunca|"
    r"n[aã]o\s+selecione|n[aã]o\s+usar|n[aã]o\s+use|"
    r"do\s+not|don't|dont"
    r")\b"
)
_BRIEF_WEAK_NAO_RE = re.compile(r"(?i)\bn[aã]o\b")
_BRIEF_QUOTED_PHRASE_RE = re.compile(
    r"[\"“”'‘’«»]([^\"“”'‘’«»]{3,160})[\"“”'‘’«»]"
)
_BRIEF_LIST_LABEL_RE = re.compile(
    r"(?i)^(t[oó]picos?|topics?|clips?|trechos?|[aâ]ngulos?|hooks?|ganchos?)"
    r"(\s+j[aá]\s+editados?)?\s*:?\s*"
)


def _normalize_brief_match_text(value: str) -> str:
    # Fold accents so "Não" in a title matches "Nao" in the brief.
    decomposed = unicodedata.normalize("NFKD", value or "")
    ascii_folded = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", ascii_folded).strip().lower()


def _extract_forbidden_phrases(brief: str) -> List[str]:
    """Pull explicit skip/forbidden phrases from a user brief for post-filters."""
    if not brief or not str(brief).strip():
        return []

    phrases: List[str] = []
    seen: set[str] = set()

    def add_phrase(raw: str) -> None:
        normalized = _normalize_brief_match_text(raw)
        normalized = normalized.strip(" .:;,-–—")
        if len(normalized) < 3 or normalized in seen:
            return
        # Ignore tiny instructional leftovers.
        if normalized in {"usar", "use", "select", "selecionar", "clips", "trechos"}:
            return
        seen.add(normalized)
        phrases.append(normalized)

    def add_list_after_marker(text: str, match: re.Match[str]) -> None:
        after = text[match.end() :]
        after = re.sub(r"^[\s:–—\-•*]+", "", after).strip()
        after = _BRIEF_LIST_LABEL_RE.sub("", after).strip()
        quoted = _BRIEF_QUOTED_PHRASE_RE.findall(after) or _BRIEF_QUOTED_PHRASE_RE.findall(
            text
        )
        if quoted:
            for quote in quoted:
                add_phrase(quote)
        for part in re.split(r"[;\n•]+", after):
            part = part.strip().strip(" .:;,-–—")
            part = _BRIEF_LIST_LABEL_RE.sub("", part).strip()
            if part:
                add_phrase(part)

    for line in str(brief).splitlines():
        line = line.strip()
        if not line:
            continue
        strong = _BRIEF_STRONG_EXCLUSION_RE.search(line)
        if strong:
            add_list_after_marker(line, strong)
            continue
        if _BRIEF_WEAK_NAO_RE.search(line):
            for quote in _BRIEF_QUOTED_PHRASE_RE.findall(line):
                add_phrase(quote)

    # Mid-paragraph "EVITAR tópicos: A; B" when brief is one long line.
    for match in _BRIEF_STRONG_EXCLUSION_RE.finditer(str(brief)):
        add_list_after_marker(str(brief), match)

    return phrases


def _segment_violates_brief(
    segment: TranscriptSegment, phrases: List[str]
) -> tuple[bool, Optional[str]]:
    if not phrases:
        return False, None
    haystacks = [
        _normalize_brief_match_text(segment.text),
        _normalize_brief_match_text(segment.hook_title or ""),
        _normalize_brief_match_text(segment.reasoning),
    ]
    for phrase in phrases:
        for haystack in haystacks:
            if phrase and phrase in haystack:
                return True, phrase
    return False, None


_WEAK_OPENER_RE = re.compile(
    r"^\s*(?:"
    r"ol[aá]\b|oi\b|e\s+a[ií]\b|fala\s+(?:galera|pessoal|gente)\b|"
    r"ent[aã]o\s+pessoal\b|bem[- ]?vind[oa]s?\b|hoje\s+(?:eu\s+)?vou\b|"
    r"nesse\s+v[ií]deo\b|neste\s+v[ií]deo\b|no\s+v[ií]deo\s+de\s+hoje\b|"
    r"hey\s+(?:guys|everyone)\b|hello\s+everyone\b|welcome\s+back\b|"
    r"so\s+(?:guys|folks)\b|today\s+(?:i'?m|we'?re)\s+going\s+to\b|"
    r"in\s+this\s+video\b|alright\s+(?:so|guys)\b"
    r")",
    re.IGNORECASE,
)


def _segment_total_score(segment: TranscriptSegment) -> int:
    if segment.virality is None:
        return 0
    return int(segment.virality.total_score or 0)


def _segment_hook_score(segment: TranscriptSegment) -> int:
    if segment.virality is None:
        return 0
    return int(segment.virality.hook_score or 0)


def _segment_has_weak_opener(segment: TranscriptSegment) -> bool:
    text = (segment.text or "").strip()
    if not text:
        return True
    # First ~3s proxy: opening clause / first ~12 words.
    head = " ".join(text.split()[:12])
    return bool(_WEAK_OPENER_RE.search(head))


def apply_quality_gate(
    candidates: List[TranscriptSegment],
    *,
    min_total: int = QUALITY_GATE_MIN_TOTAL,
    min_hook: int = QUALITY_GATE_MIN_HOOK,
    top_gap: int = QUALITY_GATE_TOP_GAP,
    min_start_gap_seconds: int = QUALITY_GATE_MIN_START_GAP_SECONDS,
    fallback_best: bool = True,
) -> tuple[List[TranscriptSegment], List[Dict[str, Any]]]:
    """Keep only excellent clips: absolute + relative scores, gap, weak-opener.

    Returns ``(kept, drop_log)``. Soft fail: if nothing passes, keep the single
    best candidate so the task is never empty.
    """
    if not candidates:
        return [], []

    ranked = sorted(
        candidates,
        key=lambda s: (_segment_total_score(s), _segment_hook_score(s), s.relevance_score),
        reverse=True,
    )
    top_total = _segment_total_score(ranked[0])
    floor_total = max(min_total, top_total - top_gap)

    excellent: List[TranscriptSegment] = []
    drop_log: List[Dict[str, Any]] = []
    accepted_starts: List[int] = []

    for segment in ranked:
        total = _segment_total_score(segment)
        hook = _segment_hook_score(segment)
        try:
            start_s = _parse_transcript_timestamp_seconds(segment.start_time)
        except ValueError:
            start_s = None

        reason: Optional[str] = None
        if total < min_total:
            reason = "low_total"
        elif hook < min_hook:
            reason = "low_hook"
        elif total < floor_total:
            reason = "far_from_top"
        elif _segment_has_weak_opener(segment):
            reason = "weak_opener"
        elif start_s is not None and any(
            abs(start_s - prev) < min_start_gap_seconds for prev in accepted_starts
        ):
            reason = "too_close_in_time"

        if reason:
            drop_entry = {
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "reason": reason,
                "total_score": total,
                "hook_score": hook,
                "hook_title": segment.hook_title,
            }
            drop_log.append(drop_entry)
            logger.info(
                "Quality gate dropped %s-%s reason=%s total=%s hook=%s top=%s title=%r",
                segment.start_time,
                segment.end_time,
                reason,
                total,
                hook,
                top_total,
                segment.hook_title,
            )
            continue

        excellent.append(segment)
        if start_s is not None:
            accepted_starts.append(start_s)

    if not excellent and fallback_best:
        best = ranked[0]
        logger.info(
            "Quality gate empty — fallback to best segment %s-%s (total=%s)",
            best.start_time,
            best.end_time,
            _segment_total_score(best),
        )
        return [best], drop_log

    # Best-first for render slice / generate-more order.
    excellent.sort(
        key=lambda s: (_segment_total_score(s), _segment_hook_score(s), s.relevance_score),
        reverse=True,
    )
    logger.info(
        "Quality gate kept %s/%s excellent (min_total=%s min_hook=%s top=%s floor=%s drops=%s)",
        len(excellent),
        len(candidates),
        min_total,
        min_hook,
        top_total,
        floor_total,
        len(drop_log),
    )
    return excellent, drop_log


def _filter_segments_by_brief(
    candidates: List[TranscriptSegment],
    brief: str,
    clip_budget: int,
) -> List[TranscriptSegment]:
    """Drop brief-forbidden segments up to the discovery ceiling."""
    budget = max(0, int(clip_budget))
    if budget <= 0:
        return []
    phrases = _extract_forbidden_phrases(brief)
    if not phrases:
        return candidates[:budget]

    kept: List[TranscriptSegment] = []
    dropped = 0
    for segment in candidates:
        violates, phrase = _segment_violates_brief(segment, phrases)
        if violates:
            dropped += 1
            logger.info(
                "Brief filter dropped segment %s-%s reason=brief (phrase=%r, hook_title=%r)",
                segment.start_time,
                segment.end_time,
                phrase,
                segment.hook_title,
            )
            continue
        kept.append(segment)
        if len(kept) >= budget:
            break

    if dropped:
        logger.info(
            "Brief filter removed %s segment(s); kept %s of discovery ceiling %s (phrases=%s)",
            dropped,
            len(kept),
            budget,
            phrases[:8],
        )
    return kept


def _merge_key_topics(analyses: List[TranscriptAnalysis], limit: int = 12) -> List[str]:
    topics: List[str] = []
    seen: set[str] = set()
    for analysis in analyses:
        for topic in analysis.key_topics or []:
            key = topic.strip().lower()
            if key and key not in seen:
                seen.add(key)
                topics.append(topic.strip())
    return topics[:limit]


async def get_most_relevant_parts_by_transcript(
    transcript: str,
    include_broll: bool = False,
    clip_signals: str | None = None,
    prefs: ClipSelectionPrefs | None = None,
) -> TranscriptAnalysis:
    """Get the most relevant parts of a transcript with virality scoring and optional B-roll detection.

    Short transcripts are analysed in one call. Long ones are split into
    overlapping windows analysed concurrently, then deduplicated and reranked
    in one comparative call so scores are comparable across windows.
    """
    selection = prefs or ClipSelectionPrefs()
    logger.info(
        "Starting AI analysis of transcript (%s chars), include_broll=%s, preset=%s, min_clips=%s, brief=%s",
        len(transcript),
        include_broll,
        selection.duration_preset,
        selection.min_clips if selection.min_clips is not None else "auto",
        bool(selection.brief),
    )

    try:
        agent = get_transcript_agent()
        transcript_lines = _parse_transcript_lines(transcript)
        transcript_spans = _parse_transcript_spans(transcript)
        windows = split_transcript_into_windows(transcript)
        source_seconds = max(
            (line["end_seconds"] for line in transcript_lines), default=0
        )
        server_max = get_config().max_clips
        harvest_ceiling = max_final_clips_for_duration(
            source_seconds,
            server_max,
            prefs=selection,
        )
        render_limit = render_ceiling(selection, discovery=harvest_ceiling)

        if len(windows) == 1:
            result = await _run_transcript_analysis(
                agent,
                build_transcript_analysis_prompt(
                    transcript=transcript,
                    include_broll=include_broll,
                    clip_signals=clip_signals,
                    prefs=selection,
                    clip_budget=harvest_ceiling,
                )
            )
            analyses = [result.output]
        else:
            logger.info("Analysing transcript in %d overlapping windows", len(windows))
            analyses = await _analyze_transcript_windows(
                agent,
                windows,
                include_broll,
                clip_signals,
                prefs=selection,
                clip_budget=harvest_ceiling,
            )

        candidates: List[TranscriptSegment] = []
        for analysis in analyses:
            logger.info(
                f"AI analysis found {len(analysis.most_relevant_segments)} segments"
            )
            candidates.extend(
                _validate_segments(
                    analysis.most_relevant_segments,
                    transcript_lines,
                    transcript_spans,
                    prefs=selection,
                )
            )

        # Best-first by virality (primary) then relevance (secondary), without
        # near-duplicates; then compare across windows in one rerank call.
        candidates = dedupe_overlapping_segments(candidates)
        if len(windows) > 1:
            candidates = await rerank_candidates(candidates)

        brief_filtered = _filter_segments_by_brief(
            candidates,
            selection.brief,
            harvest_ceiling,
        )
        # Brief drops are also recorded for calibration.
        brief_phrases = _extract_forbidden_phrases(selection.brief)
        brief_drop_log: List[Dict[str, Any]] = []
        if brief_phrases:
            kept_ids = {id(s) for s in brief_filtered}
            for segment in candidates:
                if id(segment) in kept_ids:
                    continue
                violates, phrase = _segment_violates_brief(segment, brief_phrases)
                if violates:
                    brief_drop_log.append(
                        {
                            "start_time": segment.start_time,
                            "end_time": segment.end_time,
                            "reason": "brief",
                            "phrase": phrase,
                            "total_score": _segment_total_score(segment),
                            "hook_score": _segment_hook_score(segment),
                            "hook_title": segment.hook_title,
                        }
                    )

        excellent_segments, gate_drop_log = apply_quality_gate(brief_filtered)
        excellent_segments = await rewrite_hook_titles(excellent_segments)
        excellent_count = len(excellent_segments)
        validated_segments = excellent_segments[:render_limit]
        drop_log = brief_drop_log + gate_drop_log

        broll_opportunities = None
        if include_broll:
            broll_opportunities = [
                opportunity
                for analysis in analyses
                for opportunity in (analysis.broll_opportunities or [])
            ] or None

        final_analysis = TranscriptAnalysis(
            most_relevant_segments=validated_segments,
            summary=" ".join(
                analysis.summary.strip() for analysis in analyses if analysis.summary
            ),
            key_topics=_merge_key_topics(analyses),
            broll_opportunities=broll_opportunities,
        )
        final_analysis._excellent_clip_count = excellent_count
        final_analysis._render_ceiling = render_limit
        final_analysis._discovery_ceiling = harvest_ceiling
        final_analysis._excellent_segments = excellent_segments
        final_analysis._drop_log = drop_log

        logger.info(
            "Selected %s/%s excellent for render (discovery=%s render_ceiling=%s drops=%s)",
            len(validated_segments),
            excellent_count,
            harvest_ceiling,
            render_limit,
            len(drop_log),
        )
        if validated_segments:
            top = validated_segments[0]
            logger.info(
                f"Top segment - relevance: {top.relevance_score:.2f}, virality: {top.virality.total_score if top.virality else 'N/A'}"
            )

        return final_analysis

    except Exception as e:
        logger.error(f"Error in transcript analysis: {e}")
        raise RuntimeError(f"Transcript analysis failed: {str(e)}") from e


def get_most_relevant_parts_sync(transcript: str) -> TranscriptAnalysis:
    """Synchronous wrapper for the async function."""
    return asyncio.run(get_most_relevant_parts_by_transcript(transcript))
