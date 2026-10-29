"""Unit tests for full-transcript export helpers and language wiring."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.media.transcription import (
    build_plain_text_from_transcript_data,
    build_srt_from_transcript_data,
    get_video_transcript,
    normalize_transcript_payload,
    unwrap_transcript_payload_from_cache,
    wrap_transcript_payload_for_cache,
)
from src.services.task_service import TaskService


def test_build_srt_from_utterances_includes_speakers():
    payload = {
        "text": "Oi mundo",
        "words": [],
        "utterances": [
            {
                "text": "Oi mundo",
                "start": 1000,
                "end": 2500,
                "speaker": "A",
            }
        ],
        "language_code": "pt",
    }
    srt = build_srt_from_transcript_data(payload)
    assert "00:00:01,000 --> 00:00:02,500" in srt
    assert "Speaker A: Oi mundo" in srt
    assert srt.strip().startswith("1\n")


def test_build_srt_from_words_chunks_sentences():
    payload = {
        "text": "",
        "utterances": [],
        "words": [
            {"text": "Olá", "start": 0, "end": 400},
            {"text": "mundo.", "start": 400, "end": 900},
            {"text": "Tudo", "start": 1200, "end": 1600},
            {"text": "bem?", "start": 1600, "end": 2000},
        ],
    }
    srt = build_srt_from_transcript_data(payload)
    assert "Olá mundo." in srt
    assert "Tudo bem?" in srt


def test_plain_text_prefers_text_field():
    assert (
        build_plain_text_from_transcript_data(
            {"text": "completo", "words": [], "utterances": []}
        )
        == "completo"
    )


def test_plain_text_prefers_cues_with_speaker_map_and_timestamps():
    from src.media.transcription import resolve_speaker_label

    payload = {
        "text": "flat text without speakers",
        "words": [],
        "utterances": [
            {
                "text": "Oi mundo",
                "start": 1000,
                "end": 2500,
                "speaker": "A",
            }
        ],
    }
    text = build_plain_text_from_transcript_data(
        payload, speaker_map={"A": "Maria"}
    )
    assert "[00:01 - 00:02] Maria: Oi mundo" in text
    assert resolve_speaker_label("A", {"A": "Maria"}) == "Maria"


def test_srt_uses_speaker_map_labels():
    payload = {
        "text": "Oi",
        "words": [],
        "utterances": [
            {"text": "Oi", "start": 0, "end": 500, "speaker": "A"},
        ],
    }
    srt = build_srt_from_transcript_data(payload, speaker_map={"A": "João"})
    assert "João: Oi" in srt
    assert "Speaker A:" not in srt


def test_wrap_unwrap_transcript_payload_roundtrip():
    payload = {
        "version": 1,
        "text": "olá",
        "words": [{"text": "olá", "start": 0, "end": 100}],
        "utterances": [],
        "language_code": "pt",
    }
    wrapped = wrap_transcript_payload_for_cache(payload)
    restored = unwrap_transcript_payload_from_cache(wrapped)
    assert restored is not None
    assert restored["text"] == "olá"
    assert restored["language_code"] == "pt"
    assert len(restored["words"]) == 1


def test_normalize_accepts_wrapped_and_raw():
    raw = normalize_transcript_payload(
        {"text": "x", "words": [], "utterances": [], "language_code": "pt"}
    )
    wrapped = normalize_transcript_payload(
        {"kind": "transcript_payload", "payload": raw}
    )
    assert raw == wrapped


def test_transcribe_cache_key_isolates_language_and_model():
    key_pt = TaskService._build_transcribe_cache_key(
        "https://youtu.be/abc",
        "youtube",
        provider="assemblyai",
        speech_model="best",
        language="pt",
    )
    key_en = TaskService._build_transcribe_cache_key(
        "https://youtu.be/abc",
        "youtube",
        provider="assemblyai",
        speech_model="best",
        language="en",
    )
    key_nano = TaskService._build_transcribe_cache_key(
        "https://youtu.be/abc",
        "youtube",
        provider="assemblyai",
        speech_model="nano",
        language="pt",
    )
    assert key_pt != key_en
    assert key_pt != key_nano


def test_get_video_transcript_passes_language_override_to_assemblyai():
    runtime = MagicMock()
    runtime.transcription_provider = "assemblyai"
    runtime.transcription_language = ""
    runtime.assembly_ai_api_key = "test-key"
    runtime.assembly_ai_http_timeout_seconds = 30

    with (
        patch("src.media.transcription.get_config", return_value=runtime),
        patch(
            "src.media.transcription._get_transcript_with_assemblyai",
            return_value="ok",
        ) as mock_aai,
    ):
        result = get_video_transcript(
            Path("/tmp/video.mp4"),
            speech_model="best",
            language="pt",
        )

    assert result == "ok"
    mock_aai.assert_called_once()
    assert mock_aai.call_args.kwargs.get("language") == "pt" or (
        len(mock_aai.call_args.args) >= 4 and mock_aai.call_args.args[3] == "pt"
    )


def test_merge_metadata_accepts_job_kind_and_language():
    from src.api.routes.tasks import _merge_task_source_metadata

    merged = _merge_task_source_metadata(
        {"processing_job_id": "x"},
        source_url="https://youtu.be/abc",
        source_type="youtube",
        job_kind="transcribe",
        language="pt",
    )
    assert merged["job_kind"] == "transcribe"
    assert merged["language"] == "pt"
    assert merged["url"] == "https://youtu.be/abc"
