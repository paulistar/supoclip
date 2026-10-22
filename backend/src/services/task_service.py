"""
Task service - orchestrates task creation and processing workflow.
"""

from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, Optional, Callable, Mapping, Sequence
import asyncio
import logging
import re
from datetime import datetime
from pathlib import Path
import json
import hashlib
from time import perf_counter


from arq.constants import job_key_prefix, in_progress_key_prefix, expires_extra_ms
from ..workers.job_queue import JobQueue
from ..repositories.task_repository import TaskRepository
from ..repositories.source_repository import SourceRepository
from ..repositories.clip_repository import ClipRepository
from ..repositories.cache_repository import CacheRepository
from .video_service import VideoService
from .clip_service import ClipEditingMixin
from ..repositories.edit_transaction import (
    commit_unless_editing,
    serialized_task_edit,
    task_edit_transaction,
    TaskCancelled,
)
from ..repositories.task_run_guard import exclusive_task_run, task_run_guard
from .billing_service import BillingService
from .task_completion_email_service import (
    TaskCompletionEmailService,
    TaskCompletionRecipient,
)
from ..config import Config, get_config
from ..database import ensure_db_alive
from ..youtube_utils import cleanup_downloaded_files, extract_video_id
from ..clip_cleanup import normalize_clip_cleanup_settings
from ..ai import TRANSCRIPT_ANALYSIS_CACHE_VERSION
from ..clip_selection import ClipSelectionPrefs, parse_clip_selection_prefs
from ..clip_review import (
    append_evitar_phrases,
    build_candidate_rows,
    estimate_render_seconds,
    select_candidates_for_render,
    segment_duration_seconds,
)
from ..clip_source_map import load_clip_caption_settings
from ..media.transcription import (
    build_plain_text_from_transcript_data,
    build_srt_from_transcript_data,
    load_cached_transcript_data,
    normalize_transcript_payload,
    unwrap_transcript_payload_from_cache,
    wrap_transcript_payload_for_cache,
)
from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError

logger = logging.getLogger(__name__)
PROCESSING_CACHE_VERSION = "20260319_grounded_segments_v1"
TRANSCRIBE_CACHE_VERSION = "20261022_transcribe_v1"


def _is_closed_db_connection(exc: BaseException) -> bool:
    """True when Postgres/asyncpg dropped the worker's checked-out connection."""
    if isinstance(exc, (InterfaceError, OperationalError)):
        return True
    if isinstance(exc, DBAPIError) and getattr(exc, "connection_invalidated", False):
        return True
    message = str(exc).lower()
    return (
        "connection is closed" in message
        or "underlying connection is closed" in message
        or "connectiondoesnotexist" in message.replace(" ", "")
    )


class TaskService(ClipEditingMixin):
    """Service for task workflow orchestration."""

    def __init__(self, db: AsyncSession, config: Config | None = None):
        self.db = db
        self.task_repo = TaskRepository()
        self.source_repo = SourceRepository()
        self.clip_repo = ClipRepository()
        self.cache_repo = CacheRepository()
        self.video_service = VideoService()
        self.config = config or get_config()

    @staticmethod
    def _build_cache_key(
        url: str,
        source_type: str,
        processing_mode: str,
        prefs: ClipSelectionPrefs | None = None,
    ) -> str:
        selection = prefs or ClipSelectionPrefs()
        payload = (
            f"{source_type}|{processing_mode}|"
            f"{PROCESSING_CACHE_VERSION}|{TRANSCRIPT_ANALYSIS_CACHE_VERSION}|"
            f"{selection.cache_token()}|{url.strip()}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _build_transcribe_cache_key(
        url: str,
        source_type: str,
        *,
        provider: str,
        speech_model: str,
        language: str,
    ) -> str:
        payload = (
            f"transcribe|{TRANSCRIBE_CACHE_VERSION}|{source_type}|"
            f"{provider}|{speech_model}|{language or 'auto'}|{url.strip()}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _transcribe_speech_model(processing_mode: str, provider: str) -> str:
        if processing_mode in {"fast", "balanced"} and provider == "assemblyai":
            return get_config().fast_mode_transcript_model
        return "best"

    async def _render_segments_concurrent(
        self,
        *,
        task_id: str,
        segments: Sequence[Mapping[str, Any]] | Sequence[Dict[str, Any]],
        video_path: Path,
        clips_output_dir: Path,
        font_family,
        font_size,
        font_color,
        caption_template,
        output_format,
        add_subtitles,
        cleanup_settings: Dict[str, Any],
        base_order: int,
        existing_clip_ids: list[str],
        progress_base: int,
        progress_span: int,
        check_cancelled,
        update_progress,
        clip_ready_callback=None,
        progress_label: str = "Creating clip",
    ) -> list[str]:
        """Render clips with bounded CPU concurrency; persist DB rows in order."""
        total = len(segments)
        if total == 0:
            return list(existing_clip_ids)

        concurrency = max(1, int(getattr(self.config, "clip_render_concurrency", 1) or 1))
        clip_ids = list(existing_clip_ids)
        pending_clip_path: Optional[Path] = None

        for wave_start in range(0, total, concurrency):
            await check_cancelled()
            # Release any autobegin txn (e.g. prior SELECT) before multi-minute
            # ffmpeg — idle_in_transaction_session_timeout is 5m in production.
            await commit_unless_editing(self.db)
            wave = list(segments[wave_start : wave_start + concurrency])
            wave_indexes = list(range(wave_start, wave_start + len(wave)))

            async def _render_one(local_i: int, segment: Mapping[str, Any] | Dict[str, Any]):
                return local_i, await self.video_service.create_single_clip(
                    video_path,
                    segment,
                    base_order + local_i,
                    clips_output_dir,
                    font_family,
                    font_size,
                    font_color,
                    caption_template,
                    output_format,
                    add_subtitles,
                    cleanup_settings,
                )

            rendered = await asyncio.gather(
                *[_render_one(i, seg) for i, seg in zip(wave_indexes, wave)]
            )
            # Persist in clip order so clip_order / SSE stay sequential.
            for local_i, clip_info in sorted(rendered, key=lambda item: item[0]):
                await check_cancelled()
                clip_progress = progress_base + (
                    int(((local_i + 1) / total) * progress_span) if total else progress_span
                )
                await self._db_after_render_gap(
                    update_progress,
                    clip_progress,
                    f"{progress_label} {local_i + 1}/{total}…",
                )
                if clip_info is None:
                    continue
                pending_clip_path = Path(clip_info["path"])

                async def _persist_clip():
                    async with self._processing_transaction(task_id):
                        await check_cancelled()
                        new_id = await self.clip_repo.create_clip(
                            self.db,
                            task_id=task_id,
                            filename=clip_info["filename"],
                            file_path=clip_info["path"],
                            start_time=clip_info["start_time"],
                            end_time=clip_info["end_time"],
                            duration=clip_info["duration"],
                            text=clip_info.get("text", ""),
                            relevance_score=clip_info.get("relevance_score", 0.0),
                            reasoning=clip_info.get("reasoning", ""),
                            clip_order=base_order + local_i + 1,
                            virality_score=clip_info.get("virality_score", 0),
                            hook_score=clip_info.get("hook_score", 0),
                            engagement_score=clip_info.get("engagement_score", 0),
                            value_score=clip_info.get("value_score", 0),
                            shareability_score=clip_info.get("shareability_score", 0),
                            hook_type=clip_info.get("hook_type"),
                            hook_title=clip_info.get("hook_title"),
                        )
                        new_ids = [*clip_ids, new_id]
                        await self.task_repo.update_task_clips(self.db, task_id, new_ids)
                        return new_id, new_ids

                clip_id, clip_ids = await self._db_after_render_gap(_persist_clip)
                pending_clip_path = None
                if clip_ready_callback:

                    async def _load_clip_ready():
                        clip_record = await self.clip_repo.get_clip_by_id(
                            self.db, clip_id
                        )
                        # End the autobegin transaction from this SELECT so the
                        # next ffmpeg wave is not idle-in-transaction for 5+ min.
                        await commit_unless_editing(self.db)
                        return clip_record

                    clip_record = await self._db_after_render_gap(_load_clip_ready)
                    if clip_record:
                        await clip_ready_callback(
                            base_order + local_i,
                            base_order + total,
                            clip_record,
                        )

        _ = pending_clip_path  # kept for parity with cancel cleanup paths
        return clip_ids

    async def _db_after_render_gap(self, op, *args, **kwargs):
        """Run a DB op after a long await; probe first and retry once if closed."""
        await ensure_db_alive(self.db)
        try:
            return await op(*args, **kwargs)
        except Exception as exc:
            if not _is_closed_db_connection(exc):
                raise
            logger.warning(
                "DB connection closed after render gap; retrying once (%s)",
                type(exc).__name__,
            )
            await ensure_db_alive(self.db)
            return await op(*args, **kwargs)

    def _is_stale_queued_task(self, task: Dict[str, Any]) -> bool:
        """Detect queued tasks that have likely stalled due to worker issues."""
        if task.get("status") != "queued":
            return False

        return self._task_age_seconds(task) >= self.config.queued_task_timeout_seconds

    async def _queued_job_is_missing(self, task: Dict[str, Any]) -> bool:
        """Confirm this reservation has no queued/running ARQ job, not worker health.

        A heartbeat gap cannot invalidate a healthy backlog. Redis outages leave
        reservations intact until job existence can be checked again.
        """
        metadata = await self.task_repo.load_source_metadata(self.db, task["id"])
        job_id = metadata.get("processing_job_id")
        if not job_id:
            # Old tasks predate tracked job IDs. Give their jobs the full ARQ
            # retention window instead of dropping them during a worker restart.
            return self._task_age_seconds(task) >= max(
                self.config.queued_task_timeout_seconds, expires_extra_ms / 1000
            )
        try:
            pool = await JobQueue.get_pool()
            return not bool(await pool.exists(
                job_key_prefix + job_id, in_progress_key_prefix + job_id
            ))
        except Exception:
            logger.exception("Could not verify queued job for task %s", task["id"])
            return False

    def _is_stale_processing_task(self, task: Dict[str, Any]) -> bool:
        """Detect processing tasks whose worker likely died mid-run.

        Recovers tasks stuck in "processing" (e.g. arq hard-killed the job past
        its timeout) so they don't stay "processing" forever. The configured
        timeout defaults well above arq's job_timeout, so a legitimately
        long-running job is never falsely swept into "error".
        """
        if task.get("status") != "processing":
            return False

        return (
            self._task_age_seconds(task)
            >= self.config.processing_task_timeout_seconds
        )

    @staticmethod
    def _task_age_seconds(task: Dict[str, Any]) -> float:
        """Seconds since the task's last update (or creation)."""
        created_at = task.get("created_at")
        updated_at = task.get("updated_at") or created_at

        if not created_at or not updated_at:
            return 0.0

        now = (
            datetime.now(updated_at.tzinfo)
            if getattr(updated_at, "tzinfo", None)
            else datetime.utcnow()
        )
        return (now - updated_at).total_seconds()

    @staticmethod
    def _cleanup_source_video(
        video_path: Optional[Path], source_type: str, url: str
    ) -> None:
        """Delete the downloaded source video + audio sidecar after a task.

        Generated clips and the tiny transcript-cache JSON are left in place.
        For YouTube sources, ``cleanup_downloaded_files`` removes downloaded
        media and partials but retains word timings. For uploads we remove the audio sidecar
        generated for transcription — the original upload is owned elsewhere.
        """
        if not video_path:
            return

        if source_type == "youtube":
            try:
                video_id = extract_video_id(url)
                if video_id:
                    cleanup_downloaded_files(video_id, source_path=video_path)
                    logger.info("Cleaned up YouTube source artifacts for %s", video_id)
                    return
            except Exception as e:
                logger.warning("Failed YouTube source cleanup: %s", e)

        # Fallback / upload path: at least drop the transcription audio sidecar
        # (e.g. "<stem>.transcription.mp3") next to the source file.
        try:
            sidecar = video_path.with_name(f"{video_path.stem}.transcription.mp3")
            if sidecar.exists():
                sidecar.unlink()
                logger.info("Removed transcription audio sidecar: %s", sidecar.name)
        except Exception as e:
            logger.warning("Failed to remove transcription sidecar: %s", e)

    async def create_task_with_source(
        self,
        user_id: str,
        url: str,
        title: Optional[str] = None,
        font_family: Optional[str] = None,
        font_size: Optional[int] = None,
        font_color: Optional[str] = None,
        caption_template: str = "default",
        include_broll: bool = False,
        processing_mode: str = "fast",
    ) -> str:
        """
        Create a new task with associated source.
        Returns the task ID.
        """
        # Validate user exists
        if not await self.task_repo.user_exists(self.db, user_id):
            raise ValueError(f"User {user_id} not found")

        # Determine source type
        source_type = self.video_service.determine_source_type(url)

        # Get or generate title
        if not title:
            if source_type == "youtube":
                title = await self.video_service.get_video_title(url)
            else:
                title = "Uploaded Video"

        # Create source
        source_id = await self.source_repo.create_source(
            self.db, source_type=source_type, title=title, url=url
        )

        # Create task
        task_id = await self.task_repo.create_task(
            self.db,
            user_id=user_id,
            source_id=source_id,
            status="queued",  # Changed from "processing" to "queued"
            font_family=font_family,
            font_size=font_size,
            font_color=font_color,
            caption_template=caption_template,
            include_broll=include_broll,
            processing_mode=processing_mode,
        )

        logger.info(f"Created task {task_id} for user {user_id}")
        return task_id

    def _processing_transaction(self, task_id: str):
        return task_edit_transaction(self.db, task_id, processing=True)

    def _task_run_guard(self, task_id: str):
        return task_run_guard(self.db, task_id)

    @exclusive_task_run
    async def process_task(
        self,
        task_id: str,
        url: str,
        source_type: str,
        user_id: Optional[str] = None,
        font_family: Optional[str] = None,
        font_size: Optional[int] = None,
        font_color: Optional[str] = None,
        caption_template: str = "default",
        processing_mode: str = "fast",
        output_format: str = "vertical",
        add_subtitles: bool = True,
        progress_callback: Optional[Callable] = None,
        should_cancel: Optional[Callable] = None,
        clip_ready_callback: Optional[Callable] = None,
        cleanup_settings: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process a task: download video, analyze, create clips.
        Returns processing results.
        """
        # Tracked so the downloaded source video can be cleaned up on both the
        # success and error paths; multi-GB YouTube downloads otherwise pile up
        # in temp/ until the disk fills and later tasks fail.
        video_path: Optional[Path] = None
        pending_clip_path: Optional[Path] = None
        clips_output_dir = Path(self.config.temp_dir) / "clips"

        async def check_cancelled():
            if should_cancel and await should_cancel():
                raise TaskCancelled()

        try:
            await check_cancelled()
            logger.info(f"Starting processing for task {task_id}")
            started_at = datetime.utcnow()
            stage_timings: Dict[str, float] = {}
            source_metadata = await self.task_repo.load_source_metadata(self.db, task_id)
            ops = source_metadata.get("ops") if isinstance(source_metadata.get("ops"), dict) else {}
            ops_action = ops.get("action")
            generate_more_count = int(ops.get("count") or 0) if ops_action == "generate_more" else 0
            generate_start_times = (
                ops.get("start_times") if ops_action == "generate_more" else None
            )
            generate_mode = (
                str(ops.get("mode") or "top") if ops_action == "generate_more" else "top"
            )
            replace_existing = bool(ops.get("replace_existing")) if ops_action == "generate_more" else False
            # One-shot ops flag — clear so a later resume doesn't re-fire.
            # JSONB merge keeps omitted keys, so explicitly blank `ops`.
            if ops_action:
                source_metadata["ops"] = {}
                await self.task_repo.save_source_metadata(
                    self.db, task_id, source_metadata
                )

            if ops_action == "generate_more":
                return await self._generate_more_clips(
                    task_id=task_id,
                    url=url,
                    source_type=source_type,
                    processing_mode=processing_mode,
                    count=max(1, generate_more_count or 5),
                    mode=generate_mode,
                    start_times=generate_start_times if isinstance(generate_start_times, list) else None,
                    font_family=font_family,
                    font_size=font_size,
                    font_color=font_color,
                    caption_template=caption_template,
                    output_format=output_format,
                    add_subtitles=add_subtitles,
                    cleanup_settings=cleanup_settings,
                    progress_callback=progress_callback,
                    should_cancel=should_cancel,
                    clip_ready_callback=clip_ready_callback,
                    source_metadata=source_metadata,
                    replace_existing=replace_existing,
                )

            if source_metadata.get("job_kind") == "transcribe":
                return await self._process_transcribe_only(
                    task_id=task_id,
                    url=url,
                    source_type=source_type,
                    user_id=user_id,
                    processing_mode=processing_mode,
                    source_metadata=source_metadata,
                    progress_callback=progress_callback,
                    should_cancel=should_cancel,
                )

            try:
                clip_prefs = parse_clip_selection_prefs(source_metadata)
            except ValueError:
                clip_prefs = ClipSelectionPrefs()
            cache_key = self._build_cache_key(
                url, source_type, processing_mode, prefs=clip_prefs
            )

            cache_entry = await self.cache_repo.get_cache(self.db, cache_key)
            cached_transcript = (
                cache_entry.get("transcript_text") if cache_entry else None
            )
            cached_analysis_json = (
                cache_entry.get("analysis_json") if cache_entry else None
            )

            # Fase 3 ops: reanalyze with STT reuse vs from scratch.
            if ops_action == "reanalyze_reuse":
                cached_analysis_json = None
                if not cached_transcript:
                    url_cache = await self.cache_repo.find_latest_by_source_url(
                        self.db, url, source_type
                    )
                    if url_cache:
                        cached_transcript = url_cache.get("transcript_text")
                        if not cache_entry and url_cache.get("video_path"):
                            cache_entry = url_cache
            elif ops_action == "reanalyze_scratch":
                cached_transcript = None
                cached_analysis_json = None

            cache_hit = bool(cached_transcript and cached_analysis_json)

            await self.task_repo.update_task_runtime_metadata(
                self.db,
                task_id,
                started_at=started_at,
                cache_hit=cache_hit,
            )

            # Update status to processing
            started = await self.task_repo.update_task_status(
                self.db, task_id, "processing", progress=0,
                progress_message="Starting...",
                expected_statuses=["queued", "processing"],
            )
            if started is False:
                raise TaskCancelled()

            # Progress callback wrapper — download/analysis gaps can exceed the
            # idle-in-transaction timeout if a prior SELECT left a txn open.
            async def update_progress(
                progress: int, message: str, status: str = "processing"
            ):
                await check_cancelled()

                async def _write_status():
                    return await self.task_repo.update_task_status(
                        self.db,
                        task_id,
                        status,
                        progress=progress,
                        progress_message=message,
                        expected_statuses=["processing"],
                    )

                changed = await self._db_after_render_gap(_write_status)
                if changed is False:
                    raise TaskCancelled()
                if progress_callback:
                    await progress_callback(progress, message, status)

            # Process video with progress updates
            max_video_duration = self.config.max_video_duration
            if source_type == "youtube" and user_id:
                billing = await BillingService(self.db, self.config).get_usage_summary(
                    user_id
                )
                max_video_duration = self.config.max_youtube_video_duration_for_plan(
                    billing.get("plan"), billing.get("subscription_status")
                )
                await commit_unless_editing(self.db)

            # Release before multi-minute download/STT/analysis pipeline.
            await commit_unless_editing(self.db)
            pipeline_start = perf_counter()
            result = await self.video_service.process_video_complete(
                url=url,
                source_type=source_type,
                task_id=task_id,
                font_family=font_family,
                font_size=font_size,
                font_color=font_color,
                caption_template=caption_template,
                processing_mode=processing_mode,
                output_format=output_format,
                add_subtitles=add_subtitles,
                max_video_duration=max_video_duration,
                cached_transcript=cached_transcript,
                cached_analysis_json=cached_analysis_json,
                progress_callback=update_progress,
                should_cancel=should_cancel,
                clip_prefs=clip_prefs,
            )
            stage_timings["pipeline_seconds"] = round(
                perf_counter() - pipeline_start, 3
            )

            normalized_cleanup_settings = normalize_clip_cleanup_settings(
                **(cleanup_settings or {})
            )

            # Render clips incrementally: render, save, notify one at a time
            segments_to_render = result.get("segments_to_render", [])
            if not segments_to_render:
                await self._db_after_render_gap(
                    self.cache_repo.upsert_cache,
                    self.db,
                    cache_key=cache_key,
                    source_url=url,
                    source_type=source_type,
                    video_path=result.get("video_path"),
                    transcript_text=result.get("transcript"),
                    analysis_json=None,
                )
                raise ValueError(
                    "No usable clip segments were selected for this video."
                )

            await self._db_after_render_gap(
                self.cache_repo.upsert_cache,
                self.db,
                cache_key=cache_key,
                source_url=url,
                source_type=source_type,
                video_path=result.get("video_path"),
                transcript_text=result.get("transcript"),
                analysis_json=result.get("analysis_json"),
            )

            # Persist excellence telemetry on the task for UI / calibration.
            excellent_total = int(
                result.get("excellent_clip_count") or len(segments_to_render)
            )
            review_before_render = bool(source_metadata.get("review_before_render"))
            excellence_meta = {
                "excellent_clip_count": excellent_total,
                "rendered_clip_count": 0 if review_before_render else int(
                    result.get("rendered_clip_count") or len(segments_to_render)
                ),
                "render_ceiling": result.get("render_ceiling"),
                "drop_log": (result.get("drop_log") or [])[:80],
            }
            # ETA for the full excellent pool (review UI).
            excellent_pool = result.get("excellent_segments") or segments_to_render
            eta_total = sum(
                estimate_render_seconds(segment_duration_seconds(seg))
                for seg in excellent_pool
            )
            excellence_meta["eta_seconds_total"] = eta_total
            await self._db_after_render_gap(
                self.task_repo.save_source_metadata,
                self.db,
                task_id,
                {**source_metadata, **excellence_meta},
            )

            # Fase 4: pause after analysis so the user picks what to render.
            if review_before_render:
                await check_cancelled()
                review_message = (
                    f"{excellent_total} excelentes — revise e escolha quais gerar"
                )

                async def _pause_review():
                    return await self.task_repo.update_task_status(
                        self.db,
                        task_id,
                        "awaiting_review",
                        progress=70,
                        progress_message=review_message,
                        expected_statuses=["processing"],
                    )

                paused = await self._db_after_render_gap(_pause_review)
                if paused is False:
                    raise TaskCancelled()
                if progress_callback:
                    await progress_callback(70, review_message, "awaiting_review")
                await self._db_after_render_gap(
                    self.task_repo.update_task_runtime_metadata,
                    self.db,
                    task_id,
                    stage_timings_json=json.dumps(stage_timings),
                    error_code="",
                )
                return {
                    "task_id": task_id,
                    "clips_count": 0,
                    "awaiting_review": True,
                    "excellent_clip_count": excellent_total,
                    "eta_seconds_total": eta_total,
                    "summary": result.get("summary"),
                    "key_topics": result.get("key_topics"),
                }

            video_path = Path(result["video_path"])
            total_clips = len(segments_to_render)
            clips_output_dir = Path(self.config.temp_dir) / "clips"
            clips_output_dir.mkdir(parents=True, exist_ok=True)

            # Retries and regenerations should replace earlier clip rows instead of
            # accumulating duplicates for the same task.
            async with self._processing_transaction(task_id):
                await check_cancelled()
                await self.clip_repo.delete_clips_by_task(self.db, task_id)
                await self.task_repo.update_task_clips(self.db, task_id, [])

            render_start = perf_counter()
            clip_ids = await self._render_segments_concurrent(
                task_id=task_id,
                segments=segments_to_render,
                video_path=video_path,
                clips_output_dir=clips_output_dir,
                font_family=font_family,
                font_size=font_size,
                font_color=font_color,
                caption_template=caption_template,
                output_format=output_format,
                add_subtitles=add_subtitles,
                cleanup_settings=normalized_cleanup_settings,
                base_order=0,
                existing_clip_ids=[],
                progress_base=70,
                progress_span=25,
                check_cancelled=check_cancelled,
                update_progress=update_progress,
                clip_ready_callback=clip_ready_callback,
            )
            pending_clip_path = None

            stage_timings["render_seconds"] = round(
                perf_counter() - render_start, 3
            )

            if not clip_ids:
                raise ValueError("No clips could be rendered. Please try again.")

            # A cancellation committed after the final render must win over completion.
            await check_cancelled()
            rendered_total = int(result.get("rendered_clip_count") or total_clips)
            if excellent_total > rendered_total:
                done_message = (
                    f"Pronto! {rendered_total} de {excellent_total} excelentes"
                )
            else:
                done_message = f"Pronto! {rendered_total} corte(s) excelente(s)"

            async def _mark_completed():
                return await self.task_repo.update_task_status(
                    self.db,
                    task_id,
                    "completed",
                    progress=100,
                    progress_message=done_message,
                    expected_statuses=["processing"],
                )

            completed = await self._db_after_render_gap(_mark_completed)
            if completed is False:
                raise TaskCancelled()

            if progress_callback:
                await progress_callback(100, done_message, "completed")

            await self._db_after_render_gap(
                self.task_repo.update_task_runtime_metadata,
                self.db,
                task_id,
                completed_at=datetime.utcnow(),
                stage_timings_json=json.dumps(stage_timings),
                error_code="",
            )
            await self._send_completion_notification_if_needed(
                task_id=task_id,
                clips_count=len(clip_ids),
            )

            logger.info(
                f"Task {task_id} completed successfully with {len(clip_ids)} clips"
            )

            self._cleanup_source_video(video_path, source_type, url)

            return {
                "task_id": task_id,
                "clips_count": len(clip_ids),
                "segments": result["segments"],
                "summary": result.get("summary"),
                "key_topics": result.get("key_topics"),
            }

        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}")
            # Clear failed writes before recording the terminal error state.
            await self.db.rollback()
            if pending_clip_path and pending_clip_path != video_path:
                try:
                    if pending_clip_path.resolve().is_relative_to(clips_output_dir.resolve()):
                        pending_clip_path.unlink(missing_ok=True)
                        pending_clip_path.with_suffix(".source_map.json").unlink(missing_ok=True)
                except OSError:
                    logger.warning("Could not remove unfinished clip %s", pending_clip_path)
            self._cleanup_source_video(video_path, source_type, url)
            if str(e) == "Task cancelled":
                await self.task_repo.update_task_status(
                    self.db,
                    task_id,
                    "cancelled",
                    progress=0,
                    progress_message="Cancelled by user",
                    expected_statuses=["queued", "processing", "awaiting_review"],
                )
                raise
            failed = await self.task_repo.update_task_status(
                self.db, task_id, "error", progress=0, progress_message=str(e),
                expected_statuses=["queued", "processing", "awaiting_review"],
            )
            if failed is False:
                raise
            error_code = "task_error"
            message = str(e).lower()
            if "download" in message or "youtube" in message:
                error_code = "download_error"
            elif "analysis" in message:
                error_code = "analysis_error"
            elif "transcript" in message:
                error_code = "transcription_error"
            elif "cancelled" in message:
                error_code = "cancelled"

            await self.task_repo.update_task_runtime_metadata(
                self.db,
                task_id,
                completed_at=datetime.utcnow(),
                error_code=error_code,
            )
            if progress_callback:
                try:
                    await progress_callback(0, str(e), "error")
                except Exception:
                    logger.exception("Could not publish task failure for %s", task_id)
            raise

    async def _send_completion_notification_if_needed(
        self, *, task_id: str, clips_count: int
    ) -> None:
        context = await self.task_repo.get_task_notification_context(self.db, task_id)
        if not context:
            logger.warning("Task %s missing notification context; skipping email", task_id)
            return

        if not context.get("notify_on_completion"):
            return

        if context.get("completion_notification_sent_at"):
            logger.info(
                "Completion notification already sent for task %s; skipping", task_id
            )
            return

        user_email = context.get("user_email")
        if not user_email:
            logger.warning(
                "Task %s has notify_on_completion enabled but user email is missing",
                task_id,
            )
            return

        email_service = TaskCompletionEmailService(self.config)
        if not email_service.is_configured:
            logger.warning(
                "Skipping completion notification for task %s because Amazon SES is not configured",
                task_id,
            )
            return

        try:
            await email_service.send_task_completed_email(
                recipient=TaskCompletionRecipient(
                    email=user_email,
                    name=context.get("user_name"),
                    first_name=context.get("user_first_name"),
                ),
                task_id=task_id,
                source_title=context.get("source_title"),
                clips_count=clips_count,
            )
            stamped = await self.task_repo.mark_completion_notification_sent(
                self.db, task_id
            )
            if not stamped:
                logger.info(
                    "Completion notification stamp already existed for task %s",
                    task_id,
                )
        except Exception:
            logger.exception(
                "Failed to send completion notification for task %s",
                task_id,
            )

    async def _generate_more_clips(
        self,
        *,
        task_id: str,
        url: str,
        source_type: str,
        processing_mode: str,
        count: int,
        mode: str = "top",
        start_times: Optional[list] = None,
        font_family: Optional[str],
        font_size: Optional[int],
        font_color: Optional[str],
        caption_template: str,
        output_format: str,
        add_subtitles: bool,
        cleanup_settings: Optional[Dict[str, Any]],
        progress_callback: Optional[Callable],
        should_cancel: Optional[Callable],
        clip_ready_callback: Optional[Callable],
        source_metadata: Dict[str, Any],
        replace_existing: bool = False,
    ) -> Dict[str, Any]:
        """Render excellent clips from analysis cache (next N, selected, or all)."""

        async def check_cancelled():
            if should_cancel and await should_cancel():
                raise TaskCancelled()

        async def update_progress(progress: int, message: str, status: str = "processing"):
            await check_cancelled()
            # Include queued: enqueue leaves the task queued until we claim it.
            # A prior bug raised TaskCancelled when the first progress update
            # rejected status=queued (render-selected failed in <100ms).

            async def _write_status():
                return await self.task_repo.update_task_status(
                    self.db,
                    task_id,
                    status,
                    progress=progress,
                    progress_message=message,
                    expected_statuses=["queued", "processing", "awaiting_review"],
                )

            changed = await self._db_after_render_gap(_write_status)
            if changed is False:
                raise TaskCancelled()
            if progress_callback:
                await progress_callback(progress, message, status)

        try:
            clip_prefs = parse_clip_selection_prefs(source_metadata)
        except ValueError:
            clip_prefs = ClipSelectionPrefs()

        cache_key = self._build_cache_key(
            url, source_type, processing_mode, prefs=clip_prefs
        )
        cache_entry = await self.cache_repo.get_cache(self.db, cache_key)
        if not cache_entry or not cache_entry.get("analysis_json"):
            url_cache = await self.cache_repo.find_latest_by_source_url(
                self.db, url, source_type
            )
            cache_entry = url_cache or cache_entry

        if not cache_entry or not cache_entry.get("analysis_json"):
            raise ValueError(
                "Nenhuma análise em cache para gerar mais cortes. Use Reanalisar primeiro."
            )
        if not cache_entry.get("video_path"):
            raise ValueError(
                "Vídeo em cache indisponível. Use Reanalisar (reuse) para regenerar."
            )

        try:
            analysis = json.loads(cache_entry["analysis_json"])
        except json.JSONDecodeError as exc:
            raise ValueError("Cache de análise inválido") from exc

        excellent = list(
            analysis.get("excellent_segments")
            or analysis.get("most_relevant_segments")
            or []
        )
        if not excellent:
            raise ValueError("Não há cortes excelentes restantes neste vídeo.")

        existing = await self.clip_repo.get_clips_by_task(self.db, task_id)
        if replace_existing:
            async with self._processing_transaction(task_id):
                await check_cancelled()
                await self.clip_repo.delete_clips_by_task(self.db, task_id)
                await self.task_repo.update_task_clips(self.db, task_id, [])
            existing = []

        rendered_starts = {
            str(clip.get("start_time") or "").strip()
            for clip in existing
            if clip.get("start_time")
        }
        skipped_starts = {
            str(s).strip()
            for s in (source_metadata.get("skipped_starts") or [])
            if s
        }
        to_render = select_candidates_for_render(
            excellent,
            mode=mode,
            start_times=start_times,
            count=count,
            rendered_starts=rendered_starts,
            skipped_starts=skipped_starts,
            ceiling=count if mode == "top" else None,
        )
        if not to_render:
            raise ValueError("Nenhum corte selecionado disponível para gerar.")

        remaining_after = [
            seg
            for seg in excellent
            if str(seg.get("start_time") or "").strip() not in rendered_starts
            and str(seg.get("start_time") or "").strip() not in skipped_starts
            and str(seg.get("start_time") or "").strip()
            not in {str(s.get("start_time") or "").strip() for s in to_render}
        ]

        # Claim queued → processing BEFORE any progress helper that used to
        # reject queued and falsely cancel render-selected / generate-more.
        async def _claim_processing():
            return await self.task_repo.update_task_status(
                self.db,
                task_id,
                "processing",
                progress=20,
                progress_message=f"Gerando {len(to_render)} cortes…",
                expected_statuses=["queued", "processing", "awaiting_review"],
            )

        started = await self._db_after_render_gap(_claim_processing)
        if started is False:
            raise TaskCancelled()
        if progress_callback:
            await progress_callback(
                20,
                f"Gerando {len(to_render)} corte(s) selecionado(s)…",
                "processing",
            )

        video_path = Path(cache_entry["video_path"])
        if not video_path.exists():
            raise ValueError(
                "Arquivo de vídeo sumiu do cache. Use Reanalisar para baixar de novo."
            )

        normalized_cleanup = normalize_clip_cleanup_settings(
            **(cleanup_settings or {})
        )
        clips_output_dir = Path(self.config.temp_dir) / "clips"
        clips_output_dir.mkdir(parents=True, exist_ok=True)

        base_order = len(existing)
        # SELECTs above (cache/clips) may still hold an autobegin txn.
        await commit_unless_editing(self.db)
        clip_ids = await self._render_segments_concurrent(
            task_id=task_id,
            segments=to_render,
            video_path=video_path,
            clips_output_dir=clips_output_dir,
            font_family=font_family,
            font_size=font_size,
            font_color=font_color,
            caption_template=caption_template,
            output_format=output_format,
            add_subtitles=add_subtitles,
            cleanup_settings=normalized_cleanup,
            base_order=base_order,
            existing_clip_ids=[str(c["id"]) for c in existing if c.get("id")],
            progress_base=30,
            progress_span=60,
            check_cancelled=check_cancelled,
            update_progress=update_progress,
            clip_ready_callback=clip_ready_callback,
            progress_label="Criando corte",
        )

        new_rendered = len(clip_ids)
        excellent_total = int(
            analysis.get("excellent_clip_count") or len(excellent)
        )
        await self._db_after_render_gap(
            self.task_repo.save_source_metadata,
            self.db,
            task_id,
            {
                **source_metadata,
                "excellent_clip_count": excellent_total,
                "rendered_clip_count": new_rendered,
                "review_before_render": False,
            },
        )

        done_message = (
            f"Pronto! {new_rendered} de {excellent_total} excelentes"
            if excellent_total > new_rendered
            else f"Pronto! {new_rendered} corte(s) excelente(s)"
        )

        async def _mark_completed():
            return await self.task_repo.update_task_status(
                self.db,
                task_id,
                "completed",
                progress=100,
                progress_message=done_message,
                expected_statuses=["processing", "awaiting_review"],
            )

        completed = await self._db_after_render_gap(_mark_completed)
        if completed is False:
            raise TaskCancelled()
        if progress_callback:
            await progress_callback(100, done_message, "completed")

        return {
            "task_id": task_id,
            "clips_count": new_rendered,
            "added": len(to_render),
            "excellent_clip_count": excellent_total,
            "remaining": len(remaining_after),
        }

    async def list_review_candidates(self, task_id: str) -> Dict[str, Any]:
        """Excellent pool for review UI (score, title, preview, ETA)."""
        task = await self.task_repo.get_task_by_id(self.db, task_id)
        if not task:
            raise ValueError("Task not found")
        metadata = await self.task_repo.load_source_metadata(self.db, task_id)
        try:
            clip_prefs = parse_clip_selection_prefs(metadata)
        except ValueError:
            clip_prefs = ClipSelectionPrefs()
        url = task.get("source_url") or metadata.get("url") or ""
        source_type = task.get("source_type") or metadata.get("source_type") or "youtube"
        processing_mode = task.get("processing_mode") or self.config.default_processing_mode
        cache_key = self._build_cache_key(url, source_type, processing_mode, prefs=clip_prefs)
        cache_entry = await self.cache_repo.get_cache(self.db, cache_key)
        if not cache_entry or not cache_entry.get("analysis_json"):
            cache_entry = await self.cache_repo.find_latest_by_source_url(
                self.db, url, source_type
            )
        if not cache_entry or not cache_entry.get("analysis_json"):
            raise ValueError("Nenhuma análise em cache para revisar.")

        analysis = json.loads(cache_entry["analysis_json"])
        excellent = list(
            analysis.get("excellent_segments")
            or analysis.get("most_relevant_segments")
            or []
        )
        clips = await self.clip_repo.get_clips_by_task(self.db, task_id)
        rendered_starts = [str(c.get("start_time") or "") for c in clips]
        skipped_starts = list(metadata.get("skipped_starts") or [])
        candidates = build_candidate_rows(
            excellent,
            rendered_starts=rendered_starts,
            skipped_starts=skipped_starts,
        )
        pending = [c for c in candidates if not c["rendered"] and not c["skipped"]]
        eta_pending = sum(int(c["eta_seconds"]) for c in pending)
        eta_selected_hint = sum(int(c["eta_seconds"]) for c in pending[:15])
        return {
            "task_id": task_id,
            "status": task.get("status"),
            "candidates": candidates,
            "excellent_clip_count": int(
                analysis.get("excellent_clip_count") or len(excellent)
            ),
            "rendered_count": len([c for c in candidates if c["rendered"]]),
            "skipped_count": len([c for c in candidates if c["skipped"]]),
            "pending_count": len(pending),
            "eta_seconds_pending": eta_pending,
            "eta_seconds_top15": eta_selected_hint,
            "render_ceiling": metadata.get("render_ceiling") or 15,
            "review_before_render": bool(metadata.get("review_before_render")),
        }

    async def skip_review_candidates(
        self,
        task_id: str,
        start_times: list[str],
        *,
        permanent: bool = False,
    ) -> Dict[str, Any]:
        """Mark candidates skipped; optional permanent → EVITAR in brief."""
        metadata = await self.task_repo.load_source_metadata(self.db, task_id)
        skipped = {
            str(s).strip()
            for s in (metadata.get("skipped_starts") or [])
            if s
        }
        wanted = {str(s).strip() for s in start_times if s}
        skipped |= wanted

        phrases: list[str] = []
        if permanent and wanted:
            try:
                candidates = await self.list_review_candidates(task_id)
                for row in candidates.get("candidates") or []:
                    if row.get("start_time") in wanted:
                        title = (row.get("hook_title") or "").strip()
                        preview = (row.get("text_preview") or "").strip()
                        phrases.append(title or preview[:80])
            except ValueError:
                phrases = []
            if phrases:
                try:
                    prefs = parse_clip_selection_prefs(metadata)
                except ValueError:
                    prefs = ClipSelectionPrefs()
                new_brief = append_evitar_phrases(prefs.brief, phrases)
                metadata["clip_brief"] = new_brief

        metadata["skipped_starts"] = sorted(skipped)
        await self.task_repo.save_source_metadata(self.db, task_id, metadata)
        return {
            "skipped_starts": metadata["skipped_starts"],
            "permanent": permanent,
            "brief_updated": bool(permanent and phrases),
        }

    async def _process_transcribe_only(
        self,
        *,
        task_id: str,
        url: str,
        source_type: str,
        user_id: Optional[str],
        processing_mode: str,
        source_metadata: Dict[str, Any],
        progress_callback: Optional[Callable] = None,
        should_cancel: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Download (if needed) + STT only; no LLM analysis or clip render."""
        video_path: Optional[Path] = None
        started_at = datetime.utcnow()
        stage_timings: Dict[str, float] = {}

        async def check_cancelled():
            if should_cancel and await should_cancel():
                raise TaskCancelled()

        language_raw = source_metadata.get("language")
        language = (
            str(language_raw).strip().lower().replace("-", "_")
            if isinstance(language_raw, str) and language_raw.strip()
            else "pt"
        )
        runtime_config = self.config
        provider = runtime_config.transcription_provider
        speech_model = self._transcribe_speech_model(processing_mode, provider)
        cache_key = self._build_transcribe_cache_key(
            url,
            source_type,
            provider=provider,
            speech_model=speech_model,
            language=language,
        )

        try:
            await check_cancelled()
            cache_entry = await self.cache_repo.get_cache(self.db, cache_key)
            cached_payload = unwrap_transcript_payload_from_cache(
                cache_entry.get("analysis_json") if cache_entry else None
            )
            cache_hit = bool(cached_payload and cached_payload.get("text"))

            await self.task_repo.update_task_runtime_metadata(
                self.db,
                task_id,
                started_at=started_at,
                cache_hit=cache_hit,
            )

            started = await self.task_repo.update_task_status(
                self.db,
                task_id,
                "processing",
                progress=0,
                progress_message="Starting...",
                expected_statuses=["queued", "processing"],
            )
            if started is False:
                raise TaskCancelled()

            async def update_progress(
                progress: int, message: str, status: str = "processing"
            ):
                await check_cancelled()

                async def _write_status():
                    return await self.task_repo.update_task_status(
                        self.db,
                        task_id,
                        status,
                        progress=progress,
                        progress_message=message,
                        expected_statuses=["processing"],
                    )

                changed = await self._db_after_render_gap(_write_status)
                if changed is False:
                    raise TaskCancelled()
                if progress_callback:
                    await progress_callback(progress, message, status)

            await commit_unless_editing(self.db)

            if cache_hit and cached_payload:
                await update_progress(90, "Reusing cached transcript…")
                payload = cached_payload
                transcript_text = build_plain_text_from_transcript_data(payload)
            else:
                max_video_duration = self.config.max_video_duration
                if source_type == "youtube" and user_id:
                    billing = await BillingService(self.db, self.config).get_usage_summary(
                        user_id
                    )
                    max_video_duration = self.config.max_youtube_video_duration_for_plan(
                        billing.get("plan"), billing.get("subscription_status")
                    )
                    await commit_unless_editing(self.db)

                await update_progress(10, "Baixando vídeo…")
                pipeline_start = perf_counter()
                if source_type == "youtube":
                    video_path = await self.video_service.download_video(
                        url, task_id=task_id
                    )
                    if not video_path:
                        raise ValueError("Failed to download video")
                else:
                    video_path = self.video_service.resolve_local_video_path(url)
                    if not video_path.exists():
                        raise ValueError("Video file not found")

                file_duration = self.video_service._get_file_duration(video_path)
                if file_duration and file_duration > max_video_duration:
                    mins = max_video_duration // 60
                    raise ValueError(
                        f"Video is too long ({int(file_duration) // 60} min). "
                        f"Maximum allowed duration is {mins} minutes."
                    )

                stage_timings["download_seconds"] = round(
                    perf_counter() - pipeline_start, 3
                )
                await check_cancelled()
                await update_progress(30, "Transcrevendo…")
                stt_start = perf_counter()
                transcript_text = await self.video_service.generate_transcript(
                    video_path,
                    processing_mode=processing_mode,
                    source_url=url if source_type == "youtube" else None,
                    language=language,
                )
                stage_timings["transcript_seconds"] = round(
                    perf_counter() - stt_start, 3
                )

                disk_payload = normalize_transcript_payload(
                    load_cached_transcript_data(video_path)
                )
                if disk_payload:
                    payload = disk_payload
                    if not payload.get("text"):
                        payload["text"] = transcript_text
                else:
                    payload = {
                        "version": 1,
                        "text": transcript_text,
                        "words": [],
                        "utterances": [],
                        "language_code": language,
                    }

                await self._db_after_render_gap(
                    self.cache_repo.upsert_cache,
                    self.db,
                    cache_key=cache_key,
                    source_url=url,
                    source_type=source_type,
                    video_path=str(video_path) if video_path else None,
                    transcript_text=transcript_text,
                    analysis_json=wrap_transcript_payload_for_cache(payload),
                )

            await update_progress(95, "Salvando transcript…")
            preview = (payload.get("text") or transcript_text or "")[:2000]
            meta_update = {
                **source_metadata,
                "job_kind": "transcribe",
                "language": language,
                "transcript_cache_key": cache_key,
                "transcript_payload": payload,
                "transcript_preview": preview,
                "transcript_char_count": len(
                    payload.get("text") or transcript_text or ""
                ),
            }
            await self._db_after_render_gap(
                self.task_repo.save_source_metadata,
                self.db,
                task_id,
                meta_update,
            )

            done_message = "Transcrição concluída"
            async def _mark_completed():
                return await self.task_repo.update_task_status(
                    self.db,
                    task_id,
                    "completed",
                    progress=100,
                    progress_message=done_message,
                    expected_statuses=["processing"],
                )

            completed = await self._db_after_render_gap(_mark_completed)
            if completed is False:
                raise TaskCancelled()
            if progress_callback:
                await progress_callback(100, done_message, "completed")

            await self._db_after_render_gap(
                self.task_repo.update_task_runtime_metadata,
                self.db,
                task_id,
                completed_at=datetime.utcnow(),
                stage_timings_json=json.dumps(stage_timings),
                error_code="",
            )

            if video_path:
                self._cleanup_source_video(video_path, source_type, url)

            logger.info(
                "Transcribe task %s completed (%d chars)",
                task_id,
                len(transcript_text or ""),
            )
            return {
                "task_id": task_id,
                "clips_count": 0,
                "transcript_char_count": len(transcript_text or ""),
                "job_kind": "transcribe",
            }

        except Exception:
            if video_path:
                self._cleanup_source_video(video_path, source_type, url)
            raise

    async def get_transcript_export(
        self, task_id: str, fmt: str = "txt"
    ) -> tuple[str, str, str]:
        """Return (body, media_type, filename) for a transcribe task export."""
        metadata = await self.task_repo.load_source_metadata(self.db, task_id)
        if metadata.get("job_kind") != "transcribe":
            raise ValueError("Task is not a transcription job")

        payload = normalize_transcript_payload(metadata.get("transcript_payload"))
        if not payload:
            cache_key = metadata.get("transcript_cache_key")
            if isinstance(cache_key, str) and cache_key:
                cache_entry = await self.cache_repo.get_cache(self.db, cache_key)
                if cache_entry:
                    payload = unwrap_transcript_payload_from_cache(
                        cache_entry.get("analysis_json")
                    )
                    if not payload and cache_entry.get("transcript_text"):
                        payload = {
                            "text": cache_entry["transcript_text"],
                            "words": [],
                            "utterances": [],
                            "language_code": metadata.get("language"),
                        }
                    video_path_raw = cache_entry.get("video_path")
                    if not payload and isinstance(video_path_raw, str) and video_path_raw:
                        disk = normalize_transcript_payload(
                            load_cached_transcript_data(Path(video_path_raw))
                        )
                        if disk:
                            payload = disk

        if not payload:
            raise ValueError("Transcript not available yet")

        fmt_normalized = (fmt or "txt").strip().lower()
        if fmt_normalized not in {"txt", "srt", "json"}:
            raise ValueError("format must be txt, srt, or json")

        task = await self.task_repo.get_task_by_id(self.db, task_id)
        raw_title = (task or {}).get("source_title") or "transcript"
        safe_title = re.sub(r"[^\w\-]+", "_", str(raw_title))[:60] or "transcript"

        if fmt_normalized == "json":
            body = json.dumps(payload, ensure_ascii=False, indent=2)
            return body, "application/json", f"{safe_title}.json"
        if fmt_normalized == "srt":
            body = build_srt_from_transcript_data(payload)
            if not body.strip():
                raise ValueError("No timed cues available for SRT export")
            return body, "application/x-subrip", f"{safe_title}.srt"
        body = build_plain_text_from_transcript_data(payload)
        return body, "text/plain; charset=utf-8", f"{safe_title}.txt"

    async def get_task_with_clips(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task details with all clips."""
        task = await self.task_repo.get_task_by_id(self.db, task_id)

        if not task:
            return None

        if self._is_stale_queued_task(task) and await self._queued_job_is_missing(task):
            timeout_seconds = self.config.queued_task_timeout_seconds
            logger.warning(
                f"Task {task_id} stuck in queued status for over {timeout_seconds}s; marking as error"
            )
            await self.task_repo.update_task_status(
                self.db,
                task_id,
                "error",
                progress=0,
                progress_message=(
                    "The queued job is no longer available. Please resume this generation."
                ),
                expected_statuses=["queued"],
            )
            task = await self.task_repo.get_task_by_id(self.db, task_id)
            if not task:
                return None

        if self._is_stale_processing_task(task):
            timeout_seconds = self.config.processing_task_timeout_seconds
            logger.warning(
                f"Task {task_id} stuck in processing status for over {timeout_seconds}s; marking as error"
            )
            await self.task_repo.update_task_status(
                self.db,
                task_id,
                "error",
                progress=0,
                progress_message=(
                    "Task stalled during processing (worker likely stopped or timed out). "
                    "Check the worker logs (docker-compose logs -f worker) and try again."
                ),
                expected_statuses=["processing"],
            )
            task = await self.task_repo.get_task_by_id(self.db, task_id)
            if not task:
                return None

        # Get clips
        clips = await self.clip_repo.get_clips_by_task(self.db, task_id)
        task["clips"] = [
            {
                **{key: value for key, value in clip.items() if key != "file_path"},
                "caption_settings": load_clip_caption_settings(Path(clip["file_path"])) if clip.get("file_path") else None,
            }
            for clip in clips
        ]
        task["clips_count"] = len(clips)
        task.update(await self._load_task_source_settings(task_id))

        return task

    async def get_user_tasks(
        self, user_id: str, limit: int = 50, *, job_kind: Optional[str] = None
    ) -> list[Dict[str, Any]]:
        """Get tasks for a user (clip Library by default; pass job_kind=transcribe)."""
        return await self.task_repo.get_user_tasks(
            self.db, user_id, limit, job_kind=job_kind
        )

    @serialized_task_edit
    async def delete_task(self, task_id: str) -> None:
        """Delete a task and all its associated clips."""
        # Delete all clips for this task
        await self.clip_repo.delete_clips_by_task(self.db, task_id)

        # Delete the task
        await self.task_repo.delete_task(self.db, task_id)

        logger.info(f"Deleted task {task_id} and all associated clips")


    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Return aggregate processing performance metrics."""
        return await self.task_repo.get_performance_metrics(self.db)
