"""
Utility functions for YouTube-related operations.
Optimized for free yt-dlp downloads with optional Apify fallback.
"""

import asyncio
from datetime import datetime
import logging
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

import requests
import yt_dlp

from .oxylabs_youtube_downloader import download_video_via_oxylabs
from .videoscale_youtube_downloader import download_video_via_videoscale
from .apify_youtube_downloader import ApifyDownloadError, download_video_via_apify
from .config import get_config

logger = logging.getLogger(__name__)

YOUTUBE_METADATA_PROVIDER_YTDLP = "yt_dlp"
YOUTUBE_METADATA_PROVIDER_DATA_API = "youtube_data_api"
YOUTUBE_DOWNLOAD_PROVIDER_YTDLP = "yt_dlp"
YOUTUBE_DOWNLOAD_PROVIDER_APIFY = "apify"
YOUTUBE_DATA_API_URL = "https://www.googleapis.com/youtube/v3/videos"


class YouTubeDownloader:
    """Enhanced YouTube downloader with optimized settings."""

    def __init__(self):
        self.temp_dir = Path(get_config().temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def get_optimal_download_options(
        self,
        video_id: str,
    ) -> Dict[str, Any]:
        """Get optimal yt-dlp options for high-quality downloads."""
        output_path = self.temp_dir / f"{video_id}.%(ext)s"

        opts = {
            "outtmpl": str(output_path),
            # Use best available video/audio to avoid quality caps from container constraints.
            "format": (
                f"bestvideo*[height<={os.environ['YOUTUBE_MAX_HEIGHT']}]+bestaudio/best[height<={os.environ['YOUTUBE_MAX_HEIGHT']}]"
                if os.environ.get("YOUTUBE_MAX_HEIGHT", "").isdigit()
                else "bestvideo*+bestaudio/best"
            ),
            "format_sort": ["res", "fps"],
            "merge_output_format": "mp4",
            "writesubtitles": False,
            "writeautomaticsub": False,
            "noplaylist": True,
            "overwrites": True,
            # Optimized for speed and reliability
            "socket_timeout": 30,
            "retries": 5,  # Increased retries
            "fragment_retries": 5,
            "http_chunk_size": 10485760,  # 10MB chunks
            # Quiet operation - only errors/warnings
            "quiet": True,
            "no_warnings": False,  # Show warnings but not info
            "ignoreerrors": False,
            # Enhanced headers to avoid 403 errors
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            },
            # Metadata extraction
            "extract_flat": False,
            "writeinfojson": False,
            # Additional bypass options
            "nocheckcertificate": True,
            "prefer_insecure": False,
            "age_limit": None,
        }

        return opts


def _cookie_runtime_dir(source: Path) -> Path:
    """Directory for disposable cookie copies next to the durable jar when possible."""
    preferred = source.parent / ".runtime"
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        probe = preferred / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return preferred
    except OSError:
        return Path(tempfile.gettempdir())


def _apply_cookies(ydl_opts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Attach YouTube cookies without letting yt-dlp rewrite the durable jar.

    yt-dlp writes refreshed cookies back to `cookiefile`. On Contabo that often
    replaces a good browser export with a bot-throttled / logged-out session, so
    later jobs fail until cookies are remounted. Always feed yt-dlp a disposable
    copy; keep YTDLP_COOKIES_FILE as the sealed source of truth.
    """
    cookie_file = os.environ.get("YTDLP_COOKIES_FILE")
    if not cookie_file:
        logger.warning(
            "YTDLP_COOKIES_FILE is unset; YouTube may block downloads with bot checks"
        )
        return ydl_opts
    source = Path(cookie_file)
    if not source.is_file():
        logger.warning(
            "YTDLP_COOKIES_FILE=%s is missing; YouTube may block downloads with bot checks",
            cookie_file,
        )
        return ydl_opts

    runtime_dir = _cookie_runtime_dir(source)
    try:
        _cleanup_stale_runtime_cookies(runtime_dir)
        handle = tempfile.NamedTemporaryFile(
            prefix="ytdlp-cookies-",
            suffix=".txt",
            dir=str(runtime_dir),
            delete=False,
        )
        handle.close()
        runtime_path = Path(handle.name)
        shutil.copy2(source, runtime_path)
        # Best-effort: discourage accidental in-place edits of the durable jar
        # if some other path still points at it.
        try:
            source.chmod(0o444)
        except OSError:
            pass
        ydl_opts["cookiefile"] = str(runtime_path)
    except OSError as exc:
        logger.warning(
            "Could not create disposable cookie jar from %s (%s); using source file",
            cookie_file,
            exc,
        )
        ydl_opts["cookiefile"] = cookie_file
    return ydl_opts


def _cleanup_stale_runtime_cookies(runtime_dir: Path, max_age_seconds: int = 3600) -> None:
    try:
        now = time.time()
        for path in runtime_dir.glob("ytdlp-cookies-*.txt"):
            try:
                if now - path.stat().st_mtime > max_age_seconds:
                    path.unlink(missing_ok=True)
            except OSError:
                continue
    except OSError:
        pass


_PROXY_CACHE_PATH = Path(
    os.environ.get("YTDLP_PROXY_CACHE_FILE", "/app/cookies/.ytdlp_proxy")
)
_PROXY_LIST_URL = (
    "https://api.proxyscrape.com/v2/"
    "?request=displayproxies&protocol=http&timeout=5000"
    "&country=all&ssl=all&anonymity=all"
)


def _apply_proxy(
    ydl_opts: Dict[str, Any],
    proxy: Optional[str] = None,
) -> Dict[str, Any]:
    """Attach an explicit proxy, or YTDLP_PROXY when set."""
    resolved = (proxy or os.environ.get("YTDLP_PROXY") or "").strip()
    if resolved:
        ydl_opts["proxy"] = resolved
    return ydl_opts


def _read_cached_proxy() -> Optional[str]:
    try:
        if not _PROXY_CACHE_PATH.is_file():
            return None
        age = time.time() - _PROXY_CACHE_PATH.stat().st_mtime
        if age > 30 * 60:
            return None
        value = _PROXY_CACHE_PATH.read_text(encoding="utf-8").strip()
        return value or None
    except OSError:
        return None


def _write_cached_proxy(proxy: str) -> None:
    try:
        _PROXY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _PROXY_CACHE_PATH.write_text(proxy.strip() + "\n", encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not cache yt-dlp proxy: %s", exc)


def _proxy_tcp_alive(proxy: str, timeout: float = 3.0) -> bool:
    """Cheap TCP connect check so we skip dead public proxies before yt-dlp."""
    try:
        parsed = urlparse(proxy if "://" in proxy else f"http://{proxy}")
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if not host or not port:
            return False
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _fetch_public_http_proxies(limit: int = 20) -> list[str]:
    """Best-effort free HTTP proxies for datacenter IP bot-checks."""
    try:
        response = requests.get(_PROXY_LIST_URL, timeout=(5, 15))
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Proxy list fetch failed: %s", exc)
        return []

    proxies: list[str] = []
    for line in response.text.splitlines():
        host = line.strip()
        if not host or ":" not in host:
            continue
        proxies.append(host if host.startswith("http") else f"http://{host}")
        if len(proxies) >= limit:
            break
    return proxies


def _iter_ytdlp_proxies(
    include_direct: bool = True,
    prefer_proxy: bool = False,
) -> list[Optional[str]]:
    """
    Candidate proxies for yt-dlp.

    Default order: direct → YTDLP_PROXY → cached → public list.
    When prefer_proxy=True (downloads on bot-blocked hosts): cached →
    YTDLP_PROXY → public → direct last.
    """
    ordered: list[Optional[str]] = []
    seen: set[str] = set()

    def _add(proxy: Optional[str]) -> None:
        if proxy is None:
            if include_direct and None not in ordered:
                ordered.append(None)
            return
        normalized = proxy.strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        ordered.append(normalized)

    if not prefer_proxy and include_direct:
        _add(None)
    _add(_read_cached_proxy() if prefer_proxy else os.environ.get("YTDLP_PROXY"))
    _add(os.environ.get("YTDLP_PROXY") if prefer_proxy else _read_cached_proxy())
    if os.environ.get("YTDLP_AUTO_PROXY", "true").lower() in {"1", "true", "yes"}:
        for proxy in _fetch_public_http_proxies(limit=30):
            # Skip obvious corpses before burning yt-dlp retries on them.
            if _proxy_tcp_alive(proxy):
                _add(proxy)
            else:
                logger.debug("Skipping dead public proxy candidate %s", proxy)
    if prefer_proxy and include_direct:
        _add(None)
    return ordered


def _is_youtube_bot_block_error(error: BaseException) -> bool:
    text = str(error).lower()
    return any(
        needle in text
        for needle in (
            "sign in to confirm",
            "not a bot",
            "confirm you're not a bot",
            "confirm you’re not a bot",
        )
    )


def _build_info_options(proxy: Optional[str] = None) -> Dict[str, Any]:
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extractaudio": False,
        "skip_download": True,
        "socket_timeout": 30,
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        },
        "nocheckcertificate": True,
    }
    return _apply_proxy(_apply_cookies(ydl_opts), proxy=proxy)


def _empty_video_info(video_id: Optional[str] = None) -> Dict[str, Any]:
    return {
        "id": video_id,
        "title": None,
        "description": "",
        "duration": None,
        "uploader": None,
        "upload_date": None,
        "view_count": None,
        "like_count": None,
        "thumbnail": None,
        "format_id": None,
        "resolution": None,
        "fps": None,
        "filesize": None,
    }


def _parse_iso8601_duration_to_seconds(value: str) -> int:
    match = re.fullmatch(
        r"P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?",
        value or "",
    )
    if not match:
        raise ValueError(f"Unsupported ISO 8601 duration: {value}")

    days = int(match.group("days") or 0)
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    seconds = int(match.group("seconds") or 0)
    return (((days * 24) + hours) * 60 + minutes) * 60 + seconds


def _pick_best_thumbnail(thumbnails: Optional[Dict[str, Any]]) -> Optional[str]:
    if not thumbnails:
        return None

    for key in ("maxres", "standard", "high", "medium", "default"):
        candidate = thumbnails.get(key)
        if isinstance(candidate, dict) and candidate.get("url"):
            return candidate["url"]

    for candidate in thumbnails.values():
        if isinstance(candidate, dict) and candidate.get("url"):
            return candidate["url"]

    return None


def _normalize_upload_date(published_at: Optional[str]) -> Optional[str]:
    if not published_at:
        return None

    try:
        return (
            datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            .strftime("%Y%m%d")
        )
    except ValueError:
        logger.warning("Could not parse YouTube publishedAt value: %s", published_at)
        return None


def _parse_optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _get_local_video_dimensions(path: Path) -> tuple[int, int]:
    """Return local video width/height using ffprobe."""
    try:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=s=x:p=0",
            str(path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout.strip()
        if not output or "x" not in output:
            return (0, 0)
        width_str, height_str = output.split("x", 1)
        return (int(width_str), int(height_str))
    except Exception:
        return (0, 0)


def _remove_cached_downloads(temp_dir: Path, video_id: str) -> None:
    cached_files = [
        file_path
        for file_path in temp_dir.glob(f"{video_id}.*")
        if file_path.is_file()
        and file_path.suffix.lower() in [".mp4", ".mkv", ".webm", ".mov", ".m4v"]
    ]
    if not cached_files:
        return

    logger.info(
        "Refreshing download for %s (found %s cached file(s))",
        video_id,
        len(cached_files),
    )
    for cached_file in cached_files:
        try:
            cached_file.unlink()
        except Exception as exc:
            logger.warning("Failed to remove stale cache file %s: %s", cached_file, exc)


def sanitize_youtube_url(url: str) -> str:
    """
    Keep only the YouTube URL when users paste a brief after ` // `.
    Example: `https://youtu.be/ID?t=600 // hook first` → URL only.
    """
    if not isinstance(url, str):
        return ""
    cleaned = url.strip()
    if "://" in cleaned and " // " in cleaned:
        cleaned = cleaned.split(" // ", 1)[0].strip()
    return cleaned


def get_youtube_video_id(url: str) -> Optional[str]:
    """
    Extract YouTube video ID from various URL formats.
    Supports standard, short, embed, and mobile URLs.
    """
    if not isinstance(url, str) or not url.strip():
        return None

    url = sanitize_youtube_url(url)

    # Comprehensive regex patterns for different YouTube URL formats
    patterns = [
        r"(?:youtube\.com/(?:.*v=|v/|embed/|shorts/)|youtu\.be/)([A-Za-z0-9_-]{11})",
        r"youtube\.com/watch\?v=([A-Za-z0-9_-]{11})",
        r"youtube\.com/embed/([A-Za-z0-9_-]{11})",
        r"youtube\.com/v/([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"youtube\.com/shorts/([A-Za-z0-9_-]{11})",
        r"m\.youtube\.com/watch\?v=([A-Za-z0-9_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url, re.IGNORECASE)
        if match:
            video_id = match.group(1)
            # Validate video ID length (YouTube IDs are always 11 characters)
            if len(video_id) == 11:
                return video_id

    # Fallback: parse query parameters
    try:
        parsed_url = urlparse(url)
        if "youtube.com" in parsed_url.netloc.lower():
            query = parse_qs(parsed_url.query)
            video_ids = query.get("v")
            if video_ids and len(video_ids[0]) == 11:
                return video_ids[0]
    except Exception as e:
        logger.warning(f"Error parsing URL query parameters: {e}")

    return None


def validate_youtube_url(url: str) -> bool:
    """Validate if URL is a proper YouTube URL."""
    video_id = get_youtube_video_id(url)
    return video_id is not None


def _fetch_video_info_with_ytdlp(url: str) -> Dict[str, Any]:
    url = sanitize_youtube_url(url)
    video_id = get_youtube_video_id(url)
    if not video_id:
        raise ValueError(f"Invalid YouTube URL: {url}")

    last_error: Optional[Exception] = None
    # Cap public-proxy attempts so preflight stays within the API timeout budget.
    # Prefer proxy first on bot-blocked datacenter IPs so preflight
    # does not burn the API timeout on a doomed direct attempt + dead proxies.
    proxy_candidates = _iter_ytdlp_proxies(include_direct=True, prefer_proxy=True)[:8]
    for proxy in proxy_candidates:
        ydl_opts = _build_info_options(proxy=proxy)
        if proxy is not None:
            ydl_opts["socket_timeout"] = 15
            ydl_opts["retries"] = 0
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
            if proxy:
                _write_cached_proxy(proxy)
                logger.info("YouTube metadata via proxy succeeded for %s", video_id)
            return {
                "id": info.get("id"),
                "title": info.get("title"),
                "description": info.get("description", ""),
                "duration": info.get("duration"),
                "uploader": info.get("uploader"),
                "upload_date": info.get("upload_date"),
                "view_count": info.get("view_count"),
                "like_count": info.get("like_count"),
                "thumbnail": info.get("thumbnail"),
                "format_id": info.get("format_id"),
                "resolution": info.get("resolution"),
                "fps": info.get("fps"),
                "filesize": info.get("filesize"),
            }
        except Exception as exc:
            last_error = exc
            if proxy is None and not _is_youtube_bot_block_error(exc):
                raise
            logger.warning(
                "yt-dlp metadata failed for %s via %s: %s",
                video_id,
                proxy or "direct",
                exc,
            )
            continue

    if last_error:
        raise last_error
    raise RuntimeError(f"yt-dlp metadata failed for {video_id}")


def _fetch_video_info_with_youtube_data_api(url: str) -> Dict[str, Any]:
    video_id = get_youtube_video_id(url)
    if not video_id:
        raise ValueError(f"Invalid YouTube URL: {url}")

    config = get_config()
    api_key = config.resolve_youtube_data_api_key()
    if not api_key:
        raise ValueError("Missing YOUTUBE_DATA_API_KEY and GOOGLE_API_KEY")

    response = requests.get(
        YOUTUBE_DATA_API_URL,
        params={
            "part": "snippet,contentDetails,statistics",
            "id": video_id,
            "key": api_key,
            "fields": (
                "items(id,"
                "snippet(title,description,channelTitle,publishedAt,"
                "thumbnails(default(url),medium(url),high(url),standard(url),maxres(url))),"
                "contentDetails(duration),"
                "statistics(viewCount,likeCount))"
            ),
        },
        timeout=(10, 30),
    )
    response.raise_for_status()
    payload = response.json()
    items = payload.get("items") or []
    if not items:
        raise ValueError(f"No YouTube Data API results for video {video_id}")

    item = items[0]
    snippet = item.get("snippet") or {}
    content_details = item.get("contentDetails") or {}
    statistics = item.get("statistics") or {}
    normalized = _empty_video_info(item.get("id") or video_id)
    normalized.update(
        {
            "title": snippet.get("title"),
            "description": snippet.get("description", ""),
            "duration": _parse_iso8601_duration_to_seconds(
                content_details.get("duration", "")
            ),
            "uploader": snippet.get("channelTitle"),
            "upload_date": _normalize_upload_date(snippet.get("publishedAt")),
            "view_count": _parse_optional_int(statistics.get("viewCount")),
            "like_count": _parse_optional_int(statistics.get("likeCount")),
            "thumbnail": _pick_best_thumbnail(snippet.get("thumbnails")),
        }
    )
    return normalized


def fetch_video_info(url: str) -> Optional[Dict[str, Any]]:
    """
    Backward-compatible metadata lookup entrypoint.
    """
    return get_youtube_video_info(url)


async def async_get_youtube_video_info(
    url: str,
    task_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    return await asyncio.to_thread(get_youtube_video_info, url, task_id)


def get_youtube_video_info(
    url: str,
    task_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    del task_id  # Reserved for future provider-specific tracing.

    video_id = get_youtube_video_id(url)
    if not video_id:
        logger.error("Invalid YouTube URL: %s", url)
        return None

    config = get_config()
    primary_provider = config.youtube_metadata_provider
    secondary_provider = (
        YOUTUBE_METADATA_PROVIDER_DATA_API
        if primary_provider == YOUTUBE_METADATA_PROVIDER_YTDLP
        else YOUTUBE_METADATA_PROVIDER_YTDLP
    )
    providers = [primary_provider, secondary_provider]
    last_error: Optional[Exception] = None

    for index, provider in enumerate(providers):
        try:
            if provider == YOUTUBE_METADATA_PROVIDER_DATA_API:
                video_info = _fetch_video_info_with_youtube_data_api(url)
            else:
                video_info = _fetch_video_info_with_ytdlp(url)

            if index == 0:
                logger.info(
                    "Fetched YouTube metadata for %s using primary provider %s",
                    video_id,
                    provider,
                )
            else:
                logger.info(
                    "Fetched YouTube metadata for %s using fallback provider %s",
                    video_id,
                    provider,
                )
            return video_info
        except Exception as exc:
            last_error = exc
            if index == 0:
                logger.warning(
                    "Primary YouTube metadata provider %s failed for %s: %s. Trying %s.",
                    provider,
                    video_id,
                    exc,
                    secondary_provider,
                )
            else:
                logger.warning(
                    "Fallback YouTube metadata provider %s failed for %s: %s",
                    provider,
                    video_id,
                    exc,
                )

    if last_error:
        logger.warning("YouTube video info fetch failed for %s: %s", video_id, last_error)
    return None


def get_youtube_video_title(
    url: str,
) -> Optional[str]:
    """
    Get the title of a YouTube video from a URL.
    Enhanced with better error handling and validation.
    """
    video_info = get_youtube_video_info(url)
    return video_info.get("title") if video_info else None


async def async_get_youtube_video_title(url: str) -> Optional[str]:
    video_info = await async_get_youtube_video_info(url)
    return video_info.get("title") if video_info else None


APIFY_LONG_VIDEO_MAX_QUALITY = "1080"


def _resolve_apify_quality(url: str) -> str:
    """Cap long videos at 1080p so the Apify run finishes before its timeout."""
    config = get_config()
    quality = config.apify_youtube_default_quality
    if int(quality) <= int(APIFY_LONG_VIDEO_MAX_QUALITY):
        return quality

    try:
        video_info = get_youtube_video_info(url)
    except Exception as exc:
        logger.warning("Could not fetch duration for Apify quality cap: %s", exc)
        video_info = None

    duration = (video_info or {}).get("duration") or 0
    if duration > config.apify_long_video_seconds:
        logger.info(
            "Video is %ss long; capping Apify quality at %sp instead of %sp",
            duration,
            APIFY_LONG_VIDEO_MAX_QUALITY,
            quality,
        )
        return APIFY_LONG_VIDEO_MAX_QUALITY
    return quality


def download_youtube_video_with_apify(
    url: str,
    video_id: str,
) -> Path:
    config = get_config()
    downloader = YouTubeDownloader()
    quality = _resolve_apify_quality(url)
    logger.info(
        "Attempting Apify YouTube download for %s with quality %s",
        video_id,
        quality,
    )
    return download_video_via_apify(
        url=url,
        video_id=video_id,
        temp_dir=downloader.temp_dir,
        api_token=config.apify_api_token,
        quality=quality,
    )


def _download_youtube_video_with_ytdlp(
    url: str,
    max_retries: int = 3,
    task_id: Optional[str] = None,
) -> Optional[Path]:
    """
    Download YouTube video with optimized settings and retry logic.
    Returns the path to the downloaded file, or None if download fails.
    """
    url = sanitize_youtube_url(url)
    logger.info(f"Starting YouTube download: {url}")

    video_id = get_youtube_video_id(url)
    if not video_id:
        logger.error(f"Could not extract video ID from URL: {url}")
        return None

    downloader = YouTubeDownloader()
    video_info = get_youtube_video_info(
        url,
        task_id=task_id,
    )
    if not video_info:
        logger.error(f"Could not retrieve video information for: {url}")
        return None

    logger.info(f"Video: '{video_info.get('title')}' ({video_info.get('duration')}s)")

    duration = video_info.get("duration", 0)
    if duration > 3600:
        logger.warning(f"Video duration ({duration}s) exceeds recommended limit")

    last_error: Optional[str] = None

    # Prefer proxies first on Contabo-like datacenter IPs; free proxies die often
    # so keep a wider candidate list than the default max_retries.
    proxy_candidates = _iter_ytdlp_proxies(include_direct=True, prefer_proxy=True)[:10]
    download_attempts = max(max_retries, min(8, len(proxy_candidates)))
    for attempt in range(download_attempts):
        proxy = proxy_candidates[min(attempt, len(proxy_candidates) - 1)]
        try:
            logger.info(
                "Download attempt %s/%s via %s",
                attempt + 1,
                download_attempts,
                proxy or "direct",
            )

            ydl_opts = _apply_proxy(
                _apply_cookies(downloader.get_optimal_download_options(video_id)),
                proxy=proxy,
            )
            # Lower format target through flaky free proxies for faster, stabler pulls.
            # Keep a broad fallback: some proxies only expose progressive or HLS forms.
            if proxy:
                ydl_opts["format"] = (
                    "bestvideo[height<=720]+bestaudio/"
                    "best[height<=720]/"
                    "bestvideo*[height<=720]+bestaudio/"
                    "bv*+ba/b"
                )
                ydl_opts["socket_timeout"] = 45
                ydl_opts["retries"] = 8
                ydl_opts["fragment_retries"] = 8

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            logger.info(f"Searching for downloaded file: {video_id}.*")
            downloaded_files = [
                file_path
                for file_path in downloader.temp_dir.glob(f"{video_id}.*")
                if file_path.is_file()
                and file_path.suffix.lower() in [".mp4", ".mkv", ".webm"]
            ]
            if downloaded_files:
                ranked_files = []
                for candidate in downloaded_files:
                    width, height = _get_local_video_dimensions(candidate)
                    ranked_files.append(
                        (
                            height,
                            width,
                            candidate.stat().st_size,
                            candidate,
                        )
                    )
                ranked_files.sort(reverse=True)
                best_downloaded_file = ranked_files[0][3]
                file_size = best_downloaded_file.stat().st_size
                width, height = _get_local_video_dimensions(best_downloaded_file)
                if proxy:
                    _write_cached_proxy(proxy)
                logger.info(
                    f"Download successful: {best_downloaded_file.name} ({file_size // 1024 // 1024}MB, {width}x{height})"
                )
                return best_downloaded_file

            logger.warning("No video file found after download attempt %s", attempt + 1)

        except yt_dlp.utils.DownloadError as e:
            last_error = str(e)
            logger.warning("Download attempt %s failed: %s", attempt + 1, e)
            if attempt < download_attempts - 1:
                wait_time = min(8, 2**attempt)
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error("All download attempts failed")

        except Exception as e:
            last_error = str(e)
            logger.error(
                "Unexpected error during download attempt %s: %s",
                attempt + 1,
                e,
            )
            if attempt < download_attempts - 1:
                wait_time = min(8, 2**attempt)
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error("All download attempts failed")

    if last_error:
        logger.error("All download attempts failed for %s: %s", url, last_error)

    return None


def download_youtube_video(
    url: str,
    max_retries: int = 3,
    task_id: Optional[str] = None,
) -> Optional[Path]:
    """
    Download YouTube video using the configured provider.
    Defaults to free yt-dlp downloads and optionally falls back to Apify when configured.
    """
    logger.info("Starting YouTube download: %s", url)

    video_id = get_youtube_video_id(url)
    if not video_id:
        logger.error("Could not extract video ID from URL: %s", url)
        return None

    config = get_config()
    if config.youtube_download_provider == "videoscale":
        return download_video_via_videoscale(url, video_id)
    if config.youtube_download_provider == "oxylabs":
        return download_video_via_oxylabs(video_id)

    downloader = YouTubeDownloader()
    _remove_cached_downloads(downloader.temp_dir, video_id)

    config = get_config()
    primary_provider = config.youtube_download_provider
    secondary_provider = (
        YOUTUBE_DOWNLOAD_PROVIDER_APIFY
        if primary_provider == YOUTUBE_DOWNLOAD_PROVIDER_YTDLP
        else YOUTUBE_DOWNLOAD_PROVIDER_YTDLP
    )

    for provider in (primary_provider, secondary_provider):
        if provider == YOUTUBE_DOWNLOAD_PROVIDER_YTDLP:
            downloaded_path = _download_youtube_video_with_ytdlp(
                url,
                max_retries,
                task_id,
            )
            if downloaded_path:
                return downloaded_path

            logger.warning("yt-dlp download failed for %s", url)
            continue

        if not config.apify_api_token:
            logger.info("APIFY_API_TOKEN not set; skipping Apify download for %s", url)
            continue

        try:
            downloaded_path = download_youtube_video_with_apify(url, video_id)
            file_size = downloaded_path.stat().st_size
            width, height = _get_local_video_dimensions(downloaded_path)
            logger.info(
                "Apify download successful: %s (%sMB, %sx%s)",
                downloaded_path.name,
                file_size // 1024 // 1024,
                width,
                height,
            )
            return downloaded_path
        except ApifyDownloadError as exc:
            logger.warning("Apify download failed for %s: %s", url, exc)
        except Exception as exc:
            logger.error("Unexpected Apify download error for %s: %s", url, exc)

    logger.error("All YouTube download providers failed for %s", url)
    return None


async def async_download_youtube_video(
    url: str,
    max_retries: int = 3,
    task_id: Optional[str] = None,
) -> Optional[Path]:
    logger.info(f"Starting async YouTube download: {url}")
    return await asyncio.to_thread(download_youtube_video, url, max_retries, task_id)


def get_video_duration(url: str) -> Optional[int]:
    """Get video duration in seconds without downloading."""
    video_info = get_youtube_video_info(url)
    return video_info.get("duration") if video_info else None


def is_video_suitable_for_processing(
    url: str, min_duration: int = 60, max_duration: int = 7200
) -> bool:
    """
    Check if video is suitable for processing based on duration and other factors.
    Default limits: 1 minute to 2 hours.
    """
    video_info = get_youtube_video_info(url)
    if not video_info:
        return False

    duration = video_info.get("duration", 0)

    # Check duration constraints
    if duration < min_duration or duration > max_duration:
        logger.warning(
            f"Video duration {duration}s outside allowed range ({min_duration}-{max_duration}s)"
        )
        return False

    # Additional checks could go here (e.g., content type, quality, etc.)

    return True


def cleanup_downloaded_files(video_id: str, source_path: Optional[Path] = None):
    """Remove downloaded media while retaining word timings for cached jobs."""
    temp_dir = Path(get_config().temp_dir)
    provider_dir = False
    if source_path is not None and source_path.parent.name.startswith((f"oxy-{video_id}-", f"videoscale-{video_id}-")):
        candidate = source_path.parent.resolve()
        if candidate.parent != temp_dir.resolve():
            logger.warning("Refusing source cleanup outside the configured download directory")
            return
        temp_dir = candidate
        provider_dir = True

    for file_path in temp_dir.glob(f"{video_id}.*"):
        if file_path.name == f"{video_id}.transcript_cache.json":
            continue
        try:
            if file_path.is_file():
                file_path.unlink()
                logger.info(f"Cleaned up: {file_path.name}")
        except Exception as e:
            logger.warning(f"Failed to cleanup {file_path.name}: {e}")

    if provider_dir:
        try:
            temp_dir.rmdir()
        except OSError:
            pass  # Keep a directory containing cached word timings or other files.


# Backward compatibility functions
def extract_video_id(url: str) -> Optional[str]:
    """Backward compatibility wrapper."""
    return get_youtube_video_id(url)
