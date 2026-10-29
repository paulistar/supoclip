import { parseApiError, formatSupportMessage } from "@/lib/api-error";

export const MAX_VIDEO_UPLOAD_BYTES = 12_000_000_000;
export const MAX_MEDIA_UPLOAD_BYTES = MAX_VIDEO_UPLOAD_BYTES;

/** Accept attribute for video + common audio formats (transcribe upload). */
export const MEDIA_UPLOAD_ACCEPT =
  "video/*,audio/*,.mp3,.wav,.ogg,.opus,.m4a,.aac,.flac,.webm,.mp4,.mov,.mkv";

const AUDIO_OR_VIDEO_EXTENSIONS = new Set([
  "mp3",
  "wav",
  "ogg",
  "opus",
  "m4a",
  "aac",
  "flac",
  "webm",
  "mp4",
  "mov",
  "mkv",
  "m4v",
  "avi",
]);

export function isAcceptedMediaFile(file: File): boolean {
  if (file.type.startsWith("video/") || file.type.startsWith("audio/")) {
    return true;
  }
  const ext = file.name.split(".").pop()?.toLowerCase() || "";
  return AUDIO_OR_VIDEO_EXTENSIONS.has(ext);
}

// Only surface the font search box once the list is long enough to need it.
export const FONT_SEARCH_THRESHOLD = 8;

type DirectUploadAuthorization = {
  directUpload: true;
  uploadUrl: string;
  headers: Record<string, string>;
};

type ProxyUploadAuthorization = {
  directUpload: false;
  reason: "signed_backend_auth_required";
};

type UploadAuthorization = DirectUploadAuthorization | ProxyUploadAuthorization;

const extractYouTubeVideoId = (value: string): string | null => {
  const input = value.trim();
  if (!input) return null;

  try {
    const parsed = new URL(input);
    const host = parsed.hostname.replace(/^www\./, "");

    if (host === "youtu.be") {
      const id = parsed.pathname.split("/").filter(Boolean)[0];
      return id && id.length === 11 ? id : null;
    }

    if (host === "youtube.com" || host === "m.youtube.com" || host === "music.youtube.com") {
      const fromSearch = parsed.searchParams.get("v");
      if (fromSearch && fromSearch.length === 11) {
        return fromSearch;
      }

      const pathParts = parsed.pathname.split("/").filter(Boolean);
      const embedId = pathParts[0] === "embed" ? pathParts[1] : null;
      if (embedId && embedId.length === 11) {
        return embedId;
      }
    }
  } catch {
    return null;
  }

  return null;
};

export const getYouTubeThumbnailUrl = (value: string): string | null => {
  const videoId = extractYouTubeVideoId(value);
  return videoId ? `https://i.ytimg.com/vi/${videoId}/hqdefault.jpg` : null;
};

async function requestUploadAuthorization(): Promise<UploadAuthorization> {
  const response = await fetch("/api/upload/authorization", {
    method: "POST",
    cache: "no-store",
  });

  if (!response.ok) {
    const uploadError = await parseApiError(
      response,
      `Upload authorization error: ${response.status}`,
    );
    throw new Error(formatSupportMessage(uploadError));
  }

  return response.json() as Promise<UploadAuthorization>;
}

export async function uploadVideoFile(file: File): Promise<string> {
  return uploadMediaFile(file);
}

/** Upload video or audio; returns an `upload://…` path for task creation. */
export async function uploadMediaFile(file: File): Promise<string> {
  if (file.size > MAX_MEDIA_UPLOAD_BYTES) {
    throw new Error("Arquivo muito grande. Envie um arquivo com menos de 12 GB.");
  }

  const uploadAuthorization = await requestUploadAuthorization();
  if (!uploadAuthorization.directUpload) {
    return uploadMediaFileViaProxy(file);
  }

  const formData = new FormData();
  formData.append("video", file);

  const uploadResponse = await fetch(uploadAuthorization.uploadUrl, {
    method: "POST",
    headers: uploadAuthorization.headers,
    body: formData,
  });

  if (!uploadResponse.ok) {
    const fallbackMessage =
      uploadResponse.status === 413
        ? "Arquivo muito grande. Envie um arquivo com menos de 12 GB."
        : `Upload error: ${uploadResponse.status}`;
    const uploadError = await parseApiError(uploadResponse, fallbackMessage);
    throw new Error(formatSupportMessage(uploadError));
  }

  const uploadResult = await uploadResponse.json();
  if (typeof uploadResult.video_path !== "string" || !uploadResult.video_path) {
    throw new Error("Upload finished without a video path. Please try again.");
  }

  return uploadResult.video_path;
}

export async function uploadVideoFileViaProxy(file: File): Promise<string> {
  return uploadMediaFileViaProxy(file);
}

export async function uploadMediaFileViaProxy(file: File): Promise<string> {
  const formData = new FormData();
  formData.append("video", file);

  const uploadResponse = await fetch("/api/upload", {
    method: "POST",
    body: formData,
  });

  if (!uploadResponse.ok) {
    const fallbackMessage =
      uploadResponse.status === 413
        ? "Arquivo muito grande. Envie um arquivo com menos de 12 GB."
        : `Upload error: ${uploadResponse.status}`;
    const uploadError = await parseApiError(uploadResponse, fallbackMessage);
    throw new Error(formatSupportMessage(uploadError));
  }

  const uploadResult = await uploadResponse.json();
  if (typeof uploadResult.video_path !== "string" || !uploadResult.video_path) {
    throw new Error("Upload finished without a video path. Please try again.");
  }

  return uploadResult.video_path;
}
