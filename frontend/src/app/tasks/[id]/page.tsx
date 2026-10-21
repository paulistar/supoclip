"use client";

import { CaptionSizeControl } from "@/components/caption-size-control";

import { useState, useEffect, useCallback } from "react";
import { toast } from "sonner";
import { useTaskProgress } from "@/hooks/use-task-progress";
import { StatusBadge, ACTIVE_TASK_STATUSES, REVIEWABLE_TASK_STATUSES } from "@/components/app/status-badge";
import { getClipUrl, requestAction, downloadBlob, EXPORT_PRESETS } from "@/lib/clip-actions";
import { trackClipExport } from "@/lib/clip-export-tracking";
import { useParams, useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { PageLoading, PageError } from "@/components/app/page-state";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
  SheetFooter,
} from "@/components/ui/sheet";
import { useSession } from "@/lib/auth-client";
import { ApiRequestError, formatSupportMessage, parseApiError } from "@/lib/api-error";
import { useBillingSummary } from "@/hooks/use-billing-summary";
import { UpgradeDialog, upgradeReasonForError, type UpgradeReason } from "@/components/billing/upgrade-prompt";
import { buildFontOptionsPayload, FONT_TEMPLATE_DEFAULT_VALUE } from "@/lib/font-options";
import {
  AlertCircle,
  ArrowLeft,
  Check,
  Clapperboard,
  Clock,
  Edit2,
  Link2Off,
  Loader2,
  PauseCircle,
  RefreshCw,
  RotateCcw,
  Settings2,
  Share2,
  Sparkles,
  Trash2,
  Upload,
  X,
  Youtube,
} from "lucide-react";
import {
  DURATION_PRESETS,
  MIN_CLIPS_OPTIONS,
  type DurationPresetId,
  type MinClipsChoice,
} from "@/lib/clip-selection";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip";
import Link from "next/link";
import { ClipTile, ClipFocus, ProcessingPanel, EmptyState } from "@/components/app/clip-wall";
import { timeAgo } from "@/lib/generations";
import { cn } from "@/lib/utils";
import { FontSelectOption, type FontOption } from "@/components/font-select-option";

interface Clip {
  id: string;
  filename: string;
  file_path: string;
  start_time: string;
  end_time: string;
  duration: number;
  text: string;
  relevance_score: number;
  reasoning: string;
  clip_order: number;
  created_at: string;
  video_url: string;
  // Virality scores
  virality_score: number;
  hook_score: number;
  engagement_score: number;
  value_score: number;
  shareability_score: number;
  hook_type: string | null;
  hook_title: string | null;
}

interface TaskDetails {
  id: string;
  user_id: string;
  source_id: string;
  source_title: string;
  source_type: string;
  status: string;
  progress?: number;
  progress_message?: string;
  clips_count: number;
  created_at: string;
  updated_at: string;
  font_family?: string | null;
  font_size?: number | null;
  font_color?: string | null;
  caption_template?: string;
  cut_long_pauses?: boolean;
  pause_threshold_ms?: number;
  remove_filler_words?: boolean;
  filtered_words?: string[];
  share_enabled?: boolean;
  clip_duration_preset?: DurationPresetId;
  min_clips?: MinClipsChoice | "auto" | number;
  clip_brief?: string;
  excellent_clip_count?: number | null;
  rendered_clip_count?: number | null;
  remaining_excellent?: number | null;
  drop_counts?: Record<string, number>;
  review_before_render?: boolean;
  eta_seconds_total?: number | null;
}

type ReviewCandidate = {
  index: number;
  start_time: string;
  end_time: string;
  duration_seconds: number;
  hook_title: string | null;
  text_preview: string;
  virality_score: number;
  hook_score: number;
  rendered: boolean;
  skipped: boolean;
  eta_seconds: number;
};

export default function TaskPage() {
  const params = useParams();
  const router = useRouter();
  const { data: session } = useSession();
  const [task, setTask] = useState<TaskDetails | null>(null);
  const [clips, setClips] = useState<Clip[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState("");
  const [isEditing, setIsEditing] = useState(false);
  const [editedTitle, setEditedTitle] = useState("");
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [deletingClipId, setDeletingClipId] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [exportPreset, setExportPreset] = useState("original");
  const [shareState, setShareState] = useState<"idle" | "copying" | "copied">("idle");
  const [isRevokingShare, setIsRevokingShare] = useState(false);

  // null means "use the caption template's own value" — mirrors the create form's contract.
  const [projectFontFamily, setProjectFontFamily] = useState<string | null>(null);
  const [projectFontSize, setProjectFontSize] = useState<number | null>(null);
  const [projectFontColor, setProjectFontColor] = useState<string | null>(null);
  const [projectCaptionTemplate, setProjectCaptionTemplate] = useState("default");
  const [projectCutLongPauses, setProjectCutLongPauses] = useState(false);
  const [projectPauseThresholdMs, setProjectPauseThresholdMs] = useState("700");
  const [projectRemoveFillerWords, setProjectRemoveFillerWords] = useState(false);
  const [projectFilteredWords, setProjectFilteredWords] = useState("");
  const [isApplyingSettings, setIsApplyingSettings] = useState(false);
  const [settingsSheetOpen, setSettingsSheetOpen] = useState(false);
  const [opsSheetOpen, setOpsSheetOpen] = useState(false);
  const [opsPreset, setOpsPreset] = useState<DurationPresetId>("auto");
  const [opsMinClips, setOpsMinClips] = useState<MinClipsChoice>("auto");
  const [opsBrief, setOpsBrief] = useState("");
  const [generateMoreCount, setGenerateMoreCount] = useState<5 | 10 | 15>(5);
  const [candidates, setCandidates] = useState<ReviewCandidate[]>([]);
  const [selectedStarts, setSelectedStarts] = useState<Set<string>>(new Set());
  const [candidatesMeta, setCandidatesMeta] = useState<{
    pending_count: number;
    eta_seconds_pending: number;
    eta_seconds_top15: number;
  } | null>(null);
  const [availableFonts, setAvailableFonts] = useState<FontOption[]>([]);
  const [deletingFontName, setDeletingFontName] = useState<string | null>(null);
  const [availableTemplates, setAvailableTemplates] = useState<
    Array<{ id: string; name: string; description: string; animation: string }>
  >([]);
  const [fontToDelete, setFontToDelete] = useState<FontOption | null>(null);
  const [pendingAction, setPendingAction] = useState<string | null>(null);
  const [billingSummary] = useBillingSummary(Boolean(session?.user?.id));
  const [upgradeReason, setUpgradeReason] = useState<UpgradeReason | null>(null);
  const [sortBy, setSortBy] = useState<"score" | "timeline">("score");
  const [focusIndex, setFocusIndex] = useState<number | null>(null);

  const taskApiUrl = "/api/tasks";

  const buildSupportError = useCallback(async (response: Response, fallbackMessage: string) => {
    const parsed = await parseApiError(response, fallbackMessage);
    return formatSupportMessage(parsed);
  }, []);

  const fetchTaskStatus = useCallback(
    async (retryCount = 0, maxRetries = 5, background = false) => {
      if (!params.id) return false;

      try {
        const taskResponse = await fetch(`${taskApiUrl}/${params.id}`, {
          cache: "no-store",
        });

        // Handle 404 with retry logic (task might not be persisted yet)
        if (taskResponse.status === 404 && retryCount < maxRetries) {
          console.log(
            `Task not found yet, retrying in ${(retryCount + 1) * 500}ms... (${retryCount + 1}/${maxRetries})`,
          );
          await new Promise((resolve) => setTimeout(resolve, (retryCount + 1) * 500));
          return fetchTaskStatus(retryCount + 1, maxRetries, background);
        }

        if (!taskResponse.ok) {
          throw new Error(await buildSupportError(taskResponse, `Failed to fetch task: ${taskResponse.status}`));
        }

        const taskData = await taskResponse.json();
        setProgress(taskData.progress ?? 0);
        setProgressMessage(taskData.progress_message ?? "");
        setError(null);
        if (!background) {
        setProjectFontFamily(taskData.font_family ?? null);
        setProjectFontSize(typeof taskData.font_size === "number" ? taskData.font_size : null);
        setProjectFontColor(taskData.font_color ?? null);
        setProjectCaptionTemplate(taskData.caption_template || "default");
        setProjectCutLongPauses(Boolean(taskData.cut_long_pauses));
        setProjectPauseThresholdMs(String(taskData.pause_threshold_ms || 700));
        setProjectRemoveFillerWords(Boolean(taskData.remove_filler_words));
        setProjectFilteredWords((taskData.filtered_words || []).join(", "));
        setOpsPreset((taskData.clip_duration_preset as DurationPresetId) || "auto");
        const minRaw = taskData.min_clips;
        setOpsMinClips(
          minRaw === "auto" || minRaw == null
            ? "auto"
            : ([5, 10, 15, 20].includes(Number(minRaw)) ? (Number(minRaw) as MinClipsChoice) : "auto"),
        );
        setOpsBrief(typeof taskData.clip_brief === "string" ? taskData.clip_brief : "");
        }

        // Fetch clips if task is completed or processing (incremental clips)
        if (taskData.status !== "queued") {
          const clipsResponse = await fetch(`${taskApiUrl}/${params.id}/clips`, {
            cache: "no-store",
          });

          if (!clipsResponse.ok) {
            throw new Error(await buildSupportError(clipsResponse, `Failed to fetch clips: ${clipsResponse.status}`));
          }

          const clipsData = await clipsResponse.json();
          const nextClips = clipsData.clips || [];
          setClips((prev) => {
            if (taskData.status === "completed") {
              return nextClips;
            }

            const merged = new Map<string, Clip>();
            for (const clip of prev) {
              merged.set(clip.id, clip);
            }
            for (const clip of nextClips) {
              merged.set(clip.id, clip);
            }
            return Array.from(merged.values()).sort(
              (a, b) => (a.clip_order ?? 0) - (b.clip_order ?? 0),
            );
          });
        }

        setTask(taskData);
        return true;
      } catch (err) {
        console.error("Error fetching task data:", err);
        if (!background) setError(err instanceof Error ? err.message : "Failed to load task");
        return false;
      }
    },
    [buildSupportError, params.id, taskApiUrl],
  );

  const fetchCandidates = useCallback(async () => {
    if (!params.id) return;
    try {
      const response = await fetch(`${taskApiUrl}/${params.id}/candidates`, { cache: "no-store" });
      if (!response.ok) return;
      const data = await response.json();
      const rows: ReviewCandidate[] = data.candidates || [];
      setCandidates(rows);
      setCandidatesMeta({
        pending_count: data.pending_count ?? 0,
        eta_seconds_pending: data.eta_seconds_pending ?? 0,
        eta_seconds_top15: data.eta_seconds_top15 ?? 0,
      });
      setSelectedStarts((prev) => {
        if (prev.size > 0) return prev;
        const pending = rows.filter((r) => !r.rendered && !r.skipped).slice(0, 15);
        return new Set(pending.map((r) => r.start_time));
      });
    } catch (err) {
      console.error("Failed to load candidates:", err);
    }
  }, [params.id, taskApiUrl]);

  // Initial fetch - runs immediately, doesn't wait for session
  useEffect(() => {
    if (!params.id) return;

    const fetchTaskData = async () => {
      try {
        setIsLoading(true);
        await fetchTaskStatus();
      } finally {
        setIsLoading(false);
      }
    };

    fetchTaskData();
  }, [params.id, fetchTaskStatus]);

  useEffect(() => {
    if (task && REVIEWABLE_TASK_STATUSES.includes(task.status)) {
      void fetchCandidates();
    }
  }, [task?.status, task?.id, fetchCandidates]);

  useEffect(() => {
    const loadFonts = async () => {
      try {
        const response = await fetch("/api/fonts", { cache: "no-store" });
        if (!response.ok) {
          return;
        }
        const data = await response.json();
        setAvailableFonts(data.fonts || []);
      } catch (loadError) {
        console.error("Failed to load fonts:", loadError);
      }
    };

    void loadFonts();

    const loadTemplates = async () => {
      try {
        const response = await fetch("/api/caption-templates");
        if (response.ok) {
          const data = await response.json();
          setAvailableTemplates(data.templates || []);
        }
      } catch (error) {
        console.error("Failed to load caption templates:", error);
      }
    };
    void loadTemplates();
  }, []);

  const { reconnecting } = useTaskProgress<Clip>({
    taskId: String(params.id || ""),
    active: ACTIVE_TASK_STATUSES.includes(task?.status || ""),
    refresh: () => fetchTaskStatus(0, 0, true),
    onProgress: (data) => {
      if (typeof data.progress === "number") setProgress(data.progress);
      if (typeof data.message === "string") setProgressMessage(data.message);
      if (typeof data.status === "string") {
        const nextStatus = data.status;
        setTask((current) =>
          current && current.status !== nextStatus
            ? { ...current, status: nextStatus }
            : current,
        );
      }
    },
    onClip: (clip) => setClips((current) =>
      [...current.filter((item) => item.id !== clip.id), clip].sort((a, b) => a.clip_order - b.clip_order)),
  });

  const runAction = async (name: string, action: () => Promise<void>) => {
    if (pendingAction) return;
    setPendingAction(name);
    try { await action(); }
    catch (err) { toast.error(err instanceof Error ? err.message : "Could not complete the action. Please try again."); }
    finally { setPendingAction(null); }
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const getHookTypeLabel = (hookType: string | null) => {
    const labels: Record<string, string> = {
      question: "Question Hook",
      statement: "Bold Statement",
      statistic: "Data/Stats",
      story: "Story Hook",
      contrast: "Contrast Hook",
      none: "No Hook",
    };
    return labels[hookType || "none"] || hookType || "None";
  };

  const handleEditTitle = async () => {
    if (!editedTitle.trim() || !session?.user?.id || !params.id) return;

    try {
      const response = await fetch(`${taskApiUrl}/${params.id}`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ title: editedTitle }),
      });

      if (response.ok) {
        setTask(task ? { ...task, source_title: editedTitle } : null);
        setIsEditing(false);
      } else {
        toast.error(await buildSupportError(response, "Failed to update title"));
      }
    } catch (err) {
      console.error("Error updating title:", err);
      toast.error(err instanceof Error ? err.message : "Failed to update title");
    }
  };

  const handleDeleteTask = async () => {
    if (!session?.user?.id || !params.id) return;

    setIsDeleting(true);
    try {
      const response = await fetch(`${taskApiUrl}/${params.id}`, {
        method: "DELETE",
      });

      if (response.ok) {
        router.push("/list");
      } else {
        toast.error(await buildSupportError(response, "Failed to delete task"));
      }
    } catch (err) {
      console.error("Error deleting task:", err);
      toast.error(err instanceof Error ? err.message : "Failed to delete task");
    } finally {
      setIsDeleting(false);
      setShowDeleteDialog(false);
    }
  };

  const handleDeleteClip = async (clipId: string) => {
    if (!session?.user?.id || !params.id) return;

    try {
      const response = await fetch(`${taskApiUrl}/${params.id}/clips/${clipId}`, {
        method: "DELETE",
      });

      if (response.ok) {
        setClips(clips.filter((clip) => clip.id !== clipId));
        setDeletingClipId(null);
      } else {
        toast.error(await buildSupportError(response, "Failed to delete clip"));
      }
    } catch (err) {
      console.error("Error deleting clip:", err);
      toast.error(err instanceof Error ? err.message : "Failed to delete clip");
    }
  };

  const handleApplyProjectSettings = async () => {
    if (!session?.user?.id || !params.id) return;
    const fontOptions = buildFontOptionsPayload(projectFontFamily, projectFontSize, projectFontColor);
    const parsedPauseThreshold = Number(projectPauseThresholdMs || "700");
    const safePauseThreshold = Number.isFinite(parsedPauseThreshold)
      ? Math.max(250, Math.min(3000, Math.round(parsedPauseThreshold)))
      : 700;
    const normalizedFilteredWords = projectFilteredWords
      .split(",")
      .map((word) => word.trim().toLowerCase())
      .filter(Boolean);

    setIsApplyingSettings(true);
    try {
      const response = await fetch(`${taskApiUrl}/${params.id}/settings`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          ...fontOptions,
          caption_template: projectCaptionTemplate,
          cut_long_pauses: projectCutLongPauses,
          pause_threshold_ms: safePauseThreshold,
          remove_filler_words: projectRemoveFillerWords,
          filtered_words: normalizedFilteredWords,
          apply_to_existing: true,
        }),
      });
      if (!response.ok) {
        toast.error(await buildSupportError(response, "Failed to apply settings"));
        return;
      }
      await fetchTaskStatus();
      setSettingsSheetOpen(false);
      toast.success("Generation settings applied");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to apply settings");
    } finally {
      setIsApplyingSettings(false);
    }
  };

  const handleDeleteFont = async (font: FontOption) => {
    if (font.scope !== "user" || deletingFontName) return;


    setDeletingFontName(font.name);
    try {
      const response = await fetch(`/api/fonts/${encodeURIComponent(font.name)}`, {
        method: "DELETE",
      });
      if (!response.ok) {
        throw new Error(await buildSupportError(response, "Failed to delete font"));
      }

      const remainingFonts = availableFonts.filter((item) => item.name !== font.name);
      setAvailableFonts(remainingFonts);
      if (projectFontFamily === font.name) {
        // The deleted font was in use — fall back to the caption template's own font.
        setProjectFontFamily(null);
      }
    } catch (deleteError) {
      toast.error(deleteError instanceof Error ? deleteError.message : "Failed to delete font");
    } finally {
      setDeletingFontName(null);
    }
  };

  const handleExportClip = async (clipId: string, fallbackFilename: string) => {
    if (!session?.user?.id || !task?.id) return;

    const response = await fetch(`${taskApiUrl}/${task.id}/clips/${clipId}/export?preset=${exportPreset}`, {
      cache: "no-store",
    });

    if (!response.ok) {
      toast.error(await buildSupportError(response, "Failed to export clip"));
      return;
    }

    const blob = await response.blob();
    downloadBlob(blob, `${fallbackFilename.replace(/\.mp4$/i, "")}_${exportPreset}.mp4`);
    void trackClipExport({ userId: session.user.id, surface: "task", format: exportPreset });
  };

  const handleDownloadClip = (clip: Clip) => {
    if (exportPreset === "original") {
      const link = document.createElement("a");
      link.href = getClipUrl(clip.video_url, clip.filename);
      link.download = clip.filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      if (session?.user?.id) void trackClipExport({ userId: session.user.id, surface: "task", format: "original" });
      return;
    }
    void runAction(clip.id, () => handleExportClip(clip.id, clip.filename));
  };

  const handleCopyShareLink = async () => {
    if (!task?.id || shareState === "copying") return;

    setShareState("copying");
    try {
      const response = await fetch(`${taskApiUrl}/${task.id}/share`, {
        method: "POST",
      });
      if (!response.ok) {
        throw new Error(await buildSupportError(response, "Failed to create share link"));
      }

      const data = (await response.json()) as { share_path: string };
      const shareUrl = new URL(data.share_path, window.location.origin).toString();
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(shareUrl);
      } else {
        const input = document.createElement("input");
        input.value = shareUrl;
        input.style.position = "fixed";
        input.style.opacity = "0";
        document.body.appendChild(input);
        input.select();
        document.execCommand("copy");
        input.remove();
      }
      setTask((currentTask) =>
        currentTask ? { ...currentTask, share_enabled: true } : currentTask,
      );
      setShareState("copied");
      window.setTimeout(() => setShareState("idle"), 2500);
    } catch (shareError) {
      setShareState("idle");
      toast.error(shareError instanceof Error ? shareError.message : "Failed to create share link");
    }
  };

  const handleRevokeShareLink = async () => {
    if (!task?.id || isRevokingShare) return;

    setIsRevokingShare(true);
    try {
      const response = await fetch(`${taskApiUrl}/${task.id}/share`, {
        method: "DELETE",
      });
      if (!response.ok) {
        throw new Error(await buildSupportError(response, "Failed to disable share link"));
      }
      setTask((currentTask) =>
        currentTask ? { ...currentTask, share_enabled: false } : currentTask,
      );
    } catch (revokeError) {
      toast.error(revokeError instanceof Error ? revokeError.message : "Failed to disable share link");
    } finally {
      setIsRevokingShare(false);
    }
  };

  if (isLoading) return <PageLoading />;
  if (error) return <PageError message={error} retry={() => void fetchTaskStatus()} />;

  const isActive = task?.status === "processing" || task?.status === "queued";
  const canCancel = isActive || task?.status === "awaiting_review";
  const showReview = Boolean(task && REVIEWABLE_TASK_STATUSES.includes(task.status) && candidates.length > 0);
  const selectedEta = candidates
    .filter((c) => selectedStarts.has(c.start_time) && !c.rendered && !c.skipped)
    .reduce((sum, c) => sum + c.eta_seconds, 0);
  const formatEta = (seconds: number) => {
    if (seconds < 60) return `~${seconds}s`;
    const mins = Math.round(seconds / 60);
    return mins < 60 ? `~${mins} min` : `~${Math.round(mins / 60)} h`;
  };
  const toggleStart = (start: string) => {
    setSelectedStarts((prev) => {
      const next = new Set(prev);
      if (next.has(start)) next.delete(start);
      else next.add(start);
      return next;
    });
  };
  const sortedClips = sortBy === "score"
    ? [...clips].sort((a, b) => (b.virality_score ?? 0) - (a.virality_score ?? 0) || a.clip_order - b.clip_order)
    : clips;
  const focusedClip = focusIndex !== null ? sortedClips[focusIndex] : null;
  const bestScore = clips.reduce((best, clip) => Math.max(best, clip.virality_score || 0), 0);
  const totalSeconds = clips.reduce((sum, clip) => sum + (clip.duration || 0), 0);

  const renderTile = (clip: Clip, index: number) => (
    <ClipTile
      key={clip.id}
      clip={clip}
      taskId={task?.id ?? String(params.id)}
      busy={pendingAction === clip.id}
      best={bestScore > 0 && clip.virality_score === bestScore}
      onOpen={() => setFocusIndex(index)}
      onDownload={() => handleDownloadClip(clip)}
      onDelete={() => setDeletingClipId(clip.id)}
      editable={task?.status === "completed"}
    />
  );

  return (
    <main className="mx-auto w-full max-w-7xl px-4 py-6 sm:px-8 md:py-8">
      <Link href="/list" className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
        <ArrowLeft className="size-3.5" />Library
      </Link>

      {task && (
        <header className="mt-3 flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div className="min-w-0 flex-1">
            {isEditing ? (
              <form className="flex items-center gap-2" onSubmit={(e) => { e.preventDefault(); void handleEditTitle(); }}>
                <Input
                  value={editedTitle}
                  onChange={(e) => setEditedTitle(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Escape") { setIsEditing(false); setEditedTitle(task.source_title); } }}
                  className="h-auto py-1 font-display text-2xl font-bold md:text-3xl"
                  aria-label="Generation title"
                  autoFocus
                />
                <Button type="submit" aria-label="Save title" size="icon" disabled={!editedTitle.trim()}><Check className="size-4" /></Button>
                <Button type="button" size="icon" variant="ghost" aria-label="Cancel title edit" onClick={() => { setIsEditing(false); setEditedTitle(task.source_title); }}><X className="size-4" /></Button>
              </form>
            ) : (
              <div className="group flex items-start gap-2">
                <h1 className={cn("min-w-0 break-words font-display text-2xl font-bold tracking-tight md:text-3xl", isActive && "shimmer")}>{task.source_title}</h1>
                <Button size="icon-sm" variant="ghost" aria-label="Edit title" className="mt-0.5 shrink-0 text-muted-foreground opacity-100 md:opacity-0 md:group-hover:opacity-100 md:focus-visible:opacity-100"
                  onClick={() => { setIsEditing(true); setEditedTitle(task.source_title); }}>
                  <Edit2 className="size-4" />
                </Button>
              </div>
            )}
            <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-2 text-sm text-muted-foreground">
              <StatusBadge status={task.status} />
              <span className="flex items-center gap-1.5 capitalize">{task.source_type === "youtube" ? <Youtube className="size-3.5" /> : <Upload className="size-3.5" />}{task.source_type}</span>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="flex cursor-default items-center gap-1.5"><Clock className="size-3.5" />{timeAgo(task.created_at)}</span>
                </TooltipTrigger>
                <TooltipContent>
                  {new Date(task.created_at).toLocaleString(undefined, { year: "numeric", month: "long", day: "numeric", hour: "2-digit", minute: "2-digit", timeZoneName: "short" })}
                </TooltipContent>
              </Tooltip>
              {clips.length > 0 && <span className="tabular-nums">{clips.length} {clips.length === 1 ? "clip" : "clips"} · {formatDuration(totalSeconds)} total</span>}
              {typeof task.excellent_clip_count === "number" && task.excellent_clip_count > clips.length && (
                <span className="tabular-nums text-foreground/80">
                  {clips.length} de {task.excellent_clip_count} excelentes
                </span>
              )}
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {canCancel && (
              <Button size="sm" variant="outline" disabled={pendingAction !== null}
                onClick={() => void runAction("cancel", async () => {
                  await requestAction(`${taskApiUrl}/${task.id}/cancel`, "POST");
                  await fetchTaskStatus();
                  toast.success("Generation cancelled");
                })}>
                Cancel
              </Button>
            )}
            {(task.status === "cancelled" || task.status === "error") && (
              <Button size="sm" disabled={pendingAction !== null}
                onClick={() => void runAction("resume", async () => {
                  try {
                    await requestAction(`${taskApiUrl}/${task.id}/resume`, "POST");
                  } catch (err) {
                    const reason = err instanceof ApiRequestError ? upgradeReasonForError(err.code, err.detail, undefined) : null;
                    if (!reason) throw err;
                    setUpgradeReason(reason);
                    return;
                  }
                  await fetchTaskStatus();
                  toast.success("Generation resumed");
                })}>
                <RotateCcw className="size-4" />Resume
              </Button>
            )}
            {(task.status === "completed" || task.status === "awaiting_review") && (
              <Button size="sm" variant="outline" onClick={() => setOpsSheetOpen(true)}>
                <RefreshCw className="size-4" /><span className="hidden sm:inline">Ops</span>
              </Button>
            )}
            {task.status === "completed" && clips.length > 0 && (
              <>
                <Button size="sm" variant="outline" onClick={() => setSettingsSheetOpen(true)}>
                  <Settings2 className="size-4" /><span className="hidden sm:inline">Restyle</span>
                </Button>
                <div className="inline-flex items-center rounded-md border shadow-xs">
                  <Button size="sm" variant="ghost" className="rounded-r-none" onClick={handleCopyShareLink} disabled={shareState === "copying"} aria-live="polite">
                    {shareState === "copied" ? <Check className="size-4" /> : <Share2 className="size-4" />}
                    <span className="max-sm:sr-only">{shareState === "copying" ? "Creating link…" : shareState === "copied" ? "Link copied" : "Copy share link"}</span>
                  </Button>
                  {task.share_enabled && (
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <Button size="icon-sm" variant="ghost" className="rounded-l-none border-l" onClick={handleRevokeShareLink} disabled={isRevokingShare} aria-label={isRevokingShare ? "Disabling…" : "Disable share link"}>
                          <Link2Off className="size-4" />
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>Disable share link</TooltipContent>
                    </Tooltip>
                  )}
                </div>
                <Button size="sm" asChild>
                  <Link href={`/tasks/${task.id}/edit`}><Clapperboard className="size-4" />Open Editor</Link>
                </Button>
              </>
            )}
            <Tooltip>
              <TooltipTrigger asChild>
                <Button size="icon-sm" variant="ghost" aria-label="Delete generation" className="text-muted-foreground hover:bg-red-50 hover:text-red-600" onClick={() => setShowDeleteDialog(true)}>
                  <Trash2 className="size-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Delete generation</TooltipContent>
            </Tooltip>
          </div>
        </header>
      )}

      <div className="mt-8">
        {reconnecting && <p role="status" className="mb-4 text-sm text-muted-foreground">Reconnecting to live updates. Your video is still processing.</p>}

        {isActive && task ? (
          <div className="space-y-8">
            <ProcessingPanel status={task.status} progress={progress} message={progressMessage} readyCount={clips.length} />
            {clips.length > 0 && (
              <div className="grid grid-cols-2 gap-x-4 gap-y-6 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
                {sortedClips.map(renderTile)}
              </div>
            )}
          </div>
        ) : !task ? (
          <div className="flex min-h-[50vh] items-center justify-center"><Loader2 className="size-5 animate-spin text-muted-foreground" /></div>
        ) : task.status === "cancelled" && clips.length === 0 ? (
          <EmptyState icon={<PauseCircle className="size-7" />} title="Generation cancelled" body="Resume this generation when you are ready to continue." />
        ) : task.status === "error" ? (
          <EmptyState tone="error" icon={<AlertCircle className="size-7" />} title="Processing Failed" body="There was an error processing your video. Resume to retry from where it stopped, or try another video.">
            <Button asChild variant="outline"><Link href="/"><ArrowLeft className="size-4" />Back to Home</Link></Button>
          </EmptyState>
        ) : showReview && (task.status === "awaiting_review" || candidates.some((c) => !c.rendered && !c.skipped)) ? (
          <div className="space-y-6">
            <div className="rounded-2xl border bg-muted/30 p-4 sm:p-5">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <h2 className="text-base font-semibold tracking-tight">Revisar cortes</h2>
                  <p className="mt-1 text-sm text-muted-foreground">
                    {candidatesMeta?.pending_count ?? 0} pendente(s)
                    {candidatesMeta ? ` · ETA pendentes ${formatEta(candidatesMeta.eta_seconds_pending)}` : ""}
                    {candidatesMeta ? ` · top 15 ${formatEta(candidatesMeta.eta_seconds_top15)}` : ""}
                    {selectedStarts.size > 0 ? ` · seleção ${formatEta(selectedEta)}` : ""}
                  </p>
                </div>
                <div className="flex flex-wrap gap-2">
                  <Button
                    size="sm"
                    disabled={pendingAction !== null || selectedStarts.size === 0}
                    onClick={() => void runAction("render-selected", async () => {
                      await requestAction(`${taskApiUrl}/${task.id}/render-selected`, "POST", {
                        mode: "selected",
                        start_times: Array.from(selectedStarts),
                      });
                      await fetchTaskStatus();
                      toast.success(`Gerando ${selectedStarts.size} corte(s)`);
                    })}
                  >
                    Gerar selecionados
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={pendingAction !== null}
                    onClick={() => void runAction("render-top", async () => {
                      await requestAction(`${taskApiUrl}/${task.id}/render-selected`, "POST", {
                        mode: "top",
                        count: 15,
                      });
                      await fetchTaskStatus();
                      toast.success("Gerando top 15");
                    })}
                  >
                    Top 15
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={pendingAction !== null}
                    onClick={() => void runAction("render-all", async () => {
                      await requestAction(`${taskApiUrl}/${task.id}/render-selected`, "POST", {
                        mode: "all",
                      });
                      await fetchTaskStatus();
                      toast.success("Gerando todos os pendentes");
                    })}
                  >
                    Todos
                  </Button>
                </div>
              </div>
            </div>

            <div className="space-y-2">
              {candidates.map((c) => {
                const disabled = c.rendered || c.skipped;
                const checked = selectedStarts.has(c.start_time);
                return (
                  <label
                    key={c.start_time}
                    className={cn(
                      "flex gap-3 rounded-xl border p-3 transition-colors",
                      disabled ? "opacity-50" : "hover:bg-accent/40",
                      checked && !disabled && "border-foreground/40 bg-accent/20",
                    )}
                  >
                    <input
                      type="checkbox"
                      className="mt-1"
                      disabled={disabled}
                      checked={checked && !disabled}
                      onChange={() => toggleStart(c.start_time)}
                    />
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                        <span className="font-medium text-sm truncate">
                          {c.hook_title || "Sem título"}
                        </span>
                        <span className="text-[11px] tabular-nums text-muted-foreground">
                          {c.start_time}–{c.end_time} · {Math.round(c.duration_seconds)}s · score {c.virality_score} · ETA {formatEta(c.eta_seconds)}
                        </span>
                        {c.rendered && <span className="text-[10px] uppercase tracking-wide text-emerald-600">gerado</span>}
                        {c.skipped && <span className="text-[10px] uppercase tracking-wide text-stone-500">skip</span>}
                      </div>
                      <p className="mt-1 text-xs text-muted-foreground line-clamp-2">{c.text_preview}</p>
                    </div>
                    {!c.rendered && !c.skipped && (
                      <div className="flex shrink-0 flex-col gap-1">
                        <Button
                          type="button"
                          size="sm"
                          variant="ghost"
                          className="text-xs"
                          disabled={pendingAction !== null}
                          onClick={(e) => {
                            e.preventDefault();
                            void runAction(`skip-${c.start_time}`, async () => {
                              await requestAction(`${taskApiUrl}/${task.id}/skip-candidates`, "POST", {
                                start_times: [c.start_time],
                                permanent: false,
                              });
                              setSelectedStarts((prev) => {
                                const next = new Set(prev);
                                next.delete(c.start_time);
                                return next;
                              });
                              await fetchCandidates();
                              toast.success("Corte ignorado nesta geração");
                            });
                          }}
                        >
                          Skip
                        </Button>
                        <Button
                          type="button"
                          size="sm"
                          variant="ghost"
                          className="text-xs text-muted-foreground"
                          disabled={pendingAction !== null}
                          onClick={(e) => {
                            e.preventDefault();
                            void runAction(`evitar-${c.start_time}`, async () => {
                              await requestAction(`${taskApiUrl}/${task.id}/skip-candidates`, "POST", {
                                start_times: [c.start_time],
                                permanent: true,
                              });
                              setSelectedStarts((prev) => {
                                const next = new Set(prev);
                                next.delete(c.start_time);
                                return next;
                              });
                              await fetchCandidates();
                              toast.success("Skip permanente → EVITAR no brief");
                            });
                          }}
                        >
                          EVITAR
                        </Button>
                      </div>
                    )}
                  </label>
                );
              })}
            </div>

            {clips.length > 0 && (
              <div className="grid grid-cols-2 gap-x-4 gap-y-7 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
                {sortedClips.map(renderTile)}
              </div>
            )}
          </div>
        ) : clips.length === 0 ? (
          <EmptyState icon={<AlertCircle className="size-7" />} title="No Clips Generated" body="The generation completed but no clips were generated. The video may not have had suitable content for clipping.">
            <Button asChild><Link href="/"><ArrowLeft className="size-4" />Try Another Video</Link></Button>
          </EmptyState>
        ) : (
          <>
            <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
              <div className="inline-flex rounded-lg border bg-background p-0.5" role="tablist" aria-label="Sort clips">
                {([["score", "Best first"], ["timeline", "Timeline"]] as const).map(([value, label]) => (
                  <button key={value} type="button" role="tab" aria-selected={sortBy === value} onClick={() => setSortBy(value)}
                    className={cn("h-7 rounded-md px-3 text-xs font-medium transition-colors", sortBy === value ? "bg-foreground text-background" : "text-muted-foreground hover:text-foreground")}>
                    {label}
                  </button>
                ))}
              </div>
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <span className="hidden sm:inline">Download as</span>
                <Select value={exportPreset} onValueChange={setExportPreset}>
                  <SelectTrigger size="sm" aria-label="Download format" className="h-8 min-w-[112px] bg-background"><SelectValue /></SelectTrigger>
                  <SelectContent align="end">
                    <SelectItem value="original">Original</SelectItem>
                    {EXPORT_PRESETS.map((preset) => <SelectItem key={preset.id} value={preset.id}>{preset.label}</SelectItem>)}
                  </SelectContent>
                </Select>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-x-4 gap-y-7 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
              {sortedClips.map(renderTile)}
            </div>
          </>
        )}
      </div>

      <ClipFocus
        clip={focusedClip}
        taskId={task?.id ?? String(params.id)}
        index={focusIndex ?? 0}
        total={sortedClips.length}
        busy={focusedClip ? pendingAction === focusedClip.id : false}
        editable={task?.status === "completed"}
        onNavigate={(delta) => setFocusIndex((i) => (i === null ? i : (i + delta + sortedClips.length) % sortedClips.length))}
        onClose={() => setFocusIndex(null)}
        onDownload={() => focusedClip && handleDownloadClip(focusedClip)}
        onDelete={() => { if (focusedClip) { setDeletingClipId(focusedClip.id); setFocusIndex(null); } }}
        hookTypeLabel={getHookTypeLabel}
      />

      <Sheet open={opsSheetOpen} onOpenChange={setOpsSheetOpen}>
        <SheetContent side="right" className="overflow-y-auto sm:max-w-md">
          <SheetHeader>
            <SheetTitle className="flex items-center gap-2"><RefreshCw className="size-4" />Ops do vídeo</SheetTitle>
            <SheetDescription>
              Mesmo vídeo, novo resultado — sem gambiarra. Reanalisar muda brief/preset; gerar mais completa os excelentes que faltam.
            </SheetDescription>
          </SheetHeader>
          <div className="space-y-5 px-4 pb-6">
            {typeof task?.excellent_clip_count === "number" && (
              <p className="rounded-lg border bg-muted/40 px-3 py-2 text-xs text-muted-foreground">
                {clips.length} renderizado(s)
                {task.excellent_clip_count > clips.length
                  ? ` · ${task.excellent_clip_count - clips.length} excelente(s) ainda não gerado(s)`
                  : " · todos os excelentes já gerados"}
                {task.drop_counts && Object.keys(task.drop_counts).length > 0 && (
                  <span className="mt-1 block">
                    Drops: {Object.entries(task.drop_counts).map(([k, v]) => `${k} ${v}`).join(" · ")}
                  </span>
                )}
              </p>
            )}

            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted-foreground">Duração</label>
              <Select value={opsPreset} onValueChange={(v) => setOpsPreset(v as DurationPresetId)}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  {DURATION_PRESETS.map((preset) => (
                    <SelectItem key={preset.id} value={preset.id}>{preset.label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted-foreground">Teto de render</label>
              <div className="grid grid-cols-3 gap-2">
                {MIN_CLIPS_OPTIONS.map((option) => (
                  <button
                    key={String(option)}
                    type="button"
                    onClick={() => setOpsMinClips(option)}
                    className={cn(
                      "rounded-lg border px-2 py-1.5 text-xs font-medium",
                      opsMinClips === option && "border-foreground ring-1 ring-foreground",
                    )}
                  >
                    {option === "auto" ? "Qualidade" : `Até ${option}`}
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted-foreground">Brief</label>
              <Textarea
                value={opsBrief}
                onChange={(e) => setOpsBrief(e.target.value)}
                rows={5}
                placeholder="HARD NEVER / EVITAR… (opcional)"
                className="text-sm"
              />
            </div>

            <div className="space-y-2">
              <Button
                className="w-full"
                disabled={pendingAction !== null}
                onClick={() => void runAction("reanalyze-reuse", async () => {
                  try {
                    await requestAction(`${taskApiUrl}/${task!.id}/reanalyze`, "POST", {
                      reuse_media: true,
                      clip_duration_preset: opsPreset,
                      min_clips: opsMinClips,
                      clip_brief: opsBrief,
                    });
                  } catch (err) {
                    const reason = err instanceof ApiRequestError ? upgradeReasonForError(err.code, err.detail, undefined) : null;
                    if (!reason) throw err;
                    setUpgradeReason(reason);
                    return;
                  }
                  setOpsSheetOpen(false);
                  await fetchTaskStatus();
                  toast.success("Reanálise na fila (reuse STT/vídeo)");
                })}
              >
                <RefreshCw className="size-4" />Reanalisar (reuse STT)
              </Button>
              <Button
                variant="outline"
                className="w-full"
                disabled={pendingAction !== null}
                onClick={() => void runAction("reanalyze-scratch", async () => {
                  try {
                    await requestAction(`${taskApiUrl}/${task!.id}/reanalyze`, "POST", {
                      reuse_media: false,
                      clip_duration_preset: opsPreset,
                      min_clips: opsMinClips,
                      clip_brief: opsBrief,
                    });
                  } catch (err) {
                    const reason = err instanceof ApiRequestError ? upgradeReasonForError(err.code, err.detail, undefined) : null;
                    if (!reason) throw err;
                    setUpgradeReason(reason);
                    return;
                  }
                  setOpsSheetOpen(false);
                  await fetchTaskStatus();
                  toast.success("Reanálise do zero na fila");
                })}
              >
                Do zero (novo STT)
              </Button>
            </div>

            <div className="border-t pt-4 space-y-2">
              <p className="text-xs font-medium text-muted-foreground flex items-center gap-1.5">
                <Sparkles className="size-3.5" />Gerar mais excelentes
              </p>
              <div className="flex gap-2">
                {([5, 10, 15] as const).map((n) => (
                  <button
                    key={n}
                    type="button"
                    onClick={() => setGenerateMoreCount(n)}
                    className={cn(
                      "flex-1 rounded-lg border px-2 py-1.5 text-xs font-medium",
                      generateMoreCount === n && "border-foreground ring-1 ring-foreground",
                    )}
                  >
                    +{n}
                  </button>
                ))}
              </div>
              <Button
                variant="secondary"
                className="w-full"
                disabled={
                  pendingAction !== null
                  || (typeof task?.excellent_clip_count === "number" && task.excellent_clip_count <= clips.length)
                }
                onClick={() => void runAction("generate-more", async () => {
                  try {
                    await requestAction(`${taskApiUrl}/${task!.id}/generate-more`, "POST", {
                      count: generateMoreCount,
                    });
                  } catch (err) {
                    const reason = err instanceof ApiRequestError ? upgradeReasonForError(err.code, err.detail, undefined) : null;
                    if (!reason) throw err;
                    setUpgradeReason(reason);
                    return;
                  }
                  setOpsSheetOpen(false);
                  await fetchTaskStatus();
                  toast.success(`Gerando mais até ${generateMoreCount} cortes`);
                })}
              >
                Gerar mais {generateMoreCount}
              </Button>
            </div>
          </div>
        </SheetContent>
      </Sheet>

      <Sheet open={settingsSheetOpen} onOpenChange={setSettingsSheetOpen}>
        <SheetContent side="right" className="overflow-y-auto sm:max-w-md">
          <SheetHeader>
            <SheetTitle className="flex items-center gap-2"><Settings2 className="size-4" />Generation settings</SheetTitle>
            <SheetDescription>Configure font, caption, and cleanup settings for this generation&apos;s clips.</SheetDescription>
          </SheetHeader>
            <div className="space-y-5 px-4">
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-gray-500">Font</label>
                <Select
                  value={projectFontFamily ?? FONT_TEMPLATE_DEFAULT_VALUE}
                  onValueChange={(value) =>
                    setProjectFontFamily(value === FONT_TEMPLATE_DEFAULT_VALUE ? null : value)
                  }
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Template default" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value={FONT_TEMPLATE_DEFAULT_VALUE}>Template default</SelectItem>
                    {availableFonts.map((font) => (
                      <FontSelectOption
                        key={font.name}
                        font={font}
                        isDeleting={deletingFontName === font.name}
                        onDelete={setFontToDelete}
                      />
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <CaptionSizeControl value={projectFontSize} onChange={setProjectFontSize} disabled={isApplyingSettings} />

              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-medium text-gray-500">Color</label>
                  <label className="flex items-center gap-1.5 text-xs text-gray-500 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={projectFontColor === null}
                      onChange={(e) => setProjectFontColor(e.target.checked ? null : "#FFFFFF")}
                      className="rounded"
                    />
                    Template default
                  </label>
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="color"
                    value={projectFontColor ?? "#FFFFFF"}
                    onChange={(e) => setProjectFontColor(e.target.value)}
                    disabled={projectFontColor === null}
                    className="h-9 w-9 rounded border border-gray-300 cursor-pointer disabled:cursor-not-allowed"
                  />
                  <Input
                    value={projectFontColor ?? ""}
                    onChange={(e) => setProjectFontColor(e.target.value)}
                    disabled={projectFontColor === null}
                    placeholder="Template default"
                  />
                </div>
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-medium text-gray-500">Caption Template</label>
                <Select value={projectCaptionTemplate} onValueChange={setProjectCaptionTemplate}>
                  <SelectTrigger>
                    <SelectValue>
                      {availableTemplates.find((t) => t.id === projectCaptionTemplate)?.name || "Select style"}
                    </SelectValue>
                  </SelectTrigger>
                  <SelectContent>
                    {availableTemplates.map((template) => (
                      <SelectItem key={template.id} value={template.id}>
                        <div>
                          <div className="font-medium">{template.name}</div>
                          <div className="text-xs text-gray-500">{template.description}</div>
                        </div>
                      </SelectItem>
                    ))}
                    {availableTemplates.length === 0 && <SelectItem value="default">Default</SelectItem>}
                  </SelectContent>
                </Select>
              </div>

              <div className="rounded-lg border bg-gray-50 p-3 space-y-3">
                <div>
                  <div className="text-sm font-medium text-gray-900">Clip cleanup</div>
                  <div className="text-xs text-gray-500">Apply silence and filler-word cuts to regenerated clips.</div>
                </div>

                <label className="flex items-center gap-2 text-sm text-gray-700">
                  <input
                    type="checkbox"
                    checked={projectCutLongPauses}
                    onChange={(e) => setProjectCutLongPauses(e.target.checked)}
                    className="rounded"
                  />
                  Shorten long pauses
                </label>

                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-gray-500">Pause threshold (ms)</label>
                  <Input
                    type="number"
                    min={250}
                    max={3000}
                    step={50}
                    value={projectPauseThresholdMs}
                    onChange={(e) => setProjectPauseThresholdMs(e.target.value)}
                    disabled={!projectCutLongPauses}
                  />
                </div>

                <label className="flex items-center gap-2 text-sm text-gray-700">
                  <input
                    type="checkbox"
                    checked={projectRemoveFillerWords}
                    onChange={(e) => setProjectRemoveFillerWords(e.target.checked)}
                    className="rounded"
                  />
                  Remove filler words
                </label>

                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-gray-500">Extra filtered words or phrases</label>
                  <Input
                    value={projectFilteredWords}
                    onChange={(e) => setProjectFilteredWords(e.target.value)}
                    placeholder="basically, literally, to be honest"
                  />
                </div>
              </div>
            </div>

            <SheetFooter>
              <Button
                className="w-full"
                onClick={handleApplyProjectSettings}
                disabled={isApplyingSettings}
              >
                {isApplyingSettings ? "Applying..." : "Apply to All Clips"}
              </Button>
            </SheetFooter>
        </SheetContent>
      </Sheet>
      <AlertDialog open={fontToDelete !== null} onOpenChange={(open) => { if (!open) setFontToDelete(null); }}>
        <AlertDialogContent>
          <AlertDialogHeader><AlertDialogTitle>Delete font?</AlertDialogTitle>
            <AlertDialogDescription>Delete {fontToDelete?.display_name}? This cannot be undone.</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter><AlertDialogCancel>Keep font</AlertDialogCancel>
            <AlertDialogAction onClick={() => { if (fontToDelete) void handleDeleteFont(fontToDelete); setFontToDelete(null); }}>Delete font</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
      {/* Delete generation confirmation */}
      <AlertDialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Generation</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete this generation? This will permanently delete all clips and cannot be
              undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={isDeleting}>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleDeleteTask} disabled={isDeleting} className="bg-red-600 hover:bg-red-700">
              {isDeleting ? "Deleting..." : "Delete"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Delete Clip Confirmation Dialog */}
      <AlertDialog open={!!deletingClipId} onOpenChange={(open) => !open && setDeletingClipId(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Clip</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete this clip? This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => deletingClipId && handleDeleteClip(deletingClipId)}
              className="bg-red-600 hover:bg-red-700"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
      <UpgradeDialog
        open={upgradeReason !== null}
        onOpenChange={(open) => { if (!open) setUpgradeReason(null); }}
        reason={upgradeReason ?? { kind: "limit" }}
        billing={billingSummary}
        source="task_resume"
      />
    </main>
  );
}
