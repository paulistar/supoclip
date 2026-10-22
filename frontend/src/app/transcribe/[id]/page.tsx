"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import {
  ArrowLeft,
  Download,
  FileJson,
  FileText,
  Loader2,
  PauseCircle,
  RotateCcw,
  Subtitles,
} from "lucide-react";
import { useSession } from "@/lib/auth-client";
import { useTaskProgress } from "@/hooks/use-task-progress";
import {
  ACTIVE_TASK_STATUSES,
  RESUMABLE_TASK_STATUSES,
  StatusBadge,
} from "@/components/app/status-badge";
import { PageError, PageLoading } from "@/components/app/page-state";
import { Button } from "@/components/ui/button";
import { formatSupportMessage, parseApiError } from "@/lib/api-error";
import { cn } from "@/lib/utils";

type TranscribeTask = {
  id: string;
  source_title: string;
  source_type: string;
  status: string;
  progress?: number | null;
  progress_message?: string | null;
  job_kind?: string;
  transcript_preview?: string | null;
  transcript_char_count?: number | null;
  language?: string | null;
  created_at: string;
};

export default function TranscribeResultPage() {
  const params = useParams();
  const taskId = String(params?.id || "");
  const { data: session, isPending } = useSession();
  const [task, setTask] = useState<TranscribeTask | null>(null);
  const [transcriptText, setTranscriptText] = useState<string>("");
  const [loadError, setLoadError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [downloadBusy, setDownloadBusy] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const response = await fetch(`/api/tasks/${taskId}`, { cache: "no-store" });
    if (!response.ok) {
      const info = await parseApiError(response, "Falha ao carregar a tarefa");
      throw new Error(formatSupportMessage(info));
    }
    const data = (await response.json()) as TranscribeTask;
    setTask(data);
    return data;
  }, [taskId]);

  const loadTranscriptText = useCallback(async () => {
    const response = await fetch(
      `/api/tasks/${taskId}/transcript?format=txt`,
      { cache: "no-store" },
    );
    if (!response.ok) {
      const info = await parseApiError(response, "Transcript indisponível");
      throw new Error(formatSupportMessage(info));
    }
    setTranscriptText(await response.text());
  }, [taskId]);

  useEffect(() => {
    if (!session?.user || !taskId) return;
    let cancelled = false;
    (async () => {
      try {
        const data = await refresh();
        if (cancelled) return;
        if (data.status === "completed") {
          try {
            await loadTranscriptText();
          } catch (error) {
            if (!cancelled) {
              setTranscriptText(data.transcript_preview || "");
              setLoadError(
                error instanceof Error ? error.message : "Falha ao carregar texto",
              );
            }
          }
        }
      } catch (error) {
        if (!cancelled) {
          setLoadError(
            error instanceof Error ? error.message : "Falha ao carregar",
          );
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [session?.user, taskId, refresh, loadTranscriptText]);

  const active = ACTIVE_TASK_STATUSES.includes(task?.status || "");

  useTaskProgress({
    taskId,
    active,
    refresh: async () => {
      const data = await refresh();
      if (data.status === "completed" && !transcriptText) {
        try {
          await loadTranscriptText();
          setLoadError(null);
        } catch {
          setTranscriptText(data.transcript_preview || "");
        }
      }
    },
    onProgress: (event) => {
      setTask((prev) =>
        prev
          ? {
              ...prev,
              progress: event.progress ?? prev.progress,
              progress_message: event.message ?? prev.progress_message,
              status: event.status || prev.status,
            }
          : prev,
      );
    },
    onClip: () => {},
  });

  const runAction = async (action: "cancel" | "resume") => {
    setBusy(true);
    try {
      const response = await fetch(`/api/tasks/${taskId}/${action}`, {
        method: "POST",
      });
      if (!response.ok) {
        const info = await parseApiError(response, `Falha ao ${action}`);
        throw new Error(formatSupportMessage(info));
      }
      await refresh();
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : "Ação falhou");
    } finally {
      setBusy(false);
    }
  };

  const download = async (format: "txt" | "srt" | "json") => {
    setDownloadBusy(format);
    try {
      const response = await fetch(
        `/api/tasks/${taskId}/transcript?format=${format}`,
        { cache: "no-store" },
      );
      if (!response.ok) {
        const info = await parseApiError(response, "Download falhou");
        throw new Error(formatSupportMessage(info));
      }
      const blob = await response.blob();
      const disposition = response.headers.get("content-disposition") || "";
      const match = /filename="?([^"]+)"?/i.exec(disposition);
      const filename = match?.[1] || `transcript.${format}`;
      const href = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = href;
      anchor.download = filename;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(href);
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : "Download falhou");
    } finally {
      setDownloadBusy(null);
    }
  };

  if (isPending || !session?.user) {
    return <PageLoading />;
  }

  if (!task && loadError) {
    return <PageError message={loadError} />;
  }

  if (!task) {
    return <PageLoading />;
  }

  const resumable = RESUMABLE_TASK_STATUSES.includes(task.status);
  const completed = task.status === "completed";
  const progressValue = Math.max(0, Math.min(100, Number(task.progress) || 0));

  return (
    <div className="mx-auto w-full max-w-3xl space-y-6 px-4 py-8 sm:px-6">
      <div className="flex items-center gap-3">
        <Button asChild variant="ghost" size="sm">
          <Link href="/transcribe">
            <ArrowLeft className="size-4" />
            Voltar
          </Link>
        </Button>
        <StatusBadge status={task.status} />
      </div>

      <header className="space-y-1">
        <h1 className="font-display text-2xl font-semibold tracking-tight">
          {task.source_title || "Transcrição"}
        </h1>
        <p className="text-sm text-muted-foreground">
          {task.source_type === "youtube" ? "YouTube" : "Upload"}
          {task.language ? ` · ${task.language}` : " · pt"}
          {task.transcript_char_count
            ? ` · ${task.transcript_char_count.toLocaleString()} caracteres`
            : ""}
        </p>
      </header>

      {active ? (
        <div className="space-y-3 rounded-xl border p-4">
          <div className="flex items-center justify-between gap-3">
            <p className="text-sm text-muted-foreground">
              {task.progress_message || "Processando…"}
            </p>
            <Button
              type="button"
              variant="outline"
              size="sm"
              disabled={busy}
              onClick={() => void runAction("cancel")}
            >
              <PauseCircle className="size-4" />
              Cancelar
            </Button>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-muted">
            <div
              className="h-full rounded-full bg-brand transition-all"
              style={{ width: `${progressValue}%` }}
            />
          </div>
        </div>
      ) : null}

      {resumable ? (
        <div className="flex flex-wrap items-center gap-3 rounded-xl border border-amber-500/30 bg-amber-500/5 p-4">
          <p className="flex-1 text-sm">
            {task.progress_message || "Esta transcrição parou."}
          </p>
          <Button
            type="button"
            size="sm"
            disabled={busy}
            onClick={() => void runAction("resume")}
          >
            <RotateCcw className="size-4" />
            Retomar
          </Button>
        </div>
      ) : null}

      {loadError && completed ? (
        <p className="text-sm text-red-600 dark:text-red-400" role="alert">
          {loadError}
        </p>
      ) : null}

      {completed ? (
        <>
          <div className="flex flex-wrap gap-2">
            <Button
              type="button"
              variant="outline"
              size="sm"
              disabled={Boolean(downloadBusy)}
              onClick={() => void download("txt")}
            >
              {downloadBusy === "txt" ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <FileText className="size-4" />
              )}
              TXT
            </Button>
            <Button
              type="button"
              variant="outline"
              size="sm"
              disabled={Boolean(downloadBusy)}
              onClick={() => void download("srt")}
            >
              {downloadBusy === "srt" ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <Subtitles className="size-4" />
              )}
              SRT
            </Button>
            <Button
              type="button"
              variant="outline"
              size="sm"
              disabled={Boolean(downloadBusy)}
              onClick={() => void download("json")}
            >
              {downloadBusy === "json" ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <FileJson className="size-4" />
              )}
              JSON
            </Button>
            <Button
              type="button"
              size="sm"
              disabled={Boolean(downloadBusy)}
              onClick={() => void download("txt")}
            >
              <Download className="size-4" />
              Baixar
            </Button>
          </div>

          <article
            className={cn(
              "max-h-[60vh] overflow-auto rounded-xl border bg-muted/20 p-4",
              "whitespace-pre-wrap text-sm leading-relaxed",
            )}
          >
            {transcriptText || task.transcript_preview || "Sem texto."}
          </article>
        </>
      ) : null}
    </div>
  );
}
