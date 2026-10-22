"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { FileText, Loader2, Upload, Youtube } from "lucide-react";
import { useSession } from "@/lib/auth-client";
import { getYouTubeThumbnailUrl, uploadVideoFile } from "@/lib/video-upload";
import { formatSupportMessage, parseApiError } from "@/lib/api-error";
import { fetchGenerations, timeAgo, type GenerationSummary } from "@/lib/generations";
import { StatusBadge, ACTIVE_TASK_STATUSES } from "@/components/app/status-badge";
import { PageLoading } from "@/components/app/page-state";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

export default function TranscribePage() {
  const { data: session, isPending } = useSession();
  const router = useRouter();
  const [url, setUrl] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState("");
  const [history, setHistory] = useState<GenerationSummary[]>([]);
  const [historyLoading, setHistoryLoading] = useState(true);

  const sourceType: "youtube" | "upload" = file ? "upload" : "youtube";
  const youtubeThumbnailUrl = sourceType === "youtube" ? getYouTubeThumbnailUrl(url) : null;
  const hasSource = sourceType === "upload" ? Boolean(file) : Boolean(url.trim());

  const loadHistory = useCallback(async () => {
    try {
      const tasks = await fetchGenerations({ jobKind: "transcribe" });
      setHistory(tasks);
    } catch {
      /* keep prior list */
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!session?.user) return;
    void loadHistory();
  }, [session?.user, loadHistory]);

  const onSubmit = async () => {
    if (!hasSource || isLoading) return;
    setIsLoading(true);
    setError(null);
    setStatusMessage("");

    try {
      let videoUrl = url.trim();
      if (sourceType === "upload" && file) {
        setStatusMessage("Enviando vídeo…");
        videoUrl = await uploadVideoFile(file);
      }

      setStatusMessage(
        sourceType === "youtube" ? "Verificando vídeo…" : "Iniciando transcrição…",
      );
      const response = await fetch("/api/tasks/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source: { url: videoUrl, title: null },
          job_kind: "transcribe",
          language: "pt",
          processing_mode: "quality",
        }),
      });

      if (!response.ok) {
        const info = await parseApiError(response, `API error: ${response.status}`);
        throw new Error(formatSupportMessage(info));
      }

      const result = await response.json();
      router.push(`/transcribe/${result.task_id}`);
    } catch (submitError) {
      setError(
        submitError instanceof Error
          ? submitError.message
          : "Falha ao iniciar a transcrição.",
      );
      setIsLoading(false);
      setStatusMessage("");
    }
  };

  if (isPending || !session?.user) {
    return (
      <div className="mx-auto w-full max-w-2xl space-y-4 p-6">
        <Skeleton className="h-9 w-56" />
        <Skeleton className="h-36 w-full rounded-2xl" />
      </div>
    );
  }

  return (
    <div className="mx-auto w-full max-w-2xl space-y-10 px-4 py-8 sm:px-6">
      <header className="space-y-2">
        <p className="text-sm font-medium text-brand">Transcrever</p>
        <h1 className="font-display text-3xl font-semibold tracking-tight">
          Transcrição completa
        </h1>
        <p className="max-w-xl text-sm text-muted-foreground">
          YouTube ou upload → texto em alta qualidade (AssemblyAI), sem cortes.
          Idioma: português.
        </p>
      </header>

      <section className="space-y-4 rounded-2xl border bg-background p-4 sm:p-5">
        <div
          className={cn(
            "relative overflow-hidden rounded-xl border border-dashed p-4 transition-colors",
            file ? "border-brand/40 bg-brand/5" : "border-border",
          )}
          onDragOver={(event) => event.preventDefault()}
          onDrop={(event) => {
            event.preventDefault();
            const dropped = event.dataTransfer.files?.[0];
            if (dropped?.type.startsWith("video/")) {
              setFile(dropped);
              setUrl("");
              setError(null);
            }
          }}
        >
          <input
            id="transcribe-upload"
            type="file"
            accept="video/*"
            className="sr-only"
            onChange={(event) => {
              const next = event.target.files?.[0] || null;
              setFile(next);
              if (next) setUrl("");
              setError(null);
            }}
          />

          {file ? (
            <div className="flex items-center gap-3">
              <Upload className="size-5 text-brand" />
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium">{file.name}</p>
                <p className="text-xs text-muted-foreground">
                  {(file.size / (1024 * 1024)).toFixed(1)} MB
                </p>
              </div>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={() => setFile(null)}
              >
                Remover
              </Button>
            </div>
          ) : (
            <div className="space-y-3">
              <label
                htmlFor="transcribe-upload"
                className="flex cursor-pointer items-center gap-2 text-sm text-muted-foreground hover:text-foreground"
              >
                <Upload className="size-4" />
                Arraste um vídeo ou clique para enviar
              </label>
              <div className="flex items-center gap-2">
                <Youtube className="size-4 shrink-0 text-muted-foreground" />
                <Input
                  id="transcribe-youtube"
                  value={url}
                  onChange={(event) => {
                    setUrl(event.target.value);
                    setFile(null);
                    setError(null);
                  }}
                  placeholder="Cole a URL do YouTube"
                  disabled={isLoading}
                />
              </div>
              {youtubeThumbnailUrl ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={youtubeThumbnailUrl}
                  alt=""
                  className="h-20 w-[120px] rounded-lg object-cover"
                />
              ) : null}
            </div>
          )}
        </div>

        {error ? (
          <p className="text-sm text-red-600 dark:text-red-400" role="alert">
            {error}
          </p>
        ) : null}
        {statusMessage ? (
          <p className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="size-3.5 animate-spin" />
            {statusMessage}
          </p>
        ) : null}

        <Button
          type="button"
          className="w-full sm:w-auto"
          disabled={!hasSource || isLoading}
          onClick={() => void onSubmit()}
        >
          {isLoading ? (
            <>
              <Loader2 className="size-4 animate-spin" />
              Iniciando…
            </>
          ) : (
            <>
              <FileText className="size-4" />
              Transcrever
            </>
          )}
        </Button>
      </section>

      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">Recentes</h2>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => {
              setHistoryLoading(true);
              void loadHistory();
            }}
          >
            Atualizar
          </Button>
        </div>
        {historyLoading ? (
          <PageLoading />
        ) : history.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            Nenhuma transcrição ainda.
          </p>
        ) : (
          <ul className="divide-y rounded-xl border">
            {history.map((task) => {
              const active = ACTIVE_TASK_STATUSES.includes(task.status);
              return (
                <li key={task.id}>
                  <Link
                    href={`/transcribe/${task.id}`}
                    className="flex items-center gap-3 px-3 py-3 transition-colors hover:bg-muted/40"
                  >
                    <FileText className="size-4 shrink-0 text-muted-foreground" />
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-medium">
                        {task.source_title || "Sem título"}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {timeAgo(task.created_at)}
                        {task.transcript_char_count
                          ? ` · ${task.transcript_char_count.toLocaleString()} chars`
                          : ""}
                        {active ? " · em andamento" : ""}
                      </p>
                    </div>
                    <StatusBadge status={task.status} />
                  </Link>
                </li>
              );
            })}
          </ul>
        )}
      </section>
    </div>
  );
}
