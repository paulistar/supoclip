"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Check, Loader2, Pause, Pencil, Play, Users } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { formatSupportMessage, parseApiError } from "@/lib/api-error";
import { cn } from "@/lib/utils";

export type SpeakerSummary = {
  id: string;
  label: string;
  utterance_count: number;
  total_duration_ms: number;
  sample_start_ms: number;
  sample_end_ms: number;
  sample_duration_ms?: number;
  sample_text?: string;
};

type SpeakersResponse = {
  speakers: SpeakerSummary[];
  speaker_map: Record<string, string>;
  has_preview_media?: boolean;
};

type SpeakerReviewProps = {
  taskId: string;
  onNamesSaved?: () => void;
  onError?: (message: string) => void;
};

function formatDuration(ms: number): string {
  const totalSec = Math.max(0, Math.round(ms / 1000));
  const minutes = Math.floor(totalSec / 60);
  const seconds = totalSec % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

export function SpeakerReview({
  taskId,
  onNamesSaved,
  onError,
}: SpeakerReviewProps) {
  const [speakers, setSpeakers] = useState<SpeakerSummary[]>([]);
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [editingId, setEditingId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [playingId, setPlayingId] = useState<string | null>(null);
  const [hasPreview, setHasPreview] = useState(true);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const stopTimerRef = useRef<number | null>(null);

  const stopPlayback = useCallback(() => {
    if (stopTimerRef.current != null) {
      window.clearTimeout(stopTimerRef.current);
      stopTimerRef.current = null;
    }
    const audio = audioRef.current;
    if (audio) {
      audio.pause();
    }
    setPlayingId(null);
  }, []);

  const loadSpeakers = useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetch(`/api/tasks/${taskId}/speakers`, {
        cache: "no-store",
      });
      if (!response.ok) {
        const info = await parseApiError(response, "Falha ao carregar falantes");
        throw new Error(formatSupportMessage(info));
      }
      const data = (await response.json()) as SpeakersResponse;
      const list = Array.isArray(data.speakers) ? data.speakers : [];
      setSpeakers(list);
      setHasPreview(data.has_preview_media !== false);
      const nextDrafts: Record<string, string> = {};
      for (const speaker of list) {
        nextDrafts[speaker.id] =
          data.speaker_map?.[speaker.id] || speaker.label || `Pessoa`;
      }
      setDrafts(nextDrafts);
    } catch (error) {
      onError?.(
        error instanceof Error ? error.message : "Falha ao carregar falantes",
      );
    } finally {
      setLoading(false);
    }
  }, [taskId, onError]);

  useEffect(() => {
    void loadSpeakers();
    return () => {
      stopPlayback();
      if (audioRef.current) {
        audioRef.current.src = "";
        audioRef.current = null;
      }
    };
  }, [loadSpeakers, stopPlayback]);

  const ensureAudio = useCallback(() => {
    if (!audioRef.current) {
      audioRef.current = new Audio(`/api/tasks/${taskId}/source-media`);
      audioRef.current.preload = "auto";
    }
    return audioRef.current;
  }, [taskId]);

  const playSample = async (speaker: SpeakerSummary) => {
    if (playingId === speaker.id) {
      stopPlayback();
      return;
    }
    stopPlayback();
    if (!hasPreview) {
      onError?.("Áudio de prévia indisponível para esta transcrição.");
      return;
    }
    try {
      const audio = ensureAudio();
      const startSec = Math.max(0, speaker.sample_start_ms / 1000);
      const endSec = Math.max(
        startSec + 0.5,
        (speaker.sample_end_ms || speaker.sample_start_ms + 3000) / 1000,
      );
      const durationMs = Math.max(500, (endSec - startSec) * 1000);

      const startWhenReady = () => {
        audio.currentTime = startSec;
        void audio.play().then(() => {
          setPlayingId(speaker.id);
          stopTimerRef.current = window.setTimeout(() => {
            stopPlayback();
          }, durationMs);
        });
      };

      if (audio.readyState >= 1) {
        startWhenReady();
      } else {
        audio.onloadedmetadata = () => {
          audio.onloadedmetadata = null;
          startWhenReady();
        };
        audio.load();
      }
    } catch (error) {
      setPlayingId(null);
      onError?.(
        error instanceof Error ? error.message : "Não foi possível tocar o áudio",
      );
    }
  };

  const saveNames = async () => {
    setSaving(true);
    stopPlayback();
    try {
      const response = await fetch(`/api/tasks/${taskId}/speakers`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ speakers: drafts }),
      });
      if (!response.ok) {
        const info = await parseApiError(response, "Falha ao salvar nomes");
        throw new Error(formatSupportMessage(info));
      }
      const data = (await response.json()) as SpeakersResponse;
      const list = Array.isArray(data.speakers) ? data.speakers : [];
      setSpeakers(list);
      setEditingId(null);
      onNamesSaved?.();
    } catch (error) {
      onError?.(
        error instanceof Error ? error.message : "Falha ao salvar nomes",
      );
    } finally {
      setSaving(false);
    }
  };

  const dirty =
    speakers.length > 0 &&
    speakers.some((speaker) => (drafts[speaker.id] || "").trim() !== speaker.label);

  if (loading) {
    return (
      <div className="flex items-center gap-2 rounded-xl border p-4 text-sm text-muted-foreground">
        <Loader2 className="size-4 animate-spin" />
        Identificando pessoas no áudio…
      </div>
    );
  }

  if (speakers.length === 0) {
    return (
      <div className="rounded-xl border p-4 text-sm text-muted-foreground">
        Nenhum falante identificado automaticamente. O texto abaixo ainda pode ser
        baixado normalmente.
      </div>
    );
  }

  return (
    <section className="space-y-3 rounded-xl border p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="space-y-1">
          <h2 className="flex items-center gap-2 text-sm font-semibold">
            <Users className="size-4 text-brand" />
            Pessoas no áudio
          </h2>
          <p className="text-xs text-muted-foreground">
            {speakers.length === 1
              ? "1 pessoa detectada. Ouça o trecho e confirme o nome."
              : `${speakers.length} pessoas detectadas. Ouça cada trecho e nomeie antes de salvar.`}
          </p>
        </div>
        <Button
          type="button"
          size="sm"
          disabled={saving || !dirty}
          onClick={() => void saveNames()}
        >
          {saving ? (
            <Loader2 className="size-4 animate-spin" />
          ) : (
            <Check className="size-4" />
          )}
          Salvar nomes
        </Button>
      </div>

      <ul className="divide-y rounded-lg border">
        {speakers.map((speaker) => {
          const isEditing = editingId === speaker.id;
          const isPlaying = playingId === speaker.id;
          return (
            <li
              key={speaker.id}
              className="flex flex-col gap-3 px-3 py-3 sm:flex-row sm:items-center"
            >
              <div className="flex min-w-0 flex-1 items-center gap-3">
                <Button
                  type="button"
                  variant="outline"
                  size="icon"
                  className="size-9 shrink-0"
                  aria-label={
                    isPlaying
                      ? `Pausar ${drafts[speaker.id] || speaker.label}`
                      : `Ouvir ${drafts[speaker.id] || speaker.label}`
                  }
                  onClick={() => void playSample(speaker)}
                >
                  {isPlaying ? (
                    <Pause className="size-4" />
                  ) : (
                    <Play className="size-4" />
                  )}
                </Button>

                <div className="min-w-0 flex-1 space-y-1">
                  {isEditing ? (
                    <Input
                      value={drafts[speaker.id] || ""}
                      autoFocus
                      onChange={(event) =>
                        setDrafts((prev) => ({
                          ...prev,
                          [speaker.id]: event.target.value,
                        }))
                      }
                      onBlur={() => setEditingId(null)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" || event.key === "Escape") {
                          setEditingId(null);
                        }
                      }}
                      placeholder="Nome da pessoa"
                      className="h-8"
                    />
                  ) : (
                    <button
                      type="button"
                      className={cn(
                        "flex max-w-full items-center gap-2 rounded-md text-left text-sm font-medium",
                        "hover:text-brand",
                      )}
                      onClick={() => setEditingId(speaker.id)}
                    >
                      <span className="truncate">
                        {drafts[speaker.id] || speaker.label}
                      </span>
                      <Pencil className="size-3.5 shrink-0 text-muted-foreground" />
                    </button>
                  )}
                  <p className="truncate text-xs text-muted-foreground">
                    {speaker.utterance_count} fala
                    {speaker.utterance_count === 1 ? "" : "s"} ·{" "}
                    {formatDuration(speaker.total_duration_ms)}
                    {speaker.sample_text
                      ? ` · “${speaker.sample_text.slice(0, 80)}${
                          speaker.sample_text.length > 80 ? "…" : ""
                        }”`
                      : ""}
                  </p>
                </div>
              </div>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
