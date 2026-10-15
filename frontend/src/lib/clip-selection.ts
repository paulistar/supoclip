export const CLIP_BRIEF_MAX_CHARS = 800;
export const CLIP_PREFS_STORAGE_KEY = "supoclip.clipSelectionPrefs.v1";

export type DurationPresetId =
  | "auto"
  | "short"
  | "long"
  | "viral"
  | "monetization"
  | "podcast";

export type MinClipsChoice = "auto" | 5 | 10 | 15 | 20;

export type DurationPreset = {
  id: DurationPresetId;
  label: string;
  hint: string;
  minSeconds: number;
  maxSeconds: number;
  idealMin: number;
  idealMax: number;
  spacing: number;
  suggestMinClips: MinClipsChoice;
};

export const DURATION_PRESETS: DurationPreset[] = [
  {
    id: "auto",
    label: "Auto (20-60s)",
    hint: "Balanced short-form clips",
    minSeconds: 20,
    maxSeconds: 60,
    idealMin: 25,
    idealMax: 50,
    spacing: 1.1,
    suggestMinClips: "auto",
  },
  {
    id: "short",
    label: "Curtos (15-30s)",
    hint: "Fast hooks for Reels/TikTok",
    minSeconds: 15,
    maxSeconds: 30,
    idealMin: 20,
    idealMax: 28,
    spacing: 1.0,
    suggestMinClips: 10,
  },
  {
    id: "long",
    label: "Longos (45-90s)",
    hint: "Room for setup and payoff",
    minSeconds: 45,
    maxSeconds: 90,
    idealMin: 50,
    idealMax: 75,
    spacing: 1.3,
    suggestMinClips: 10,
  },
  {
    id: "viral",
    label: "Virais (2-3 min)",
    hint: "Story-style social clips",
    minSeconds: 120,
    maxSeconds: 180,
    idealMin: 135,
    idealMax: 165,
    spacing: 1.6,
    suggestMinClips: 5,
  },
  {
    id: "monetization",
    label: "Monetização (3-5 min)",
    hint: "Longer ads and offer explainers",
    minSeconds: 180,
    maxSeconds: 300,
    idealMin: 210,
    idealMax: 270,
    spacing: 1.8,
    suggestMinClips: 5,
  },
  {
    id: "podcast",
    label: "Podcast (9-16 min)",
    hint: "Deep cuts and episode chapters",
    minSeconds: 540,
    maxSeconds: 960,
    idealMin: 600,
    idealMax: 840,
    spacing: 2.0,
    suggestMinClips: 5,
  },
];

export const MIN_CLIPS_OPTIONS: MinClipsChoice[] = ["auto", 5, 10, 15, 20];

export function getDurationPreset(id: DurationPresetId): DurationPreset {
  return DURATION_PRESETS.find((preset) => preset.id === id) ?? DURATION_PRESETS[0];
}

export function physicalMaxClips(sourceSeconds: number | null | undefined, presetId: DurationPresetId): number | null {
  if (!sourceSeconds || sourceSeconds <= 0) return null;
  const preset = getDurationPreset(presetId);
  return Math.max(1, Math.floor(sourceSeconds / preset.minSeconds));
}

export function presetIsFeasible(sourceSeconds: number | null | undefined, presetId: DurationPresetId): boolean {
  if (!sourceSeconds || sourceSeconds <= 0) return true;
  return sourceSeconds >= getDurationPreset(presetId).minSeconds;
}

export function clampMinClipsChoice(
  choice: MinClipsChoice,
  sourceSeconds: number | null | undefined,
  presetId: DurationPresetId,
): MinClipsChoice {
  if (choice === "auto") return "auto";
  const fit = physicalMaxClips(sourceSeconds, presetId);
  if (fit == null) return choice;
  if (choice <= fit) return choice;
  const allowed = ([5, 10, 15, 20] as const).filter((value) => value <= fit);
  return allowed.length ? allowed[allowed.length - 1] : "auto";
}

export function formatCapacityHint(
  sourceSeconds: number | null | undefined,
  presetId: DurationPresetId,
): string | null {
  const fit = physicalMaxClips(sourceSeconds, presetId);
  if (fit == null) return null;
  const minutes = Math.max(1, Math.round(sourceSeconds! / 60));
  return `Neste vídeo (~${minutes} min) cabem no máximo ~${fit} cortes neste preset.`;
}

export type StoredClipPrefs = {
  clipDurationPreset: DurationPresetId;
  minClips: MinClipsChoice;
  clipBrief: string;
};

export function loadStoredClipPrefs(): StoredClipPrefs | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(CLIP_PREFS_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<StoredClipPrefs>;
    const preset = DURATION_PRESETS.some((item) => item.id === parsed.clipDurationPreset)
      ? (parsed.clipDurationPreset as DurationPresetId)
      : "auto";
    const minClips = MIN_CLIPS_OPTIONS.includes(parsed.minClips as MinClipsChoice)
      ? (parsed.minClips as MinClipsChoice)
      : "auto";
    const clipBrief = typeof parsed.clipBrief === "string" ? parsed.clipBrief.slice(0, CLIP_BRIEF_MAX_CHARS) : "";
    return { clipDurationPreset: preset, minClips, clipBrief };
  } catch {
    return null;
  }
}

export function saveStoredClipPrefs(prefs: StoredClipPrefs) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(CLIP_PREFS_STORAGE_KEY, JSON.stringify(prefs));
  } catch {
    // ignore quota / private mode
  }
}

export function normalizeClipBrief(value: string): string {
  return value.replace(/\s+/g, " ").trim().slice(0, CLIP_BRIEF_MAX_CHARS);
}
