import { create } from "zustand";
import type { Lang } from "./i18n";

// Verdict/walk state for the blind H3 labeling view. SEPARATE from useViewer (display toggles) so a
// labeling session never leaks into the public viewer. The blind invariant is enforced upstream (the
// `answers/` data is fetched only after a verdict locks), so this store only tracks the walk + votes.

export type Verdict = "yes" | "no" | "skip";
export type LabelTask = { task_id: number; query_id: string; scene_id: string };
export type LabelQuery = { id: string; nl: string; rationale: string; scope: string };
export type Pool = {
  pool_id: string;
  sealed_at: string;
  is_pilot: boolean;
  scoring_policy: string;
  honest_scope: string;
  queries: LabelQuery[];
  tasks: LabelTask[];
};
export type VerdictRecord = {
  scene_id: string;
  query_id: string;
  verdict: Verdict;
  frame_seen: number;
  timestamp: string;
  session_id: string;
};

function newSession(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `sess-${Math.floor(performance.now())}-${Math.round(Math.random() * 1e9)}`;
}

// Verdicts persist to localStorage on EVERY change, keyed by pool, so a reload / navigate / crash
// never loses labeling work (the prior memory-only store lost a full pass). "data changed => saved"
// is a property of the store, not a "remember to click Save" checklist.
const vkey = (poolId: string) => `occquery-verdicts-${poolId}`;

function persistVerdicts(poolId: string, verdicts: Record<number, VerdictRecord>): void {
  if (typeof window !== "undefined") window.localStorage.setItem(vkey(poolId), JSON.stringify(verdicts));
}

function loadVerdicts(poolId: string): Record<number, VerdictRecord> {
  if (typeof window === "undefined") return {};
  try {
    return JSON.parse(window.localStorage.getItem(vkey(poolId)) ?? "{}");
  } catch {
    return {};
  }
}

type LabelStore = {
  pool: Pool | null;
  sessionId: string;
  current: number; // index into pool.tasks
  verdicts: Record<number, VerdictRecord>; // keyed by task_id
  revealed: boolean; // current task's verdict is locked -> answers may be shown (QA, non-binding)
  lang: Lang; // labeler display language (en canonical / ko translation); not part of the seal
  setLang: (lang: Lang) => void;
  setPool: (p: Pool) => void;
  vote: (verdict: Verdict, frameSeen: number) => void;
  undo: () => void;
  go: (idx: number) => void;
  next: () => void;
  prev: () => void;
};

export const useLabel = create<LabelStore>((set, get) => ({
  pool: null,
  sessionId: newSession(),
  current: 0,
  verdicts: {},
  revealed: false,
  lang: "en", // SSR-safe default; the labeler restores the persisted choice on mount
  setLang: (lang) => {
    if (typeof window !== "undefined") window.localStorage.setItem("occquery-label-lang", lang);
    set({ lang });
  },
  setPool: (p) => {
    // rehydrate any prior verdicts for this pool, and resume at the first un-voted task
    const restored = loadVerdicts(p.pool_id);
    const firstOpen = p.tasks.findIndex((t) => !restored[t.task_id]);
    const current = firstOpen === -1 ? 0 : firstOpen;
    set({ pool: p, current, verdicts: restored, revealed: Boolean(restored[p.tasks[current]?.task_id]) });
  },
  vote: (verdict, frameSeen) => {
    const { pool, current, sessionId } = get();
    if (!pool) return;
    const task = pool.tasks[current];
    const rec: VerdictRecord = {
      scene_id: task.scene_id,
      query_id: task.query_id,
      verdict,
      frame_seen: frameSeen,
      timestamp: new Date().toISOString(),
      session_id: sessionId,
    };
    set((s) => {
      const verdicts = { ...s.verdicts, [task.task_id]: rec };
      persistVerdicts(pool.pool_id, verdicts);
      return { verdicts, revealed: true };
    });
  },
  undo: () => {
    const { pool, current } = get();
    if (!pool) return;
    const task = pool.tasks[current];
    set((s) => {
      const v = { ...s.verdicts };
      delete v[task.task_id];
      persistVerdicts(pool.pool_id, v);
      return { verdicts: v, revealed: false };
    });
  },
  go: (idx) =>
    set((s) => {
      if (!s.pool) return {};
      const i = Math.max(0, Math.min(idx, s.pool.tasks.length - 1));
      const task = s.pool.tasks[i];
      return { current: i, revealed: Boolean(s.verdicts[task.task_id]) };
    }),
  next: () => get().go(get().current + 1),
  prev: () => get().go(get().current - 1),
}));
