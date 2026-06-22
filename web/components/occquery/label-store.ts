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
  setPool: (p) => set({ pool: p, current: 0, verdicts: {}, revealed: false }),
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
    set((s) => ({ verdicts: { ...s.verdicts, [task.task_id]: rec }, revealed: true }));
  },
  undo: () => {
    const { pool, current } = get();
    if (!pool) return;
    const task = pool.tasks[current];
    set((s) => {
      const v = { ...s.verdicts };
      delete v[task.task_id];
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
