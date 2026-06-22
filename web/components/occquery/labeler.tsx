"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { ArrowLeft, Check, X, SkipForward, Undo2, Save, Lock } from "lucide-react";
import type * as THREE from "three";
import { Scene3D, type Obstacle, type ReachableField } from "./scene3d";
import { GlassPanel } from "./glass";
import { useViewer } from "./store";
import { useLabel, type Pool, type Verdict } from "./label-store";
import { UI, QUERY_KO, GUIDE, LOOK_FOR, type Lang } from "./i18n";

const BASE = "/data/occquery/label";

type SceneMeta = {
  scene: string;
  voxel_size: number;
  ego: { width: number; length: number; height: number };
  n_frames: number;
  frames: { t: number; speed: number; n_obstacles_band: number }[];
};
type Answers = {
  scene: string;
  frames: { t: number; reachable: ReachableField; predicates: Record<string, number | boolean | null> }[];
  query_verdicts: Record<string, { retrieved: boolean; matching_frames: number[] }>;
};

export function Labeler() {
  const { pool, current, verdicts, revealed, lang, setLang, setPool, vote, undo, next, prev, sessionId } = useLabel();
  const setV = useViewer((s) => s.set);
  const applyPreset = useViewer((s) => s.applyPreset);
  const t = UI[lang];

  const [frameIdx, setFrameIdx] = useState(0);
  const [sceneMeta, setSceneMeta] = useState<SceneMeta | null>(null);
  const [obstacles, setObstacles] = useState<Obstacle[]>([]);
  const [answers, setAnswers] = useState<Answers | null>(null); // ONLY set after a verdict locks (blind)
  const [saveMsg, setSaveMsg] = useState("");
  const sceneCache = useRef<Map<string, SceneMeta>>(new Map());
  const frameCache = useRef<Map<string, Obstacle[]>>(new Map());
  const glRef = useRef<THREE.WebGLRenderer | null>(null);

  // Mount: restore language, load the sealed pool, force the BLIND display (no reachable overlay).
  useEffect(() => {
    const saved = typeof window !== "undefined" ? window.localStorage.getItem("occquery-label-lang") : null;
    if (saved === "ko" || saved === "en") setLang(saved as Lang);
    setV("showReachable", false);
    setV("showVoxels", true);
    setV("showEgo", true);
    setV("showGrid", true);
    setV("showForward", false);
    setV("colorMode", "flat"); // all obstacles ONE color so none collide with the red ego (pedestrian
    setV("playing", false);    // semantic is also red) -- the ego must be unmistakable for clearance judgment
    fetch(`${BASE}/pool.json`)
      .then((r) => r.json())
      .then((p: Pool) => setPool(p))
      .catch(() => setSaveMsg("could not load pool.json — run export_label.py first"));
  }, [setPool, setV, setLang]);

  // Land the camera top-down once the scene is mounted (the honest view for distance + beside-vs-ahead).
  // Keyed on sceneMeta so the nonce bump lands AFTER Scene3D mounts -> CameraController re-applies it.
  useEffect(() => {
    if (sceneMeta) applyPreset("bev");
  }, [sceneMeta, applyPreset]);

  const task = pool && pool.tasks[current] ? pool.tasks[current] : null;
  const query = pool && task ? pool.queries.find((q) => q.id === task.query_id) ?? null : null;
  const sceneId = task?.scene_id ?? null;
  const ko = lang === "ko" && query ? QUERY_KO[query.id] : undefined;
  const nl = ko?.nl ?? query?.nl;
  const rationale = ko?.rationale ?? query?.rationale;
  const g = GUIDE[lang];
  const lf = query ? LOOK_FOR[query.id] : undefined;

  // Task change: reset frame, drop any revealed answer, re-blind the display.
  useEffect(() => {
    if (!sceneId) return;
    setFrameIdx(0);
    setAnswers(null);
    setV("showReachable", false);
    const cached = sceneCache.current.get(sceneId);
    if (cached) {
      setSceneMeta(cached);
      return;
    }
    setSceneMeta(null);
    fetch(`${BASE}/${sceneId}.json`)
      .then((r) => r.json())
      .then((m: SceneMeta) => {
        sceneCache.current.set(sceneId, m);
        setSceneMeta(m);
      })
      .catch(() => {});
  }, [sceneId, setV]);

  // Frame obstacles (the BLIND geometry the human judges).
  useEffect(() => {
    if (!sceneId || !sceneMeta) return;
    const key = `${sceneId}/${frameIdx}`;
    const cached = frameCache.current.get(key);
    if (cached) {
      setObstacles(cached);
      return;
    }
    fetch(`${BASE}/${sceneId}/f${frameIdx}.json`)
      .then((r) => r.json())
      .then((f: { obstacles: Obstacle[] }) => {
        frameCache.current.set(key, f.obstacles);
        setObstacles(f.obstacles);
      })
      .catch(() => {});
  }, [sceneId, sceneMeta, frameIdx]);

  // BLIND ENFORCEMENT: the answers file (predicate verdict + reachable) is fetched ONLY once a verdict
  // is locked. Until then it is not in the browser at all, so it cannot bias the judgment.
  useEffect(() => {
    if (!sceneId || !revealed) {
      setAnswers(null);
      return;
    }
    fetch(`${BASE}/answers/${sceneId}.json`)
      .then((r) => r.json())
      .then((a: Answers) => {
        setAnswers(a);
        setV("showReachable", true); // QA: now it is honest to show the measured free-space field
      })
      .catch(() => {});
  }, [sceneId, revealed, setV]);

  const onVote = useCallback((v: Verdict) => vote(v, frameIdx), [vote, frameIdx]);

  const flush = useCallback(async (records: object[], manual: boolean) => {
    if (!pool) return;
    try {
      const res = await fetch("/api/labels", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ pool_id: pool.pool_id, session_id: sessionId, verdicts: records }),
      });
      const j = await res.json();
      setSaveMsg(res.ok ? `${manual ? "saved" : "auto-saved"} ${j.written ?? records.length} → ${j.path ?? "labels/"}` : `save failed: ${j.error ?? res.status}`);
    } catch (e) {
      setSaveMsg(`save failed (dev server down?): ${String(e)}`);
    }
  }, [pool, sessionId]);

  const save = useCallback(() => flush(Object.values(verdicts), true), [flush, verdicts]);

  // Auto-save to disk on every verdict change (localStorage already persisted it instantly in the
  // store; this keeps labels/<pool>.jsonl current even if the browser closes). No "remember to Save".
  const nVerdicts = Object.keys(verdicts).length;
  useEffect(() => {
    if (pool && nVerdicts > 0) flush(Object.values(verdicts), false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nVerdicts]);

  // Keyboard: arrows step frames; Y/N/S vote (blind); Enter next; U undo (revealed).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      const nf = sceneMeta?.n_frames ?? 1;
      if (e.key === "ArrowRight") setFrameIdx((i) => Math.min(i + 1, nf - 1));
      else if (e.key === "ArrowLeft") setFrameIdx((i) => Math.max(i - 1, 0));
      else if (!revealed && (e.key === "y" || e.key === "Y")) onVote("yes");
      else if (!revealed && (e.key === "n" || e.key === "N")) onVote("no");
      else if (!revealed && (e.key === "s" || e.key === "S")) onVote("skip");
      else if (revealed && e.key === "Enter") next();
      else if (revealed && (e.key === "u" || e.key === "U")) undo();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [revealed, sceneMeta, onVote, next, undo]);

  if (!pool) {
    return (
      <div className="flex h-screen items-center justify-center bg-neutral-950 text-neutral-400">
        <p className="text-sm">{saveMsg || t.loading}</p>
      </div>
    );
  }

  const fm = sceneMeta?.frames[frameIdx];
  const verdict = task ? verdicts[task.task_id] : undefined;
  const qv = answers && task ? answers.query_verdicts[task.query_id] : undefined;
  const reachable = revealed && answers ? answers.frames.find((f) => f.t === frameIdx)?.reachable ?? null : null;
  const done = Object.keys(verdicts).length;
  const langBtn = (l: Lang, label: string) => (
    <button
      onClick={() => setLang(l)}
      className={`rounded-md px-2 py-0.5 text-[10px] transition ${lang === l ? "bg-white/15 text-neutral-100" : "text-neutral-500 hover:text-neutral-300"}`}
    >
      {label}
    </button>
  );

  return (
    <div className="relative h-screen w-screen overflow-hidden bg-neutral-950">
      <div className="absolute inset-0">
        {sceneMeta && (
          <Scene3D
            obstacles={obstacles}
            ego={sceneMeta.ego}
            voxelSize={sceneMeta.voxel_size}
            reachable={reachable}
            onGl={(gl) => (glRef.current = gl)}
          />
        )}
      </div>

      {/* top-left: home + lang toggle + progress + honest-scope */}
      <div className="absolute left-4 top-4 w-[22rem]">
        <GlassPanel className="p-4">
          <div className="flex items-center justify-between">
            <Link href="/occquery" className="flex items-center gap-1.5 text-xs text-neutral-400 hover:text-neutral-200">
              <ArrowLeft className="h-3.5 w-3.5" /> occquery
            </Link>
            <div className="flex items-center gap-2">
              <div className="flex items-center rounded-lg border border-white/10 p-0.5">
                {langBtn("en", "EN")}
                {langBtn("ko", "한국어")}
              </div>
              <span className="rounded-full border border-white/10 px-2 py-0.5 text-[10px] uppercase tracking-wide text-neutral-400">
                {pool.is_pilot ? "pilot" : pool.pool_id} · {pool.scoring_policy}
              </span>
            </div>
          </div>
          <div className="mt-3 flex items-baseline justify-between">
            <h1 className="text-sm font-medium text-neutral-100">{t.title}</h1>
            <span className="text-xs tabular-nums text-neutral-400">{t.progress(current + 1, pool.tasks.length, done)}</span>
          </div>
          <p className="mt-2 text-[11px] leading-relaxed text-neutral-500">{t.banner}</p>
        </GlassPanel>
      </div>

      {/* right: the query + verdict */}
      <div className="absolute right-4 top-4 bottom-4 flex w-[24rem] flex-col gap-3">
        <GlassPanel className="p-4">
          <p className="text-[10px] uppercase tracking-wide text-neutral-500">{t.scope(query?.scope)}</p>
          <p className="mt-1.5 text-[15px] font-medium leading-snug text-neutral-100">{nl}</p>
          <p className="mt-2 max-h-28 overflow-auto text-[11px] leading-relaxed text-neutral-400">{rationale}</p>
        </GlassPanel>

        {/* frame stepper */}
        <GlassPanel className="p-4">
          <div className="flex items-center justify-between text-xs text-neutral-400">
            <span className="tabular-nums">{t.frame(frameIdx + 1, sceneMeta?.n_frames ?? "…")}</span>
            <span className="tabular-nums">{fm ? t.frameStat(fm.speed, fm.n_obstacles_band) : ""}</span>
          </div>
          <input
            type="range"
            min={0}
            max={(sceneMeta?.n_frames ?? 1) - 1}
            value={frameIdx}
            onChange={(e) => setFrameIdx(Number(e.target.value))}
            className="mt-2 w-full accent-neutral-300"
          />
          <p className="mt-1 text-[10px] text-neutral-600">{sceneId ? t.frameHint(sceneId) : ""}</p>
          <div className="mt-2 grid grid-cols-4 gap-1">
            {([["bev", g.views.top], ["iso", g.views.iso], ["side", g.views.side], ["front", g.views.front]] as const).map(
              ([id, label]) => (
                <button
                  key={id}
                  onClick={() => applyPreset(id)}
                  className="rounded-md border border-white/10 py-1 text-[10px] text-neutral-400 transition hover:bg-white/10 hover:text-neutral-200"
                >
                  {label}
                </button>
              ),
            )}
          </div>
        </GlassPanel>

        {/* verdict / reveal */}
        <GlassPanel className="flex-1 p-4">
          {!revealed ? (
            <div className="flex h-full flex-col">
              <p className="text-[10px] uppercase tracking-wide text-neutral-500">{g.lookForLabel}</p>
              <p className="mt-1 text-xs leading-relaxed text-neutral-100">{lf ? lf[lang] : t.prompt}</p>
              {lf?.speedGateKmh && (
                <p className="mt-2 rounded-lg border border-amber-400/15 bg-amber-400/[0.05] p-2 text-[10px] leading-relaxed text-amber-200/75">
                  {g.speedAuto(lf.speedGateKmh)}
                </p>
              )}
              <p className="mt-3 text-[11px] leading-relaxed text-neutral-400">{g.rule}</p>
              <p className="mt-1 text-[10px] leading-relaxed text-neutral-600">{g.coarse}</p>
              <div className="mt-4 grid grid-cols-3 gap-2">
                <button
                  onClick={() => onVote("yes")}
                  className="flex flex-col items-center gap-1 rounded-xl border border-white/10 bg-white/[0.04] py-3 text-xs text-neutral-200 transition hover:bg-white/10"
                >
                  <Check className="h-4 w-4" /> {t.yes} <span className="text-[9px] text-neutral-500">Y</span>
                </button>
                <button
                  onClick={() => onVote("no")}
                  className="flex flex-col items-center gap-1 rounded-xl border border-white/10 bg-white/[0.04] py-3 text-xs text-neutral-200 transition hover:bg-white/10"
                >
                  <X className="h-4 w-4" /> {t.no} <span className="text-[9px] text-neutral-500">N</span>
                </button>
                <button
                  onClick={() => onVote("skip")}
                  className="flex flex-col items-center gap-1 rounded-xl border border-white/10 bg-white/[0.04] py-3 text-xs text-neutral-400 transition hover:bg-white/10"
                >
                  <SkipForward className="h-4 w-4" /> {t.skip} <span className="text-[9px] text-neutral-500">S</span>
                </button>
              </div>
              <div className="mt-auto flex items-center gap-1.5 pt-3 text-[10px] text-neutral-600">
                <Lock className="h-3 w-3" /> {t.hidden}
              </div>
            </div>
          ) : (
            <div className="flex h-full flex-col">
              <p className="text-xs text-neutral-400">
                {t.youVoted} <span className="font-medium text-neutral-100">{verdict?.verdict.toUpperCase()}</span>
              </p>
              {qv && (
                <div className="mt-3 rounded-xl border border-white/10 bg-white/[0.03] p-3">
                  <p className="text-[11px] text-neutral-300">{t.predicate(qv.retrieved, qv.matching_frames.length)}</p>
                  {verdict && (
                    <p className="mt-1 text-[11px]">
                      {verdict.verdict === "skip" ? (
                        <span className="text-neutral-500">{t.skipped}</span>
                      ) : (verdict.verdict === "yes") === qv.retrieved ? (
                        <span className="text-emerald-400/80">{t.agrees}</span>
                      ) : (
                        <span className="text-amber-400/80">{t.disagrees}</span>
                      )}
                    </p>
                  )}
                  <p className="mt-2 text-[10px] text-neutral-600">{t.overlayNote}</p>
                </div>
              )}
              <div className="mt-auto grid grid-cols-2 gap-2 pt-3">
                <button
                  onClick={undo}
                  className="flex items-center justify-center gap-1.5 rounded-xl border border-white/10 py-2.5 text-xs text-neutral-400 transition hover:bg-white/10"
                >
                  <Undo2 className="h-3.5 w-3.5" /> {t.undo}
                </button>
                <button
                  onClick={next}
                  className="flex items-center justify-center gap-1.5 rounded-xl border border-white/10 bg-white/10 py-2.5 text-xs text-neutral-100 transition hover:bg-white/20"
                >
                  {t.next}
                </button>
              </div>
            </div>
          )}
        </GlassPanel>

        {/* save */}
        <GlassPanel className="p-3">
          <button
            onClick={save}
            className="flex w-full items-center justify-center gap-1.5 rounded-xl border border-white/10 py-2.5 text-xs text-neutral-200 transition hover:bg-white/10"
          >
            <Save className="h-3.5 w-3.5" /> {t.save(done)}
          </button>
          {saveMsg && <p className="mt-2 text-[10px] leading-relaxed text-neutral-500">{saveMsg}</p>}
        </GlassPanel>
      </div>
    </div>
  );
}
