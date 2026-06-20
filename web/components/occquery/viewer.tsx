"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type * as THREE from "three";
import { CLASS_NAMES, SEMANTIC_COLORS, Scene3D, type Obstacle } from "./scene3d";
import { ControlPanel } from "./controls";
import { useViewer } from "./store";

type Predicates = {
  ego_width: number;
  min_free_width: number | null;
  lateral_clearance: number | null;
  free_path_blocked: boolean;
};
type FrameMeta = { t: number; speed: number; n_obstacles_band: number; n_obstacles_total: number; predicates: Predicates };
type SceneMeta = { scene: string; voxel_size: number; ego: { width: number; length: number; height: number }; n_frames: number; frames: FrameMeta[] };

const BASE = "/data/occquery";

function Row({ k, v, hot }: { k: string; v: string; hot?: boolean }) {
  return (
    <div className="flex justify-between gap-4">
      <span className="text-muted-foreground">{k}</span>
      <span className={hot ? "text-red-400" : ""}>{v}</span>
    </div>
  );
}

function Legend({ obstacles }: { obstacles: Obstacle[] }) {
  const present = Array.from(new Set(obstacles.map((o) => o[3]))).sort((a, b) => a - b);
  if (!present.length) return null;
  return (
    <div className="mb-3 rounded-md border p-2">
      <div className="mb-1 text-[10px] uppercase tracking-wide text-muted-foreground">classes (box-only sees ~vehicles + peds; occquery sees all)</div>
      <div className="grid grid-cols-2 gap-x-2 gap-y-0.5 text-[11px]">
        {present.map((c) => (
          <div key={c} className="flex items-center gap-1.5">
            <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: SEMANTIC_COLORS[c] ?? "#9ca3af" }} />
            <span className="text-muted-foreground">{CLASS_NAMES[c] ?? `class ${c}`}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function OccqueryViewer() {
  const [scenes, setScenes] = useState<string[]>([]);
  const [scene, setScene] = useState("scene-0061");
  const [meta, setMeta] = useState<SceneMeta | null>(null);
  const [frameIdx, setFrameIdx] = useState(0);
  const [obstacles, setObstacles] = useState<Obstacle[]>([]);
  const [copied, setCopied] = useState(false);
  const glRef = useRef<THREE.WebGLRenderer | null>(null);
  const cache = useRef<Map<string, Obstacle[]>>(new Map());

  const playing = useViewer((s) => s.playing);
  const speed = useViewer((s) => s.speed);
  const loop = useViewer((s) => s.loop);
  const set = useViewer((s) => s.set);
  const reset = useViewer((s) => s.reset);
  const colorMode = useViewer((s) => s.colorMode);

  useEffect(() => {
    fetch(`${BASE}/index.json`).then((r) => r.json()).then((d) => setScenes(d.scenes.map((s: { scene: string }) => s.scene))).catch(() => {});
  }, []);

  useEffect(() => {
    setMeta(null);
    setFrameIdx(0);
    cache.current.clear();
    fetch(`${BASE}/${scene}.json`).then((r) => r.json()).then(setMeta).catch(() => {});
  }, [scene]);

  useEffect(() => {
    if (!meta) return;
    const t = meta.frames[frameIdx]?.t;
    if (t === undefined) return;
    const load = (tt: number): Promise<Obstacle[]> => {
      const key = `${scene}/${tt}`;
      const hit = cache.current.get(key);
      if (hit) return Promise.resolve(hit);
      return fetch(`${BASE}/${scene}/f${tt}.json`).then((r) => r.json()).then((d) => {
        cache.current.set(key, d.obstacles);
        return d.obstacles as Obstacle[];
      });
    };
    load(t).then(setObstacles).catch(() => setObstacles([]));
    const nt = meta.frames[frameIdx + 1]?.t;
    if (nt !== undefined) load(nt).catch(() => {});
  }, [meta, frameIdx, scene]);

  // playback loop
  useEffect(() => {
    if (!playing || !meta) return;
    const id = setInterval(() => {
      setFrameIdx((i) => {
        const next = i + 1;
        if (next >= meta.n_frames) {
          if (loop) return 0;
          set("playing", false);
          return i;
        }
        return next;
      });
    }, 500 / speed);
    return () => clearInterval(id);
  }, [playing, speed, loop, meta, set]);

  // keyboard shortcuts
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
      const k = e.key.toLowerCase();
      const st = useViewer.getState();
      if (e.key === " ") { e.preventDefault(); set("playing", !st.playing); }
      else if (k === "arrowright") setFrameIdx((i) => Math.min(i + 1, (meta?.n_frames ?? 1) - 1));
      else if (k === "arrowleft") setFrameIdx((i) => Math.max(i - 1, 0));
      else if (k === "o") st.toggle("showVoxels");
      else if (k === "e") st.toggle("showEgo");
      else if (k === "g") st.toggle("showGrid");
      else if (k === "r") reset();
      else if (k === "f") {
        if (document.fullscreenElement) document.exitFullscreen();
        else document.documentElement.requestFullscreen();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [meta, set, reset]);

  const copyView = useCallback(() => {
    const gl = glRef.current;
    if (!gl) return;
    gl.domElement.toBlob(async (blob) => {
      if (!blob) return;
      try {
        await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      } catch (err) {
        console.error(err);
      }
    }, "image/png");
  }, []);

  const downloadView = useCallback(() => {
    const gl = glRef.current;
    if (!gl) return;
    const a = document.createElement("a");
    a.href = gl.domElement.toDataURL("image/png");
    a.download = `${scene}_f${meta?.frames[frameIdx]?.t ?? 0}.png`;
    a.click();
  }, [scene, meta, frameIdx]);

  const fm = meta?.frames[frameIdx];
  const p = fm?.predicates;

  return (
    <div className="flex h-[calc(100vh-3.5rem)] w-full">
      <div className="relative flex-1">
        {meta && (
          <Scene3D obstacles={obstacles} ego={meta.ego} voxelSize={meta.voxel_size} onGl={(gl) => (glRef.current = gl)} />
        )}

        <div className="absolute right-4 top-4 flex gap-2">
          <button onClick={copyView} className="rounded-md bg-white/10 px-3 py-2 text-sm text-white backdrop-blur transition hover:bg-white/20">
            {copied ? "✓ copied" : "📷 copy"}
          </button>
          <button onClick={downloadView} className="rounded-md bg-white/10 px-3 py-2 text-sm text-white backdrop-blur transition hover:bg-white/20">
            ⤓ PNG
          </button>
        </div>

        {meta && fm && (
          <div className="absolute inset-x-4 bottom-4 flex items-center gap-3 rounded-lg bg-black/50 px-3 py-2 backdrop-blur">
            <button onClick={() => set("playing", !playing)} className="rounded bg-white/10 px-2.5 py-1 text-sm text-white">
              {playing ? "⏸" : "▶"}
            </button>
            <button onClick={() => setFrameIdx((i) => Math.max(0, i - 1))} className="text-white/60 hover:text-white">⏮</button>
            <input type="range" min={0} max={meta.n_frames - 1} value={frameIdx} onChange={(e) => setFrameIdx(Number(e.target.value))} className="flex-1 accent-blue-500" />
            <button onClick={() => setFrameIdx((i) => Math.min(meta.n_frames - 1, i + 1))} className="text-white/60 hover:text-white">⏭</button>
            <span className="w-14 text-right font-mono text-xs text-white/60">{fm.t}/{meta.n_frames - 1}</span>
            <select value={speed} onChange={(e) => set("speed", Number(e.target.value))} className="rounded bg-transparent text-xs text-white/60">
              {[0.25, 0.5, 1, 2, 4].map((s) => (
                <option key={s} value={s} className="bg-black">{s}x</option>
              ))}
            </select>
          </div>
        )}

        <div className="pointer-events-none absolute bottom-20 left-4 text-[11px] text-white/40">
          drag rotate · scroll zoom · space play · ←→ step · F fullscreen · R reset
        </div>
      </div>

      <aside className="flex w-80 shrink-0 flex-col overflow-y-auto border-l bg-background p-3 text-sm">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-base font-semibold">occquery 3D</h2>
          <button onClick={reset} className="text-xs text-muted-foreground hover:text-foreground">reset</button>
        </div>

        <select value={scene} onChange={(e) => setScene(e.target.value)} className="mb-3 w-full rounded border bg-transparent px-2 py-1 text-sm">
          {scenes.map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>

        {meta && fm && p && (
          <div className="mb-3 space-y-1 rounded-md border p-2 font-mono text-xs">
            <Row k="speed" v={`${fm.speed} m/s`} />
            <Row k="obstacles" v={`${fm.n_obstacles_band}`} />
            <Row k="min_free_width" v={p.min_free_width === null ? "none" : `${p.min_free_width} m`} />
            <Row k="lateral_clearance" v={p.lateral_clearance === null ? "none" : `${p.lateral_clearance} m`} />
            <Row k="free_path_blocked" v={p.free_path_blocked ? "TRUE" : "false"} hot={p.free_path_blocked} />
          </div>
        )}

        {colorMode === "semantic" && <Legend obstacles={obstacles} />}

        <ControlPanel />

        <p className="mt-3 text-[10px] leading-snug text-muted-foreground">
          measurements only — &quot;danger&quot; (lead car vs wall) is the dynfield layer. occquery shows what
          box-only can&apos;t: free-space geometry.
        </p>
      </aside>
    </div>
  );
}
