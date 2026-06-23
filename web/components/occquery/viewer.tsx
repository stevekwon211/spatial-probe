"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type * as THREE from "three";
import Link from "next/link";
import { Camera, Check, ChevronLeft, Download, Pause, Play, RotateCcw, SkipBack, SkipForward, Sparkles } from "lucide-react";
import { CLASS_NAMES, SEMANTIC_COLORS, Scene3D, type LidarPoint, type Obstacle, type ReachableField } from "./scene3d";
import { ControlPanel } from "./controls";
import { useViewer, type RenderMode } from "./store";
import { GlassPanel } from "./glass";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { cn } from "@/lib/utils";

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
      <span className="text-white/40">{k}</span>
      <span className={hot ? "text-red-400" : "text-white/80"}>{v}</span>
    </div>
  );
}

function Legend({ obstacles }: { obstacles: Obstacle[] }) {
  const present = Array.from(new Set(obstacles.map((o) => o[3]))).sort((a, b) => a - b);
  if (!present.length) return null;
  return (
    <div className="mb-3 rounded-lg border border-white/10 bg-white/[0.03] p-2">
      <div className="mb-1 text-[10px] uppercase tracking-wide text-white/40">box-only sees vehicles and peds, occquery sees all</div>
      <div className="grid grid-cols-2 gap-x-2 gap-y-0.5 text-[11px]">
        {present.map((c) => (
          <div key={c} className="flex items-center gap-1.5">
            <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: SEMANTIC_COLORS[c] ?? "#9ca3af" }} />
            <span className="text-white/50">{CLASS_NAMES[c] ?? `class ${c}`}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function IconButton({ onClick, label, children }: { onClick: () => void; label: string; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      aria-label={label}
      className="flex size-8 items-center justify-center rounded-lg text-white/60 transition-colors hover:bg-white/10 hover:text-white"
    >
      {children}
    </button>
  );
}

export function OccqueryViewer() {
  const [scenes, setScenes] = useState<string[]>([]);
  const [scene, setScene] = useState("scene-0061");
  const [meta, setMeta] = useState<SceneMeta | null>(null);
  const [frameIdx, setFrameIdx] = useState(0);
  const [obstacles, setObstacles] = useState<Obstacle[]>([]);
  const [reachable, setReachable] = useState<ReachableField | null>(null);
  const [points, setPoints] = useState<LidarPoint[] | null>(null);
  const [copied, setCopied] = useState(false);
  const glRef = useRef<THREE.WebGLRenderer | null>(null);
  const cache = useRef<Map<string, { obstacles: Obstacle[]; reachable: ReachableField | null }>>(new Map());
  const lidarCache = useRef<Map<string, LidarPoint[]>>(new Map());

  const renderMode = useViewer((s) => s.renderMode);
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
    const load = (tt: number): Promise<{ obstacles: Obstacle[]; reachable: ReachableField | null }> => {
      const key = `${scene}/${tt}`;
      const hit = cache.current.get(key);
      if (hit) return Promise.resolve(hit);
      return fetch(`${BASE}/${scene}/f${tt}.json`).then((r) => r.json()).then((d) => {
        const frame = { obstacles: d.obstacles as Obstacle[], reachable: (d.reachable ?? null) as ReachableField | null };
        cache.current.set(key, frame);
        return frame;
      });
    };
    load(t).then((f) => { setObstacles(f.obstacles); setReachable(f.reachable); }).catch(() => { setObstacles([]); setReachable(null); });
    const nt = meta.frames[frameIdx + 1]?.t;
    if (nt !== undefined) load(nt).catch(() => {});
  }, [meta, frameIdx, scene]);

  // raw LiDAR point cloud — only fetched when a point-rendering mode is active
  useEffect(() => {
    if (!meta || renderMode === "voxel") { setPoints(null); return; }
    const t = meta.frames[frameIdx]?.t;
    if (t === undefined) return;
    const key = `${scene}/${t}`;
    const hit = lidarCache.current.get(key);
    if (hit) { setPoints(hit); return; }
    fetch(`${BASE}/${scene}/lidar${t}.json`).then((r) => r.json()).then((d) => {
      const pts = d.points as LidarPoint[];
      lidarCache.current.set(key, pts);
      setPoints(pts);
    }).catch(() => setPoints(null));
  }, [meta, frameIdx, scene, renderMode]);

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
    <div className="relative h-screen w-full overflow-hidden bg-[#080808]">
      {meta && <Scene3D obstacles={obstacles} ego={meta.ego} voxelSize={meta.voxel_size} reachable={reachable} points={points} onGl={(gl) => (glRef.current = gl)} />}

      {/* top-left — context + view controls */}
      <GlassPanel className="absolute top-4 bottom-4 left-4 flex w-72 flex-col text-white">
        <div className="flex items-center justify-between px-3 pt-3 pb-2">
          <Link
            href="/"
            title="back to pipeline"
            className="flex items-center gap-1 text-sm font-medium tracking-tight text-white/90 transition-colors hover:text-white"
          >
            <ChevronLeft className="size-3.5 text-white/40" />
            occquery
          </Link>
          <button onClick={reset} className="flex items-center gap-1 text-[11px] text-white/40 transition-colors hover:text-white/70">
            <RotateCcw className="size-3" />
            reset
          </button>
        </div>

        <div className="px-3 pb-2">
          <Select value={scene} onValueChange={(v) => setScene(v as string)}>
            <SelectTrigger className="w-full border-white/10 bg-white/[0.04]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {scenes.map((s) => (
                <SelectItem key={s} value={s}>{s}</SelectItem>
              ))}
            </SelectContent>
          </Select>

          {/* render mode: discretized Occ3D voxels vs the raw LiDAR scan they are built from */}
          <div className="mt-2 grid grid-cols-3 gap-1 rounded-lg border border-white/10 p-0.5">
            {(["voxel", "points", "both"] as RenderMode[]).map((m) => (
              <button
                key={m}
                onClick={() => set("renderMode", m)}
                className={cn(
                  "rounded-md py-1 text-[11px] capitalize transition-colors",
                  renderMode === m ? "bg-white/15 text-white" : "text-white/45 hover:text-white/80",
                )}
              >
                {m === "voxel" ? "voxel" : m === "points" ? "LiDAR" : "both"}
              </button>
            ))}
          </div>
        </div>

        {meta && fm && p && (
          <div className="mx-3 mb-2 space-y-1 rounded-lg bg-white/[0.04] p-2.5 font-mono text-xs">
            <Row k="speed" v={`${fm.speed} m/s`} />
            <Row k="obstacles" v={`${fm.n_obstacles_band}`} />
            <Row k="min_free_width" v={p.min_free_width === null ? "none" : `${p.min_free_width} m`} />
            <Row k="lateral_clearance" v={p.lateral_clearance === null ? "none" : `${p.lateral_clearance} m`} />
            <Row k="free_path_blocked" v={p.free_path_blocked ? "TRUE" : "false"} hot={p.free_path_blocked} />
          </div>
        )}

        <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-3">
          {colorMode === "semantic" && <Legend obstacles={obstacles} />}
          <ControlPanel />
        </div>
      </GlassPanel>

      {/* right — query / chat (summoned, placeholder for now) */}
      <GlassPanel className="absolute top-4 right-4 bottom-4 flex w-80 flex-col text-white">
        <div className="flex items-center gap-2 border-b border-white/10 px-4 py-3">
          <Sparkles className="size-4 text-white/60" />
          <span className="text-sm font-medium">Ask occquery</span>
          <span className="ml-auto rounded-full bg-white/10 px-2 py-0.5 text-[10px] tracking-wide text-white/50">soon</span>
        </div>
        <div className="flex flex-1 items-center justify-center px-6 text-center">
          <p className="text-xs leading-relaxed text-white/40">
            Ask in plain language. occquery turns the question into geometric predicates and jumps to the matching frames.
          </p>
        </div>
        <div className="border-t border-white/10 p-3">
          <div className="flex h-9 items-center rounded-xl bg-white/[0.04] px-3 text-xs text-white/30">Ask a question…</div>
        </div>
      </GlassPanel>

      {/* bottom-center — time + capture */}
      {meta && fm && (
        <GlassPanel className="absolute bottom-4 left-1/2 flex -translate-x-1/2 items-center gap-2 px-3 py-2 text-white">
          <IconButton onClick={() => set("playing", !playing)} label={playing ? "pause" : "play"}>
            {playing ? <Pause className="size-4" /> : <Play className="size-4" />}
          </IconButton>
          <IconButton onClick={() => setFrameIdx((i) => Math.max(0, i - 1))} label="previous frame">
            <SkipBack className="size-4" />
          </IconButton>
          <Slider
            className="mx-1 w-64"
            min={0}
            max={meta.n_frames - 1}
            value={[frameIdx]}
            onValueChange={(v) => setFrameIdx(Array.isArray(v) ? v[0] : v)}
          />
          <IconButton onClick={() => setFrameIdx((i) => Math.min(meta.n_frames - 1, i + 1))} label="next frame">
            <SkipForward className="size-4" />
          </IconButton>
          <span className="w-12 text-right font-mono text-xs text-white/50">{fm.t}/{meta.n_frames - 1}</span>
          <Select value={String(speed)} onValueChange={(v) => set("speed", Number(v))}>
            <SelectTrigger size="sm" className="border-white/10 bg-white/[0.04]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {[0.25, 0.5, 1, 2, 4].map((s) => (
                <SelectItem key={s} value={String(s)}>{s}x</SelectItem>
              ))}
            </SelectContent>
          </Select>
          <div className="mx-1 h-5 w-px bg-white/10" />
          <IconButton onClick={copyView} label="copy view to clipboard">
            {copied ? <Check className="size-4 text-white" /> : <Camera className="size-4" />}
          </IconButton>
          <IconButton onClick={downloadView} label="download PNG">
            <Download className="size-4" />
          </IconButton>
        </GlassPanel>
      )}

    </div>
  );
}
