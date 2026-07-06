"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslations } from "next-intl";
import type * as THREE from "three";
import Link from "next/link";
import { Camera, Check, ChevronLeft, Download, Pause, Play, RotateCcw, SkipBack, SkipForward, Sparkles } from "lucide-react";
import { CLASS_NAMES, SEMANTIC_COLORS, Scene3D, type Box, type LidarPoint, type Obstacle, type ReachableField } from "./scene3d";
import { FreeSpaceLayers } from "@/components/freespace/freespace-scene";
import { useFreeSpaceGeometry } from "@/components/freespace/use-freespace-geometry";
import { GeometryControls } from "@/components/freespace/geometry-controls";
import { ControlPanel } from "./controls";
import { useViewer, type RenderMode } from "./store";
import { GlassPanel } from "./glass";
import { LocaleToggle } from "@/components/locale-toggle";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { cn } from "@/lib/utils";
import { VerdictHero } from "./verdict-hero";
import { FailureRibbon } from "./failure-ribbon";
import { deltaSeries, deriveEvents } from "./events";
import { FINDINGS } from "@/lib/findings";

type Predicates = {
  ego_width: number;
  min_free_width: number | null;
  lateral_clearance: number | null;
  free_path_blocked: boolean;
  box_distance: number | null;
};
type FrameMeta = { t: number; speed: number; n_obstacles_band: number; n_obstacles_total: number; predicates: Predicates };
type SceneMeta = { scene: string; voxel_size: number; ego: { width: number; length: number; height: number }; n_frames: number; frames: FrameMeta[] };

// Scene data is hosted on Supabase Storage (public bucket `occquery-scenes` in the kencall project)
// so the deploy serves the full 3D scenes without committing ~285MB to git. The viewer already
// fetches per-frame on demand, so only viewed frames transfer. Override via NEXT_PUBLIC_OCCQUERY_BASE
// (e.g. to move to Cloudflare R2 later) — the public URL is not a secret.
const BASE =
  process.env.NEXT_PUBLIC_OCCQUERY_BASE ??
  "https://fppucwdfkkxyaqvgfixa.supabase.co/storage/v1/object/public/occquery-scenes";

const H1_FINDING = FINDINGS.find((f) => f.id === "h1");
const H3_FINDING = FINDINGS.find((f) => f.id === "h3");

function Row({ k, v, hot }: { k: string; v: string; hot?: boolean }) {
  return (
    <div className="flex justify-between gap-4">
      <span className="text-white/40">{k}</span>
      <span className={hot ? "text-red-400" : "text-white/80"}>{v}</span>
    </div>
  );
}

function Legend({ obstacles }: { obstacles: Obstacle[] }) {
  const t = useTranslations();
  const present = Array.from(new Set(obstacles.map((o) => o[3]))).sort((a, b) => a - b);
  if (!present.length) return null;
  return (
    <div className="mb-3 rounded-lg border border-white/10 bg-white/[0.03] p-2">
      <div className="mb-1 text-[10px] uppercase tracking-wide text-white/40">{t("occquery.legend")}</div>
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
  const t = useTranslations();
  const [scenes, setScenes] = useState<string[]>([]);
  const [scene, setScene] = useState("scene-0061");
  const [meta, setMeta] = useState<SceneMeta | null>(null);
  const [frameIdx, setFrameIdx] = useState(0);
  const [obstacles, setObstacles] = useState<Obstacle[]>([]);
  const [reachable, setReachable] = useState<ReachableField | null>(null);
  const [points, setPoints] = useState<LidarPoint[] | null>(null);
  const [boxes, setBoxes] = useState<Box[] | null>(null);
  const [copied, setCopied] = useState(false);
  const glRef = useRef<THREE.WebGLRenderer | null>(null);
  // free-space GEOMETRY layers (mesh / blocky / LiDAR recon / ground / texture), overlaid in this same
  // Canvas frame-swapped so it registers with the occquery voxels. One viewer, both representations.
  const [showGeometry, setShowGeometry] = useState(false);
  const geom = useFreeSpaceGeometry(scene);
  const cache = useRef<Map<string, { obstacles: Obstacle[]; boxes: Box[]; reachable: ReachableField | null }>>(new Map());
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
    const load = (tt: number): Promise<{ obstacles: Obstacle[]; boxes: Box[]; reachable: ReachableField | null }> => {
      const key = `${scene}/${tt}`;
      const hit = cache.current.get(key);
      if (hit) return Promise.resolve(hit);
      return fetch(`${BASE}/${scene}/f${tt}.json`).then((r) => r.json()).then((d) => {
        const frame = { obstacles: d.obstacles as Obstacle[], boxes: (d.boxes ?? []) as Box[], reachable: (d.reachable ?? null) as ReachableField | null };
        cache.current.set(key, frame);
        return frame;
      });
    };
    load(t).then((f) => { setObstacles(f.obstacles); setBoxes(f.boxes); setReachable(f.reachable); }).catch(() => { setObstacles([]); setBoxes(null); setReachable(null); });
    const nt = meta.frames[frameIdx + 1]?.t;
    if (nt !== undefined) load(nt).catch(() => {});
  }, [meta, frameIdx, scene]);

  // Background-prefetch ALL frames of the current scene into the cache so stepping/jumping is
  // instant (each frame is the only network hop; gzipped ~80KB). Throttled so it doesn't flood;
  // the load() above returns from cache once a frame is prefetched.
  useEffect(() => {
    if (!meta) return;
    let cancelled = false;
    const ts = meta.frames.map((f) => f.t).filter((t): t is number => t !== undefined);
    let next = 0;
    const fetchOne = async (tt: number) => {
      const key = `${scene}/${tt}`;
      if (cache.current.has(key)) return;
      try {
        const d = await fetch(`${BASE}/${scene}/f${tt}.json`).then((r) => r.json());
        cache.current.set(key, {
          obstacles: d.obstacles as Obstacle[],
          boxes: (d.boxes ?? []) as Box[],
          reachable: (d.reachable ?? null) as ReachableField | null,
        });
      } catch { /* ignore prefetch failures */ }
    };
    const worker = async () => { while (!cancelled && next < ts.length) await fetchOne(ts[next++]); };
    for (let i = 0; i < 6; i++) worker();
    return () => { cancelled = true; };
  }, [meta, scene]);

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
      else if (k === "b") st.toggle("showBoxes");
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
  const series = useMemo(() => (meta ? deltaSeries(meta.frames) : []), [meta]);
  const events = useMemo(() => (meta ? deriveEvents(meta.frames) : []), [meta]);

  return (
    <div className="relative h-screen w-full overflow-hidden bg-[#080808]">
      {meta && (
        <Scene3D obstacles={obstacles} ego={meta.ego} voxelSize={meta.voxel_size} reachable={reachable} points={points} boxes={boxes} onGl={(gl) => (glRef.current = gl)}>
          {showGeometry && <FreeSpaceLayers {...geom.layers} />}
        </Scene3D>
      )}

      {/* top-center — GEOMETRY overlay: the free-space mesh/blocky/LiDAR-recon/texture in the same scene */}
      <div className="absolute left-1/2 top-3 z-30 -translate-x-1/2">
        <GlassPanel className="flex items-center gap-2 px-2 py-1.5 text-white">
          <button
            onClick={() => setShowGeometry((v) => !v)}
            className={cn("rounded-full px-3 py-1 text-xs font-medium transition-colors", showGeometry ? "bg-white text-black" : "text-white/60 hover:text-white")}
          >
            Geometry
          </button>
          {showGeometry && <GeometryControls c={geom.controls} layers={geom.layers} splatMeta={geom.splatMeta} mesh={geom.mesh} />}
        </GlassPanel>
      </div>

      {/* top-left — context + view controls */}
      <GlassPanel className="absolute top-4 bottom-4 left-4 flex w-72 flex-col text-white">
        <div className="flex items-center justify-between px-3 pt-3 pb-2">
          <Link
            href="/overview"
            title="back to pipeline"
            className="flex items-center gap-1 text-sm font-medium tracking-tight text-white/90 transition-colors hover:text-white"
          >
            <ChevronLeft className="size-3.5 text-white/40" />
            {t("occquery.back")}
          </Link>
          <div className="flex items-center gap-2">
            <LocaleToggle />
            <button onClick={reset} className="flex items-center gap-1 text-[11px] text-white/40 transition-colors hover:text-white/70">
              <RotateCcw className="size-3" />
              {t("occquery.reset")}
            </button>
          </div>
        </div>

        {/* program-level verdict — stated ONCE, achromatic chrome, NOT per-frame (no oracle) */}
        <div className="px-3 pb-1 font-mono text-[10px] tracking-wide text-white/30">
          {t("occquery.programVerdict", { h1: H1_FINDING?.verdict ?? "HOLDS", h3: H3_FINDING?.verdict ?? "INCONCLUSIVE" })}
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
                {m === "voxel" ? t("occquery.renderMode.voxel") : m === "points" ? t("occquery.renderMode.lidar") : t("occquery.renderMode.both")}
              </button>
            ))}
          </div>
        </div>

        {meta && fm && p && (
          <div className="mx-3 mb-2 space-y-2">
            {/* VerdictHero — per-frame claim sentence, the reasoning HERO (measurement, not a verdict) */}
            <VerdictHero
              lateralClearance={p.lateral_clearance}
              boxDistance={p.box_distance ?? null}
              freePathBlocked={p.free_path_blocked}
            />
            <div className="space-y-1 rounded-lg bg-white/[0.04] p-2.5 font-mono text-xs">
              <Row k="speed" v={`${fm.speed} m/s`} />
              <Row k="obstacles" v={`${fm.n_obstacles_band}`} />
              <Row k="min_free_width" v={p.min_free_width === null ? t("occquery.rows.none") : `${p.min_free_width} m`} />
              <Row k="lateral_clearance" v={p.lateral_clearance === null ? t("occquery.rows.none") : `${p.lateral_clearance} m`} />
              <Row k="box_distance" v={p.box_distance == null ? t("occquery.rows.none") : `${p.box_distance} m`} />
              <Row k="free_path_blocked" v={p.free_path_blocked ? "TRUE" : "false"} hot={p.free_path_blocked} />
            </div>
            {/* not exported — never faked as a live number (see repo result-framing law) */}
            <div className="space-y-1 rounded-lg border border-white/[0.05] p-2.5 font-mono text-xs opacity-40">
              <div className="mb-0.5 text-[9px] uppercase tracking-wide text-white/20">{t("occquery.needsExport")}</div>
              <Row k="occlusion %" v="—" />
              <Row k="TTC" v="—" />
              <Row k="action-delta" v="—" />
            </div>
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
          <span className="text-sm font-medium">{t("occquery.ask.title")}</span>
          <span className="ml-auto rounded-full bg-white/10 px-2 py-0.5 text-[10px] tracking-wide text-white/50">{t("occquery.ask.soon")}</span>
        </div>
        <div className="flex flex-1 items-center justify-center px-6 text-center">
          <p className="text-xs leading-relaxed text-white/40">
            {t("occquery.ask.body")}
          </p>
        </div>
        <div className="border-t border-white/10 p-3">
          <div className="flex h-9 items-center rounded-xl bg-white/[0.04] px-3 text-xs text-white/30">{t("occquery.ask.placeholder")}</div>
        </div>
      </GlassPanel>

      {/* FailureRibbon — the time axis indexed BY the occ-vs-box measurement, not uniform time */}
      {meta && series.length > 0 && (
        <GlassPanel className="absolute bottom-20 left-1/2 -translate-x-1/2 px-3 py-2 text-white">
          <FailureRibbon series={series} events={events} frameIdx={frameIdx} onSeek={setFrameIdx} totalFrames={meta.n_frames} />
        </GlassPanel>
      )}

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
