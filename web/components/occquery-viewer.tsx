"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { Grid, Line, OrbitControls } from "@react-three/drei";
import * as THREE from "three";

type Predicates = {
  ego_width: number;
  min_free_width: number | null;
  lateral_clearance: number | null;
  free_path_blocked: boolean;
};
type FrameMeta = {
  t: number;
  speed: number;
  n_obstacles_band: number;
  n_obstacles_total: number;
  predicates: Predicates;
};
type SceneMeta = {
  scene: string;
  voxel_size: number;
  ground_height: number;
  ego: { width: number; length: number; height: number };
  n_frames: number;
  frames: FrameMeta[];
};

const BASE = "/data/occquery";

// Ego frame: x = forward, y = left, z = up. Three.js is y-up, so we map
// (forward, left, up) -> three (x=forward, y=up, z=left).
function Voxels({ obstacles, size }: { obstacles: number[][]; size: number }) {
  const ref = useRef<THREE.InstancedMesh>(null!);
  useEffect(() => {
    if (!ref.current) return;
    const m = new THREE.Matrix4();
    for (let i = 0; i < obstacles.length; i++) {
      const [fwd, left, up] = obstacles[i];
      m.setPosition(fwd, up, left);
      ref.current.setMatrixAt(i, m);
    }
    ref.current.instanceMatrix.needsUpdate = true;
  }, [obstacles]);
  return (
    <instancedMesh
      key={obstacles.length}
      ref={ref}
      args={[undefined, undefined, Math.max(obstacles.length, 1)]}
    >
      <boxGeometry args={[size * 0.85, size * 0.85, size * 0.85]} />
      <meshStandardMaterial color="#3b82f6" />
    </instancedMesh>
  );
}

function Ego({ w, l, h }: { w: number; l: number; h: number }) {
  return (
    <mesh position={[0, h / 2, 0]}>
      <boxGeometry args={[l, h, w]} />
      <meshStandardMaterial color="#ef4444" transparent opacity={0.55} />
    </mesh>
  );
}

function Row({ k, v, hot }: { k: string; v: string; hot?: boolean }) {
  return (
    <div className="flex justify-between gap-4">
      <span className="text-muted-foreground">{k}</span>
      <span className={hot ? "text-red-400" : ""}>{v}</span>
    </div>
  );
}

export function OccqueryViewer() {
  const [scenes, setScenes] = useState<string[]>([]);
  const [scene, setScene] = useState("scene-0061");
  const [meta, setMeta] = useState<SceneMeta | null>(null);
  const [frameIdx, setFrameIdx] = useState(0);
  const [obstacles, setObstacles] = useState<number[][]>([]);
  const [copied, setCopied] = useState(false);
  const glRef = useRef<THREE.WebGLRenderer | null>(null);

  useEffect(() => {
    fetch(`${BASE}/index.json`)
      .then((r) => r.json())
      .then((d) => setScenes(d.scenes.map((s: { scene: string }) => s.scene)))
      .catch(() => setScenes([]));
  }, []);

  useEffect(() => {
    setMeta(null);
    setFrameIdx(0);
    fetch(`${BASE}/${scene}.json`)
      .then((r) => r.json())
      .then(setMeta)
      .catch(() => setMeta(null));
  }, [scene]);

  useEffect(() => {
    if (!meta) return;
    const t = meta.frames[frameIdx]?.t;
    if (t === undefined) return;
    fetch(`${BASE}/${scene}/f${t}.json`)
      .then((r) => r.json())
      .then((d) => setObstacles(d.obstacles))
      .catch(() => setObstacles([]));
  }, [meta, frameIdx, scene]);

  const capture = useCallback(() => {
    const gl = glRef.current;
    if (!gl) return;
    gl.domElement.toBlob(async (blob) => {
      if (!blob) return;
      try {
        await navigator.clipboard.write([
          new ClipboardItem({ "image/png": blob }),
        ]);
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      } catch (e) {
        console.error("clipboard write failed", e);
      }
    }, "image/png");
  }, []);

  const fm = meta?.frames[frameIdx];
  const p = fm?.predicates;

  return (
    <div className="flex h-[calc(100vh-3.5rem)] w-full">
      <div className="relative flex-1">
        <Canvas
          camera={{ position: [-14, 11, 13], fov: 50 }}
          gl={{ preserveDrawingBuffer: true }}
          onCreated={({ gl }) => {
            glRef.current = gl;
          }}
        >
          <color attach="background" args={["#0a0a0a"]} />
          <ambientLight intensity={0.6} />
          <directionalLight position={[10, 20, 10]} intensity={0.8} />
          {meta && (
            <Ego w={meta.ego.width} l={meta.ego.length} h={meta.ego.height} />
          )}
          {meta && <Voxels obstacles={obstacles} size={meta.voxel_size} />}
          <Line
            points={[
              [0, 0.2, 0],
              [8, 0.2, 0],
            ]}
            color="#ef4444"
            lineWidth={3}
          />
          <Grid
            args={[80, 80]}
            cellSize={2}
            sectionSize={10}
            infiniteGrid
            fadeDistance={60}
            cellColor="#222"
            sectionColor="#333"
          />
          <OrbitControls makeDefault />
        </Canvas>
        <button
          onClick={capture}
          className="absolute right-4 top-4 rounded-md bg-white/10 px-3 py-2 text-sm text-white backdrop-blur transition hover:bg-white/20"
        >
          {copied ? "✓ copied" : "📷 copy view"}
        </button>
        <div className="absolute bottom-4 left-4 text-xs text-white/50">
          red box = ego &middot; red line = forward &middot; blue = obstacle voxels (ego height band)
        </div>
      </div>

      <aside className="w-80 shrink-0 overflow-y-auto border-l bg-background p-4 text-sm">
        <h2 className="mb-3 text-base font-semibold">occquery 3D</h2>

        <label className="mb-1 block text-xs text-muted-foreground">scene</label>
        <select
          value={scene}
          onChange={(e) => setScene(e.target.value)}
          className="mb-4 w-full rounded border bg-transparent px-2 py-1"
        >
          {scenes.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>

        {meta && fm && p ? (
          <>
            <label className="mb-1 block text-xs text-muted-foreground">
              frame {fm.t} / {meta.n_frames - 1}
            </label>
            <input
              type="range"
              min={0}
              max={meta.n_frames - 1}
              value={frameIdx}
              onChange={(e) => setFrameIdx(Number(e.target.value))}
              className="mb-4 w-full"
            />

            <div className="space-y-1 rounded-md border p-3 font-mono text-xs">
              <Row k="voxel_size" v={`${meta.voxel_size} m`} />
              <Row k="ego_speed" v={`${fm.speed} m/s`} />
              <Row k="obstacles (band)" v={`${fm.n_obstacles_band}`} />
              <Row k="obstacles (total)" v={`${fm.n_obstacles_total}`} />
              <hr className="my-2 border-border" />
              <Row k="ego_width" v={`${p.ego_width} m`} />
              <Row
                k="min_free_width"
                v={p.min_free_width === null ? "none" : `${p.min_free_width} m`}
              />
              <Row
                k="lateral_clearance"
                v={
                  p.lateral_clearance === null
                    ? "none"
                    : `${p.lateral_clearance} m`
                }
              />
              <Row
                k="free_path_blocked"
                v={p.free_path_blocked ? "TRUE" : "false"}
                hot={p.free_path_blocked}
              />
            </div>
            <p className="mt-3 text-[11px] leading-snug text-muted-foreground">
              corridor MATCH 판정은 보류 중 (positive set + 독립 oracle 필요).
              min_free_width는 측정값일 뿐 판정이 아니다.
            </p>
          </>
        ) : (
          <p className="text-xs text-muted-foreground">loading…</p>
        )}
      </aside>
    </div>
  );
}
