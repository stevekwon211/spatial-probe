"use client";

import { useEffect, useRef } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { Grid, Line, OrbitControls, OrthographicCamera, PerspectiveCamera, Stats } from "@react-three/drei";
import * as THREE from "three";
import { CAMERA_PRESETS, useViewer, type ColorMode } from "./store";

export type Obstacle = [number, number, number, number]; // ego frame: forward, left, up, class

type Ranges = { minU: number; maxU: number; maxF: number; maxL: number };

// Pre-allocated instance budget; mesh.count is set per frame so we never re-mount on frame change.
const MAX_VOXELS = 12000;

export const SEMANTIC_COLORS: Record<number, string> = {
  0: "#9ca3af", 1: "#f59e0b", 2: "#a855f7", 3: "#0ea5e9", 4: "#3b82f6",
  5: "#6366f1", 6: "#a855f7", 7: "#ef4444", 8: "#fbbf24", 9: "#0ea5e9",
  10: "#2563eb", 15: "#64748b", 16: "#22c55e",
};

export const CLASS_NAMES: Record<number, string> = {
  0: "other", 1: "barrier", 2: "bicycle", 3: "bus", 4: "car", 5: "constr veh",
  6: "motorcycle", 7: "pedestrian", 8: "cone", 9: "trailer", 10: "truck",
  15: "manmade", 16: "vegetation",
};

function colorFor(mode: ColorMode, f: number, l: number, u: number, cls: number, r: Ranges, c: THREE.Color) {
  if (mode === "semantic") return c.set(SEMANTIC_COLORS[cls] ?? "#9ca3af");
  if (mode === "flat" || mode === "state") return c.set("#3b82f6");
  let t = 0;
  if (mode === "height") t = (u - r.minU) / (r.maxU - r.minU + 1e-6);
  else if (mode === "forward") t = Math.min(f, r.maxF) / (r.maxF + 1e-6);
  else if (mode === "lateral") t = Math.abs(l) / (r.maxL + 1e-6);
  const clamped = Math.max(0, Math.min(1, t));
  return c.setHSL(0.66 - 0.62 * clamped, 0.75, 0.55);
}

function Voxels({ obstacles, size }: { obstacles: Obstacle[]; size: number }) {
  const ref = useRef<THREE.InstancedMesh>(null!);
  const colorMode = useViewer((s) => s.colorMode);
  const voxelScale = useViewer((s) => s.voxelScale);
  const voxelShape = useViewer((s) => s.voxelShape);
  const voxelOpacity = useViewer((s) => s.voxelOpacity);
  const wireframe = useViewer((s) => s.wireframe);

  // Update matrices/colors in place (no React re-render of the mesh, no re-mount on frame change).
  useEffect(() => {
    const mesh = ref.current;
    if (!mesh) return;
    const m = new THREE.Matrix4();
    const c = new THREE.Color();
    let minU = Infinity, maxU = -Infinity, maxF = 1, maxL = 1;
    for (const [f, l, u] of obstacles) {
      minU = Math.min(minU, u); maxU = Math.max(maxU, u);
      maxF = Math.max(maxF, f); maxL = Math.max(maxL, Math.abs(l));
    }
    const r: Ranges = { minU, maxU, maxF, maxL };
    const sc = size * voxelScale;
    const n = Math.min(obstacles.length, MAX_VOXELS);
    for (let i = 0; i < n; i++) {
      const [f, l, u, cls] = obstacles[i];
      m.makeScale(sc, sc, sc);
      m.setPosition(f, u, l);
      mesh.setMatrixAt(i, m);
      mesh.setColorAt(i, colorFor(colorMode, f, l, u, cls, r, c));
    }
    mesh.count = n;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [obstacles, colorMode, voxelScale, size]);

  // key only on shape, so swapping cube<->sphere re-mounts but frame stepping never does
  return (
    <instancedMesh key={voxelShape} ref={ref} args={[undefined, undefined, MAX_VOXELS]}>
      {voxelShape === "sphere" ? <sphereGeometry args={[0.5, 10, 10]} /> : <boxGeometry args={[1, 1, 1]} />}
      <meshStandardMaterial transparent opacity={voxelOpacity} wireframe={wireframe} toneMapped={false} />
    </instancedMesh>
  );
}

function Ego({ w, l, h }: { w: number; l: number; h: number }) {
  const egoOpacity = useViewer((s) => s.egoOpacity);
  const wireframe = useViewer((s) => s.wireframe);
  return (
    <mesh position={[0, h / 2, 0]}>
      <boxGeometry args={[l, h, w]} />
      <meshStandardMaterial color="#ef4444" transparent opacity={egoOpacity} wireframe={wireframe} />
    </mesh>
  );
}

function CameraController() {
  const { camera, controls } = useThree() as unknown as {
    camera: THREE.Camera;
    controls: { target: THREE.Vector3; update: () => void } | null;
  };
  const cam = useViewer((s) => s.cam);
  useEffect(() => {
    const p = CAMERA_PRESETS.find((x) => x.id === cam.preset);
    if (!p) return;
    camera.position.set(p.pos[0], p.pos[1], p.pos[2]);
    if (controls) {
      controls.target.set(0, 0, 0);
      controls.update();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cam.nonce]);
  return null;
}

export function Scene3D({
  obstacles,
  ego,
  voxelSize,
  onGl,
}: {
  obstacles: Obstacle[];
  ego: { width: number; length: number; height: number };
  voxelSize: number;
  onGl: (gl: THREE.WebGLRenderer) => void;
}) {
  const showVoxels = useViewer((s) => s.showVoxels);
  const showEgo = useViewer((s) => s.showEgo);
  const showForward = useViewer((s) => s.showForward);
  const showGrid = useViewer((s) => s.showGrid);
  const showStats = useViewer((s) => s.showStats);
  const projection = useViewer((s) => s.projection);

  return (
    <Canvas gl={{ preserveDrawingBuffer: true }} onCreated={({ gl }) => onGl(gl)}>
      {projection === "orthographic" ? (
        <OrthographicCamera makeDefault position={[-14, 11, 13]} zoom={18} near={0.1} far={400} />
      ) : (
        <PerspectiveCamera makeDefault position={[-14, 11, 13]} fov={50} near={0.1} far={400} />
      )}
      <color attach="background" args={["#0a0a0a"]} />
      <ambientLight intensity={0.6} />
      <directionalLight position={[10, 20, 10]} intensity={0.8} />
      {showEgo && <Ego w={ego.width} l={ego.length} h={ego.height} />}
      {showVoxels && <Voxels obstacles={obstacles} size={voxelSize} />}
      {showForward && <Line points={[[0, 0.2, 0], [8, 0.2, 0]]} color="#ef4444" lineWidth={3} />}
      {showGrid && (
        <Grid args={[80, 80]} cellSize={2} sectionSize={10} infiniteGrid fadeDistance={60} cellColor="#222" sectionColor="#333" />
      )}
      {showStats && <Stats />}
      <OrbitControls makeDefault enableDamping dampingFactor={0.1} />
      <CameraController />
    </Canvas>
  );
}
