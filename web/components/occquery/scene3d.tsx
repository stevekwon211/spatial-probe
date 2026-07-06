"use client";

import { useEffect, useRef, type ReactNode } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { Grid, Line, OrbitControls, OrthographicCamera, PerspectiveCamera, Stats } from "@react-three/drei";
import * as THREE from "three";
import { CAMERA_PRESETS, useViewer, type ColorMode } from "./store";

export type Obstacle = [number, number, number, number]; // ego frame: forward, left, up, class
export type LidarPoint = [number, number, number, number]; // ego frame: forward, left, up, intensity
export type Box = [number, number, number, number, number, number, number, string]; // ego: forward, left, up, length, width, height, yaw, label

export type ReachableField = {
  forward_min: number;
  lateral_min: number;
  resolution: number;
  ego_cell: [number, number];
  shape: [number, number]; // [NF, NL]
  mask: number[]; // flattened (NF, NL) row-major; 1 = reachable free-space
};

type Ranges = { minU: number; maxU: number; maxF: number; maxL: number };

// Pre-allocated instance budget; mesh.count is set per frame so we never re-mount on frame change.
// Holds a full-height occupancy frame without truncation. The voxels come from export_web sorted by
// (i,j,k) voxel index, so a cap BELOW the frame count drops a spatial corner (high-index voxels near the
// ego vanish) -- the "voxels don't follow the ego" artifact. Full-height frames run to ~40k voxels; keep
// headroom. One instanced draw call, so the cost is the per-frame matrix loop, not the GPU.
const MAX_VOXELS = 48000;

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

// The reachable free-space the predicates measure, rendered as thin ground tiles.
// This is the VISUAL of H1 (rendered == measured field), not a measurement-accuracy claim.
function ReachableOverlay({ field }: { field: ReachableField }) {
  const ref = useRef<THREE.InstancedMesh>(null!);
  useEffect(() => {
    const mesh = ref.current;
    if (!mesh) return;
    const m = new THREE.Matrix4();
    const [nf, nl] = field.shape;
    const res = field.resolution;
    const tile = res * 0.92;
    let n = 0;
    for (let fi = 0; fi < nf; fi++) {
      for (let li = 0; li < nl; li++) {
        if (field.mask[fi * nl + li]) {
          const forward = field.forward_min + fi * res;
          const lateral = field.lateral_min + li * res;
          m.makeScale(tile, 0.02, tile);
          m.setPosition(forward, 0.02, lateral); // three (x,y,z) = (forward, up, left)
          mesh.setMatrixAt(n, m);
          n++;
        }
      }
    }
    mesh.count = n;
    mesh.instanceMatrix.needsUpdate = true;
  }, [field]);
  const max = field.shape[0] * field.shape[1];
  return (
    <instancedMesh ref={ref} args={[undefined, undefined, max]}>
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial color="#cbd5e1" transparent opacity={0.18} toneMapped={false} depthWrite={false} />
    </instancedMesh>
  );
}

// The RAW LiDAR scan (the measurement the voxels are discretized FROM), rendered as a point cloud.
// Toggling voxel <-> points shows what discretization keeps vs drops -- the state-vs-render theme.
function Points({ points }: { points: LidarPoint[] }) {
  const ref = useRef<THREE.Points>(null!);
  const pointSize = useViewer((s) => s.pointSize);
  const colorMode = useViewer((s) => s.colorMode);
  useEffect(() => {
    const geom = ref.current?.geometry;
    if (!geom) return;
    const n = points.length;
    const pos = new Float32Array(n * 3);
    const col = new Float32Array(n * 3);
    const c = new THREE.Color();
    let minU = Infinity, maxU = -Infinity;
    for (const [, , u] of points) { minU = Math.min(minU, u); maxU = Math.max(maxU, u); }
    for (let i = 0; i < n; i++) {
      const [f, l, u, inten] = points[i];
      pos[i * 3] = f; pos[i * 3 + 1] = u; pos[i * 3 + 2] = l; // ego (fwd,left,up) -> three (x,y,z)
      // color: height by default; intensity when colorMode=flat/semantic (LiDAR has no class)
      const t = colorMode === "flat" || colorMode === "semantic"
        ? Math.min(1, inten / 60)
        : (u - minU) / (maxU - minU + 1e-6);
      col.set(c.setHSL(0.62 - 0.5 * Math.max(0, Math.min(1, t)), 0.7, 0.55).toArray(), i * 3);
    }
    geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geom.setAttribute("color", new THREE.BufferAttribute(col, 3));
    geom.attributes.position.needsUpdate = true;
  }, [points, colorMode]);
  return (
    <points ref={ref}>
      <bufferGeometry />
      <pointsMaterial size={pointSize} vertexColors sizeAttenuation toneMapped={false} />
    </points>
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

// The tracked-box pipeline's view: each box a white wireframe cuboid. Overlaid on the colored
// occupancy voxels, the counterfactual is visible -- voxels with NO box around them are structure
// the box pipeline cannot express (the H1 thesis, made visual). Achromatic by the color law: boxes
// are geometry/chrome, not a measured value, so they stay white, never teal.
function Boxes({ boxes }: { boxes: Box[] }) {
  return (
    <group>
      {boxes.map((b, i) => (
        <mesh key={i} position={[b[0], b[2], b[1]]} rotation={[0, -b[6], 0]}>
          <boxGeometry args={[b[3], b[5], b[4]]} />
          <meshStandardMaterial color="#e5e7eb" transparent opacity={0.4} wireframe toneMapped={false} />
        </mesh>
      ))}
    </group>
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
  reachable,
  points,
  boxes,
  onGl,
  children,
}: {
  obstacles: Obstacle[];
  ego: { width: number; length: number; height: number };
  voxelSize: number;
  reachable?: ReachableField | null;
  points?: LidarPoint[] | null;
  boxes?: Box[] | null;
  onGl: (gl: THREE.WebGLRenderer) => void;
  children?: ReactNode; // extra layers (e.g. the free-space geometry, frame-swapped into this Canvas)
}) {
  const renderMode = useViewer((s) => s.renderMode);
  const showVoxels = useViewer((s) => s.showVoxels);
  const showBoxes = useViewer((s) => s.showBoxes);
  const showEgo = useViewer((s) => s.showEgo);
  const showForward = useViewer((s) => s.showForward);
  const showGrid = useViewer((s) => s.showGrid);
  const showReachable = useViewer((s) => s.showReachable);
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
      {showVoxels && renderMode !== "points" && <Voxels obstacles={obstacles} size={voxelSize} />}
      {renderMode !== "voxel" && points && points.length > 0 && <Points points={points} />}
      {showBoxes && boxes && boxes.length > 0 && <Boxes boxes={boxes} />}
      {showReachable && reachable && <ReachableOverlay field={reachable} />}
      {showForward && <Line points={[[0, 0.2, 0], [8, 0.2, 0]]} color="#ef4444" lineWidth={3} />}
      {showGrid && (
        <Grid args={[80, 80]} cellSize={2} sectionSize={10} infiniteGrid fadeDistance={60} cellColor="#222" sectionColor="#333" />
      )}
      {showStats && <Stats />}
      {children}
      <OrbitControls makeDefault enableDamping dampingFactor={0.1} />
      <CameraController />
    </Canvas>
  );
}
