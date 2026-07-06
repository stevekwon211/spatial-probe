"use client";
// Fetch the precomputed Occ3D free-space data: scene index (api route) + per-scene occupancy bytes
// and corridor JSON (static /public/occ).
import type { OccGrid } from "./mdc-client";

export interface Corridor {
  half_width: number; horizon: number;
  aggregated_clearance: number | null; confirmed_free_to: number | null;
  states: { x: number; state: "free" | "unknown" | "blocked"; free_frac: number }[];
}
export interface FreeSpace {
  scene: string; nx: number; ny: number; nz: number; voxel_size: number; origin: [number, number, number];
  unknown_frac_single_sweep: number;
  observed: [number, number, number][];
  corridor: Corridor;
}
export interface OccIndex {
  available: boolean; scenes: string[];
  nx: number; ny: number; nz: number; voxel_size: number; origin: [number, number, number];
}

/** One camera's intrinsics + ego->sensor rotation (Rt, row-major) + sensor origin (t), frame-0. */
export interface Cam {
  cam: string; img: string; w: number; h: number;
  fx: number; fy: number; cx: number; cy: number;
  Rt: number[]; t: number[];
}
export interface Cams { cameras: Cam[]; }

export async function fetchIndex(): Promise<OccIndex> {
  const r = await fetch("/api/occ");
  return r.json();
}

/** The frame-0 six cameras for render-time projective texturing, or null if not prepped. */
export async function fetchCams(scene: string): Promise<Cams | null> {
  const r = await fetch(`/occ/${scene}.cams.json`);
  if (!r.ok) return null;
  return r.json();
}

/** gsplat sidecar meta (rendered Gaussian count + sub-voxel ratio). Its presence is the feature
 * flag for the splat toggle — if the (gitignored/derived) asset isn't deployed, this 404s and the
 * splat stays disabled instead of the <Splat> loader crashing the page. */
export interface SplatMeta { count: number; voxelVerts: number; ratio: number; preCull: number; }
export async function fetchSplatMeta(scene: string): Promise<SplatMeta | null> {
  const r = await fetch(`/gsplat/${scene}/gsplat.meta.json`);
  if (!r.ok) return null;
  return r.json();
}

export async function fetchFreespace(scene: string): Promise<FreeSpace> {
  const r = await fetch(`/occ/${scene}.freespace.json`);
  return r.json();
}

export async function fetchOccGrid(scene: string, idx: OccIndex): Promise<OccGrid> {
  const r = await fetch(`/occ/${scene}.occ.bin`);
  const occ = new Uint8Array(await r.arrayBuffer());
  return { occ, nx: idx.nx, ny: idx.ny, nz: idx.nz, vs: idx.voxel_size, origin: idx.origin };
}

/** The honest finer reconstruction: 0.2m surface-hit occupancy from ALL 39 keyframe LiDAR sweeps
 * accumulated (ego-motion aligned) with moving objects removed (box tracks). 2x finer than Occ3D, every
 * cell a real return. Own dims sidecar (400x400x32). null if not prepped for this scene. */
export async function fetchLidarOcc(scene: string): Promise<OccGrid | null> {
  const [rb, rj] = await Promise.all([fetch(`/occ/${scene}.lidar02.occ.bin`), fetch(`/occ/${scene}.lidar02.json`)]);
  if (!rb.ok || !rj.ok) return null;
  const d = await rj.json();
  return { occ: new Uint8Array(await rb.arrayBuffer()), nx: d.nx, ny: d.ny, nz: d.nz, vs: d.voxel_size, origin: d.origin };
}

/** Per-voxel RGB (uint8, [x*ny*nz+y*nz+z]*3) — the camera-projective color. */
export async function fetchColor(scene: string): Promise<Uint8Array | null> {
  const r = await fetch(`/occ/${scene}.color.bin`);
  if (!r.ok) return null;
  return new Uint8Array(await r.arrayBuffer());
}

/** The FREE drivable-surface layer: a BEV top-ground heightfield ([x*ny+y], 255 = no ground label) +
 * per-cell subtype. Occ3D classes 11-14 (drivable/other-flat/sidewalk/terrain) are mapped to FREE, so
 * they are absent from the OCCUPIED mesh — this is the road, exported as its own honest layer. */
export interface Ground { hz: Uint8Array; sub: Uint8Array; nx: number; ny: number; }
export async function fetchGround(scene: string, idx: OccIndex): Promise<Ground | null> {
  const [rh, rs] = await Promise.all([fetch(`/occ/${scene}.ground.bin`), fetch(`/occ/${scene}.groundsub.bin`)]);
  if (!rh.ok) return null;
  const hz = new Uint8Array(await rh.arrayBuffer());
  const sub = rs.ok ? new Uint8Array(await rs.arrayBuffer()) : new Uint8Array(hz.length);
  return { hz, sub, nx: idx.nx, ny: idx.ny };
}

/** Triangulate the ground heightfield into one flat 0.4m tile per labeled cell (top face of the top
 * ground voxel). Independent tiles: same-height neighbors abut exactly, a height step reads as an
 * honest curb, and cells with NO ground label (255 — Occ3D leaves ground unlabeled under obstacles)
 * are simply absent, so the floor has genuine holes and obstacles stay honestly detached. */
export function buildGroundGeometry(g: Ground, idx: OccIndex): { pos: Float32Array; idx: Uint32Array; color: Float32Array } {
  const { nx, ny } = g;
  const { voxel_size: vs, origin } = idx;
  const half = vs / 2;
  const pos: number[] = [], ind: number[] = [], col: number[] = [];
  let v = 0;
  for (let x = 0; x < nx; x++) for (let y = 0; y < ny; y++) {
    const h = g.hz[x * ny + y];
    if (h === 255) continue;
    const cx = origin[0] + x * vs, cy = origin[1] + y * vs, z = origin[2] + (h + 0.5) * vs;
    pos.push(cx - half, cy - half, z, cx + half, cy - half, z, cx + half, cy + half, z, cx - half, cy + half, z);
    for (let k = 0; k < 4; k++) col.push(0.5, 0.5, 0.5); // neutral projective-fallback where a camera didn't see the road
    ind.push(v, v + 1, v + 2, v, v + 2, v + 3);
    v += 4;
  }
  return { pos: new Float32Array(pos), idx: new Uint32Array(ind), color: new Float32Array(col) };
}

/** Per-mesh-vertex "floating debris" flag (1 = belongs to a tiny isolated component). We run 6-conn
 * connected components over the occupied grid and flag a component as debris only when it is BOTH tiny
 * (<= threshVox voxels) AND short (z-extent <= threshZ voxels) — the size class that is dominated by
 * vegetation speckle / sensor noise while a real pole is thin-but-TALL and a vehicle is bigger. This is
 * for DISPLAY de-emphasis only (fade, never delete): a sparse GT shatters a real mid-range vehicle into
 * several <=3-voxel components, so deleting by size would erase real cars — measured, rejected. The
 * viewer keeps every voxel and lets a reviewer un-fade the flagged fragments. Returns a per-vertex mask
 * aligned to `pos` (nearest-occupied-voxel lookup, same 8-neighbour sample as vertexColors). */
export function debrisFlags(pos: Float32Array, g: OccGrid, threshVox = 3, threshZ = 3): Uint8Array {
  const { occ, nx, ny, nz, vs, origin } = g;
  const li = (x: number, y: number, z: number) => (x * ny + y) * nz + z;
  const comp = new Int32Array(nx * ny * nz).fill(-1);
  const size: number[] = [], zmin: number[] = [], zmax: number[] = [];
  const stack: number[] = [];
  let cid = 0;
  for (let x = 0; x < nx; x++) for (let y = 0; y < ny; y++) for (let z = 0; z < nz; z++) {
    const s = li(x, y, z);
    if (!occ[s] || comp[s] >= 0) continue;
    comp[s] = cid; stack.length = 0; stack.push(x, y, z);
    let sz = 0, zlo = z, zhi = z;
    while (stack.length) {
      const cz = stack.pop()!, cy = stack.pop()!, cx = stack.pop()!;
      sz++; if (cz < zlo) zlo = cz; if (cz > zhi) zhi = cz;
      const nb = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
      for (const [dx, dy, dz] of nb) {
        const ax = cx + dx, ay = cy + dy, az = cz + dz;
        if (ax < 0 || ax >= nx || ay < 0 || ay >= ny || az < 0 || az >= nz) continue;
        const ai = li(ax, ay, az);
        if (occ[ai] && comp[ai] < 0) { comp[ai] = cid; stack.push(ax, ay, az); }
      }
    }
    size.push(sz); zmin.push(zlo); zmax.push(zhi); cid++;
  }
  const isDebris = (c: number) => size[c] <= threshVox && (zmax[c] - zmin[c] + 1) <= threshZ;
  const n = pos.length / 3;
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const fx = Math.floor((pos[i * 3] - origin[0]) / vs);
    const fy = Math.floor((pos[i * 3 + 1] - origin[1]) / vs);
    const fz = Math.floor((pos[i * 3 + 2] - origin[2]) / vs);
    let c = -1;
    for (let dx = 0; dx <= 1 && c < 0; dx++) for (let dy = 0; dy <= 1 && c < 0; dy++) for (let dz = 0; dz <= 1 && c < 0; dz++) {
      const x = fx + dx, y = fy + dy, z = fz + dz;
      if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz) continue;
      const s = li(x, y, z); if (occ[s]) c = comp[s];
    }
    out[i] = c >= 0 && isDebris(c) ? 1 : 0;
  }
  return out;
}

/** Map each mesh vertex to the color of its nearest occupied voxel (the vertex sits on a voxel
 * boundary, so we sample the 8 surrounding voxels and take the first colored one). Returns rgb in
 * [0,1] per vertex, or null if no color grid. */
export function vertexColors(pos: Float32Array, color: Uint8Array, idx: OccIndex): Float32Array {
  const { nx, ny, nz, voxel_size: vs, origin } = idx;
  const n = pos.length / 3;
  const out = new Float32Array(n * 3);
  const sample = (x: number, y: number, z: number): number => {
    if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz) return -1;
    const li = (x * ny * nz + y * nz + z) * 3;
    return color[li] | color[li + 1] | color[li + 2] ? li : -1;
  };
  for (let i = 0; i < n; i++) {
    const gx = (pos[i * 3] - origin[0]) / vs;
    const gy = (pos[i * 3 + 1] - origin[1]) / vs;
    const gz = (pos[i * 3 + 2] - origin[2]) / vs;
    const fx = Math.floor(gx), fy = Math.floor(gy), fz = Math.floor(gz);
    let li = -1;
    for (let dx = 0; dx <= 1 && li < 0; dx++)
      for (let dy = 0; dy <= 1 && li < 0; dy++)
        for (let dz = 0; dz <= 1 && li < 0; dz++) li = sample(fx + dx, fy + dy, fz + dz);
    if (li >= 0) { out[i * 3] = color[li] / 255; out[i * 3 + 1] = color[li + 1] / 255; out[i * 3 + 2] = color[li + 2] / 255; }
    else { out[i * 3] = 0.42; out[i * 3 + 1] = 0.5; out[i * 3 + 2] = 0.82; } // uncolored -> the shaded blue
  }
  return out;
}
