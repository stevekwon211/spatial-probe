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

/** Per-voxel RGB (uint8, [x*ny*nz+y*nz+z]*3) — the camera-projective color. */
export async function fetchColor(scene: string): Promise<Uint8Array | null> {
  const r = await fetch(`/occ/${scene}.color.bin`);
  if (!r.ok) return null;
  return new Uint8Array(await r.arrayBuffer());
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
