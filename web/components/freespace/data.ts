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

export async function fetchIndex(): Promise<OccIndex> {
  const r = await fetch("/api/occ");
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
