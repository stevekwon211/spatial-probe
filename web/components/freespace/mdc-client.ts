"use client";
// Client-side loader for the Rust/WASM MDC mesher (ported from mini-prism). The .wasm ships in
// /public/mdc and is fetched once; meshing then runs on the occupancy the Next api route serves.
import initWasm, { surface_nets, mdc_mesh, mdc_model_sdf } from "./mdc/mdc_wasm.js";

// The .wasm sits next to the generated glue, so webpack emits it as an asset and the glue's default
// `new URL('mdc_wasm_bg.wasm', import.meta.url)` resolves in dev + build (no next.config wasm flags).
let ready: Promise<unknown> | null = null;
const ensure = () => (ready ??= initWasm());

export type Algo = "qef" | "nets";

export interface Meshed {
  pos: Float32Array;
  idx: Uint32Array;
  normals?: Float32Array;
  tris: number;
  verts: number;
  fallback: number;
  defects: number;
  ms: number;
}

export interface OccGrid {
  occ: Uint8Array;
  nx: number; ny: number; nz: number; vs: number; origin: [number, number, number];
}

export async function meshOccupancy(g: OccGrid, algo: Algo): Promise<Meshed> {
  await ensure();
  const t0 = performance.now();
  if (algo === "qef") {
    const m = mdc_mesh(g.occ, g.nx, g.ny, g.nz, g.vs, g.origin[0], g.origin[1], g.origin[2]);
    return {
      pos: m.positions, idx: m.indices, normals: m.normals, tris: m.triangle_count,
      verts: m.vertex_count, fallback: m.fallback_count, defects: m.manifold_defects,
      ms: Math.round(performance.now() - t0),
    };
  }
  const m = surface_nets(g.occ, g.nx, g.ny, g.nz, g.vs, g.origin[0], g.origin[1], g.origin[2]);
  return {
    pos: m.positions, idx: m.indices, tris: m.triangle_count, verts: m.vertex_count,
    fallback: 0, defects: 0, ms: Math.round(performance.now() - t0),
  };
}

export async function modelSdf(g: OccGrid): Promise<Float32Array> {
  await ensure();
  return mdc_model_sdf(g.occ, g.nx, g.ny, g.nz).sdf;
}
