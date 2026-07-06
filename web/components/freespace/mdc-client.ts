"use client";
// Client-side loader for the Rust/WASM MDC mesher (ported from mini-prism). The .wasm ships in
// /public/mdc and is fetched once; meshing then runs on the occupancy the Next api route serves.
import initWasm, { surface_nets, mdc_mesh, mdc_model_sdf } from "./mdc/mdc_wasm.js";

// The .wasm sits next to the generated glue, so webpack emits it as an asset and the glue's default
// `new URL('mdc_wasm_bg.wasm', import.meta.url)` resolves in dev + build (no next.config wasm flags).
let ready: Promise<unknown> | null = null;
const ensure = () => (ready ??= initWasm());

export type Algo = "qef" | "nets" | "blocky";

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
  if (algo === "blocky") return blockyMesh(g); // pure JS, no wasm — the honest cube render
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

// Blocky render: each occupied 0.4m voxel drawn as a cube, emitting only the faces on the occupied↔empty
// boundary (interior faces culled). No SDF, no smoothing — the surface IS the grid, so a lone voxel is a
// cube and a 1-voxel wall is one honest slab, never rounded into a blob. This is the field-standard way to
// show a discrete occupancy grid (OctoMap/SurroundOcc cube mode): for a QA viewer, truthful > smooth.
const FACES: { d: [number, number, number]; c: [number, number, number][] }[] = [
  { d: [1, 0, 0], c: [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]] },
  { d: [-1, 0, 0], c: [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]] },
  { d: [0, 1, 0], c: [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]] },
  { d: [0, -1, 0], c: [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]] },
  { d: [0, 0, 1], c: [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]] },
  { d: [0, 0, -1], c: [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]] },
];
export function blockyMesh(g: OccGrid): Meshed {
  const t0 = performance.now();
  const { occ, nx, ny, nz, vs, origin } = g;
  const at = (x: number, y: number, z: number) =>
    x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz ? 0 : occ[(x * ny + y) * nz + z];
  const pos: number[] = [], idx: number[] = [], nrm: number[] = [];
  let v = 0;
  for (let x = 0; x < nx; x++) for (let y = 0; y < ny; y++) for (let z = 0; z < nz; z++) {
    if (!occ[(x * ny + y) * nz + z]) continue;
    for (const f of FACES) {
      if (at(x + f.d[0], y + f.d[1], z + f.d[2])) continue; // interior face — cull
      for (const c of f.c) {
        pos.push(origin[0] + (x + c[0]) * vs, origin[1] + (y + c[1]) * vs, origin[2] + (z + c[2]) * vs);
        nrm.push(f.d[0], f.d[1], f.d[2]);
      }
      idx.push(v, v + 1, v + 2, v, v + 2, v + 3);
      v += 4;
    }
  }
  return {
    pos: new Float32Array(pos), idx: new Uint32Array(idx), normals: new Float32Array(nrm),
    tris: idx.length / 3, verts: v, fallback: 0, defects: 0, ms: Math.round(performance.now() - t0),
  };
}
