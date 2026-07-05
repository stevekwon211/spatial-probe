/* tslint:disable */
/* eslint-disable */

export class MdcMesh {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Cells where the QEF was rank-deficient and fell back to the mass point (a QA signal — the
     * "no unique corner here" count, the geometry analog of the corridor's fog).
     */
    readonly fallback_count: number;
    readonly indices: Uint32Array;
    /**
     * Edges not shared by exactly 2 triangles (holes / non-manifold). 0 = watertight.
     */
    readonly manifold_defects: number;
    readonly normals: Float32Array;
    readonly positions: Float32Array;
    readonly triangle_count: number;
    readonly vertex_count: number;
    readonly vertex_unknown: Float32Array;
}

export class Mesh {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    readonly indices: Uint32Array;
    readonly positions: Float32Array;
    readonly triangle_count: number;
    readonly vertex_count: number;
}

/**
 * Model-friendly output: the signed distance field (f32, row-major, in voxel units) — a regular,
 * differentiable tensor a perception/physics model consumes, NOT a rendered image. The `unknown`
 * mask is a SEPARATE channel (fetch it alongside), never fused into the SDF (design §4).
 */
export class ModelSdf {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    readonly sdf: Float32Array;
}

/**
 * QEF Manifold-Dual-Contouring mesh of `occ` (the upgrade over `surface_nets`): EDT->SDF->Hermite
 * ->QEF vertex placement, so sharp corners are recovered instead of rounded. Returns positions +
 * indices + analytic normals + a per-vertex `unknown` channel (kept separate from geometry).
 */
export function mdc_mesh(occ: Uint8Array, nx: number, ny: number, nz: number, vs: number, ox: number, oy: number, oz: number): MdcMesh;

/**
 * Compute the SDF grid for the model audience (the same field the mesher uses internally).
 */
export function mdc_model_sdf(occ: Uint8Array, nx: number, ny: number, nz: number): ModelSdf;

/**
 * Surface-nets mesh of `occ`. `vs` = voxel size (m); `(ox,oy,oz)` = world coord of voxel (0,0,0).
 * Normals are left to the caller (three.js `computeVertexNormals()`) to keep this lean + correct.
 */
export function surface_nets(occ: Uint8Array, nx: number, ny: number, nz: number, vs: number, ox: number, oy: number, oz: number): Mesh;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_mdcmesh_free: (a: number, b: number) => void;
    readonly mdcmesh_fallback_count: (a: number) => number;
    readonly mdcmesh_indices: (a: number) => [number, number];
    readonly mdcmesh_manifold_defects: (a: number) => number;
    readonly mdcmesh_normals: (a: number) => [number, number];
    readonly mdcmesh_positions: (a: number) => [number, number];
    readonly mdcmesh_triangle_count: (a: number) => number;
    readonly mdcmesh_vertex_count: (a: number) => number;
    readonly mdcmesh_vertex_unknown: (a: number) => [number, number];
    readonly __wbg_mesh_free: (a: number, b: number) => void;
    readonly __wbg_modelsdf_free: (a: number, b: number) => void;
    readonly mdc_mesh: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => number;
    readonly mdc_model_sdf: (a: number, b: number, c: number, d: number, e: number) => number;
    readonly mesh_indices: (a: number) => [number, number];
    readonly mesh_positions: (a: number) => [number, number];
    readonly mesh_triangle_count: (a: number) => number;
    readonly mesh_vertex_count: (a: number) => number;
    readonly modelsdf_sdf: (a: number) => [number, number];
    readonly surface_nets: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
