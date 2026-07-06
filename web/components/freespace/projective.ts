import * as THREE from "three";
import type { Cam } from "./data";

// Render-time PROJECTIVE texturing: each mesh fragment is projected back into the frame-0 cameras
// (p_sensor = Rt·(p_ego − t), then K·p/z → pixel) and samples the FULL-RES image, instead of the
// 0.4 m per-voxel color that smears. Front-face gate (normal·dirToCamera) is a cheap occlusion
// proxy — it stops back-face bleed but not a wall occluded by another wall both facing the camera
// (that needs per-camera depth pre-passes; a documented v2). Blend = facing-weighted average across
// the cameras that see a fragment; uncovered fragments fall back to the shaded blue (honest gap).

const VERT = /* glsl */ `
  attribute vec3 color;      // per-voxel camera color (approximate) — the uncovered fallback
  attribute vec3 aVisA;      // per-vertex visibility to cameras 0,1,2 (0/1) — occlusion pre-pass
  attribute vec3 aVisB;      // ... cameras 3,4,5
  varying vec3 vWorld;
  varying vec3 vNrm;
  varying vec3 vVox;
  varying vec3 vVisA;
  varying vec3 vVisB;
  void main() {
    vec4 wp = modelMatrix * vec4(position, 1.0);
    vWorld = wp.xyz;
    vNrm = normalize(mat3(modelMatrix) * normal);
    vVox = color;
    vVisA = aVisA;
    vVisB = aVisB;
    gl_Position = projectionMatrix * viewMatrix * wp;
  }
`;

const FRAG = /* glsl */ `
  precision highp float;
  varying vec3 vWorld;
  varying vec3 vNrm;
  varying vec3 vVox;
  varying vec3 vVisA;
  varying vec3 vVisB;
  uniform sampler2D uTex0, uTex1, uTex2, uTex3, uTex4, uTex5;
  uniform mat3 uRt[6];   // ego -> sensor (R^T)
  uniform vec3 uT[6];    // sensor origin in ego
  uniform vec4 uK[6];    // fx, fy, cx, cy
  uniform vec2 uRes;     // image W, H (shared)
  uniform float uOcclude; // 0 = normal-gate only; 1 = also require the pre-pass visibility bit

  vec4 samp(sampler2D tex, mat3 Rt, vec3 t, vec4 K, vec3 n, float vis) {
    if (uOcclude > 0.5 && vis < 0.5) return vec4(0.0);   // occluded: this camera does not SEE this surface
    vec3 pc = Rt * (vWorld - t);
    if (pc.z < 0.2) return vec4(0.0);                    // behind / too close to camera
    float u = K.x * pc.x / pc.z + K.z;
    float v = K.y * pc.y / pc.z + K.w;
    if (u < 0.0 || u >= uRes.x || v < 0.0 || v >= uRes.y) return vec4(0.0);
    float f = dot(n, normalize(t - vWorld));             // incidence: how head-on this camera sees the surface
    if (f <= 0.15) return vec4(0.0);                     // reject grazing views — a near-horizontal ray
                                                          // stretches one image column into a long smear
                                                          // (the "stretched road" streaks); fall back instead
    vec3 c = texture2D(tex, vec2(u / uRes.x, v / uRes.y)).rgb;
    float w = f * f;                                      // steeper weight: head-on views dominate the blend
    return vec4(c * w, w);
  }

  void main() {
    vec3 n = normalize(vNrm);
    vec4 a = vec4(0.0);
    a += samp(uTex0, uRt[0], uT[0], uK[0], n, vVisA.x);
    a += samp(uTex1, uRt[1], uT[1], uK[1], n, vVisA.y);
    a += samp(uTex2, uRt[2], uT[2], uK[2], n, vVisA.z);
    a += samp(uTex3, uRt[3], uT[3], uK[3], n, vVisB.x);
    a += samp(uTex4, uRt[4], uT[4], uK[4], n, vVisB.y);
    a += samp(uTex5, uRt[5], uT[5], uK[5], n, vVisB.z);
    // covered by >=1 camera -> crisp projected pixel; else the approximate per-voxel color. With the
    // occlusion test on, a fragment no camera SAW gets the approximation DIMMED, so "a camera actually
    // verified this pixel" reads bright/crisp and "we're guessing (unseen)" reads recessed — honest,
    // without punching a hard blue hole.
    vec3 col = a.w > 1e-3 ? a.rgb / a.w : (uOcclude > 0.5 ? vVox * 0.5 : vVox);
    gl_FragColor = vec4(col, 1.0);
  }
`;

export function buildProjectiveMaterial(
  cams: Cam[],
  textures: THREE.Texture[],
): THREE.ShaderMaterial {
  const c = cams.slice(0, 6);
  const Rt = c.map((k) => new THREE.Matrix3().set(...(k.Rt as [number, number, number, number, number, number, number, number, number])));
  const uniforms: Record<string, THREE.IUniform> = {
    uRt: { value: Rt },
    uT: { value: c.map((k) => new THREE.Vector3(k.t[0], k.t[1], k.t[2])) },
    uK: { value: c.map((k) => new THREE.Vector4(k.fx, k.fy, k.cx, k.cy)) },
    uRes: { value: new THREE.Vector2(c[0].w, c[0].h) },
    uOcclude: { value: 0 },
  };
  textures.forEach((tex, i) => { uniforms[`uTex${i}`] = { value: tex }; });
  // pad to 6 samplers so the shader always has uTex0..5 bound
  for (let i = textures.length; i < 6; i++) uniforms[`uTex${i}`] = { value: textures[0] ?? null };
  return new THREE.ShaderMaterial({
    vertexShader: VERT, fragmentShader: FRAG, uniforms, side: THREE.DoubleSide,
  });
}

/** Load the 6 camera JPEGs as textures ready for projection (no flipY, sRGB). */
export async function loadCamTextures(cams: Cam[]): Promise<THREE.Texture[]> {
  const loader = new THREE.TextureLoader();
  const texs = await Promise.all(cams.map((c) => loader.loadAsync(`/occ/${c.img}`)));
  for (const t of texs) { t.flipY = false; t.colorSpace = THREE.SRGBColorSpace; t.needsUpdate = true; }
  return texs;
}

/** Per-vertex, per-camera VISIBILITY for the render-time occlusion test (the honest fix for the
 * front-wall-pixels-leak-onto-back-walls bleed the normal gate can't catch). It is a CPU depth pre-pass
 * that mirrors the shader's exact projection (p_sensor = Rt·(p_ego − t), u=fx·x/z+cx): project every
 * mesh vertex into each camera, z-buffer them into 4px pixel buckets, and mark a vertex visible to a
 * camera only if it is within TOL of the nearest surface in its bucket (so a whole front wall stays
 * visible but an occluded back wall is rejected). Returns two vec3 attributes (cams 0-2 / 3-5), each
 * 0/1, fed to the shader as aVisA/aVisB and gated by uOcclude. Same algorithm as prep's voxel z-buffer,
 * so no projection-matrix derivation and no sign-convention risk. */
export function cameraVisibility(pos: Float32Array, cams: Cam[]): { visA: Float32Array; visB: Float32Array } {
  const n = pos.length / 3;
  const nc = Math.min(cams.length, 6);
  const TOL = 0.5; // m — same-front-surface depth spread within a bucket
  const visA = new Float32Array(n * 3), visB = new Float32Array(n * 3);
  const uu = new Float32Array(n), vv = new Float32Array(n), zz = new Float32Array(n);
  for (let ci = 0; ci < nc; ci++) {
    const cam = cams[ci], R = cam.Rt, t = cam.t;
    const bw = (cam.w >> 2) + 1;
    const bucket = new Map<number, number>();
    for (let i = 0; i < n; i++) {
      const dx = pos[i * 3] - t[0], dy = pos[i * 3 + 1] - t[1], dz = pos[i * 3 + 2] - t[2];
      const px = R[0] * dx + R[1] * dy + R[2] * dz;
      const py = R[3] * dx + R[4] * dy + R[5] * dz;
      const pz = R[6] * dx + R[7] * dy + R[8] * dz;
      if (pz <= 0.2) { zz[i] = -1; continue; }
      const u = cam.fx * px / pz + cam.cx, v = cam.fy * py / pz + cam.cy;
      if (u < 0 || u >= cam.w || v < 0 || v >= cam.h) { zz[i] = -1; continue; }
      uu[i] = u; vv[i] = v; zz[i] = pz;
      const key = (Math.floor(v) >> 2) * bw + (Math.floor(u) >> 2);
      const cur = bucket.get(key);
      if (cur === undefined || pz < cur) bucket.set(key, pz);
    }
    const dst = ci < 3 ? visA : visB, comp = ci % 3;
    for (let i = 0; i < n; i++) {
      if (zz[i] <= 0) continue;
      const key = (Math.floor(vv[i]) >> 2) * bw + (Math.floor(uu[i]) >> 2);
      if (zz[i] <= (bucket.get(key) as number) + TOL) dst[i * 3 + comp] = 1;
    }
  }
  return { visA, visB };
}
