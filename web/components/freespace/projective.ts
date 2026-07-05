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
  varying vec3 vWorld;
  varying vec3 vNrm;
  varying vec3 vVox;
  void main() {
    vec4 wp = modelMatrix * vec4(position, 1.0);
    vWorld = wp.xyz;
    vNrm = normalize(mat3(modelMatrix) * normal);
    vVox = color;
    gl_Position = projectionMatrix * viewMatrix * wp;
  }
`;

const FRAG = /* glsl */ `
  precision highp float;
  varying vec3 vWorld;
  varying vec3 vNrm;
  varying vec3 vVox;
  uniform sampler2D uTex0, uTex1, uTex2, uTex3, uTex4, uTex5;
  uniform mat3 uRt[6];   // ego -> sensor (R^T)
  uniform vec3 uT[6];    // sensor origin in ego
  uniform vec4 uK[6];    // fx, fy, cx, cy
  uniform vec2 uRes;     // image W, H (shared)

  vec4 samp(sampler2D tex, mat3 Rt, vec3 t, vec4 K, vec3 n) {
    vec3 pc = Rt * (vWorld - t);
    if (pc.z < 0.2) return vec4(0.0);                    // behind / too close to camera
    float u = K.x * pc.x / pc.z + K.z;
    float v = K.y * pc.y / pc.z + K.w;
    if (u < 0.0 || u >= uRes.x || v < 0.0 || v >= uRes.y) return vec4(0.0);
    float f = dot(n, normalize(t - vWorld));             // facing this camera?
    if (f <= 0.05) return vec4(0.0);                     // front-face gate (cheap occlusion proxy)
    vec3 c = texture2D(tex, vec2(u / uRes.x, v / uRes.y)).rgb;
    return vec4(c * f, f);
  }

  void main() {
    vec3 n = normalize(vNrm);
    vec4 a = vec4(0.0);
    a += samp(uTex0, uRt[0], uT[0], uK[0], n);
    a += samp(uTex1, uRt[1], uT[1], uK[1], n);
    a += samp(uTex2, uRt[2], uT[2], uK[2], n);
    a += samp(uTex3, uRt[3], uT[3], uK[3], n);
    a += samp(uTex4, uRt[4], uT[4], uK[4], n);
    a += samp(uTex5, uRt[5], uT[5], uK[5], n);
    // covered by >=1 camera -> crisp projected pixel; else the approximate per-voxel color (no holes)
    vec3 col = a.w > 1e-3 ? a.rgb / a.w : vVox;
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
