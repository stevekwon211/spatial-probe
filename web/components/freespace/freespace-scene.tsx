"use client";

import { useEffect, useMemo, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { Grid, OrbitControls, PerspectiveCamera, Splat } from "@react-three/drei";
import * as THREE from "three";
import { toCreasedNormals } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import type { Meshed } from "./mdc-client";
import type { FreeSpace, Cams, Ground, OccIndex } from "./data";
import { buildGroundGeometry } from "./data";
import { buildProjectiveMaterial, loadCamTextures, cameraVisibility } from "./projective";

// nuScenes ego frame: x-forward, y-left, z-up. We keep those axes and set the camera up to +z, so
// the free-space plane reads as the ground (matches the occquery viewer's ego convention).
const CORRIDOR_COLOR = { free: 0x22c55e, unknown: 0xf59e0b, blocked: 0xef4444 } as const;

function MeshObject({ mesh, colors, debris, cams, showMesh, textured, showDebris, occlude }: {
  mesh: Meshed | null; colors: Float32Array | null; debris: Uint8Array | null; cams: Cams | null;
  showMesh: boolean; textured: boolean; showDebris: boolean; occlude: boolean;
}) {
  const useColor = textured && !!colors && colors.length === (mesh?.pos.length ?? -1);
  // Per-vertex, per-camera visibility for the honest occlusion test (CPU depth pre-pass). Computed once
  // per (mesh, cams); fed to the projective shader as aVisA/aVisB and gated by the occlude toggle.
  const vis = useMemo(() => (mesh && cams?.cameras?.length ? cameraVisibility(mesh.pos, cams.cameras) : null), [mesh, cams]);
  // Split the mesh into a SOLID part and a DEBRIS part (triangles whose 3 verts all belong to a tiny
  // isolated component — the floating-island snowstorm). Nothing is deleted; the debris part just
  // renders faded by default so a reviewer can still un-fade and inspect it. A triangle is debris only
  // when all 3 verts are flagged, so a real surface bordering noise stays solid.
  const geoms = useMemo(() => {
    if (!mesh) return null;
    let vcol = colors;
    if (!vcol || vcol.length !== mesh.pos.length) {
      vcol = new Float32Array(mesh.pos.length);
      for (let i = 0; i < vcol.length; i += 3) { vcol[i] = 0.42; vcol[i + 1] = 0.5; vcol[i + 2] = 0.82; }
    }
    const idx = mesh.idx;
    const solidIdx: number[] = [], debrisIdx: number[] = [];
    for (let t = 0; t < idx.length; t += 3) {
      const a = idx[t], b = idx[t + 1], c = idx[t + 2];
      (debris && debris[a] && debris[b] && debris[c] ? debrisIdx : solidIdx).push(a, b, c);
    }
    // Crease-aware normals (30deg): the mesher's smooth SDF-gradient normals round every corner and
    // shade flat walls like clay (the "blobby" look); creasing keeps walls flat + corners hard so the
    // 0.4m voxel shape reads crisply. De-indexes; copies the color attribute too.
    const make = (indices: number[]) => {
      if (!indices.length) return null;
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(mesh.pos, 3));
      g.setAttribute("color", new THREE.BufferAttribute(vcol!, 3));
      if (vis) { // occlusion pre-pass visibility (carried through the creased de-index)
        g.setAttribute("aVisA", new THREE.BufferAttribute(vis.visA, 3));
        g.setAttribute("aVisB", new THREE.BufferAttribute(vis.visB, 3));
      }
      g.setIndex(new THREE.BufferAttribute(new Uint32Array(indices), 1));
      return toCreasedNormals(g, THREE.MathUtils.degToRad(30));
    };
    return { solid: make(solidIdx), debris: make(debrisIdx) };
  }, [mesh, colors, debris, vis]);
  useEffect(() => () => { geoms?.solid?.dispose(); geoms?.debris?.dispose(); }, [geoms]);

  // Render-time projective texturing (full-res camera images) when textured + cams available.
  const [projMat, setProjMat] = useState<THREE.ShaderMaterial | null>(null);
  useEffect(() => {
    if (!textured || !cams?.cameras?.length) { setProjMat(null); return; }
    let alive = true; let texs: THREE.Texture[] = [];
    loadCamTextures(cams.cameras).then((t) => {
      if (!alive) { t.forEach((x) => x.dispose()); return; }
      texs = t;
      setProjMat(buildProjectiveMaterial(cams.cameras, t));
    });
    return () => { alive = false; texs.forEach((x) => x.dispose()); };
  }, [cams, textured]);
  useEffect(() => () => projMat?.dispose(), [projMat]);
  // honest occlusion test: on only when the toggle is set AND we have the pre-pass visibility.
  useEffect(() => { if (projMat) projMat.uniforms.uOcclude.value = occlude && vis ? 1 : 0; }, [projMat, occlude, vis]);

  // Imperative shaded material: declaring <meshStandardMaterial> with a dynamic vertexColors flag does
  // NOT reconcile in R3F (the compiled program is cached, so toggling projected->shaded kept rendering
  // the old vertexColors=true material = flat voxel color). Building it here, keyed on useColor, gives
  // a fresh correctly-compiled material. Opaque neutral clay so the light rig reads as solid 3D form.
  const shadedMat = useMemo(() => new THREE.MeshStandardMaterial({
    vertexColors: useColor,
    color: new THREE.Color(useColor ? "#ffffff" : "#6b7688"),
    side: THREE.DoubleSide, roughness: 0.8, metalness: 0,
  }), [useColor]);
  useEffect(() => () => shadedMat.dispose(), [shadedMat]);

  // Faded material for the debris part: translucent, no depth write, so the floating-island snowstorm
  // recedes to a low-confidence haze instead of reading as solid geometry (toggle showDebris to inspect).
  const fadedMat = useMemo(() => new THREE.MeshStandardMaterial({
    color: new THREE.Color("#8a94a6"), transparent: true, opacity: 0.12, depthWrite: false,
    side: THREE.DoubleSide, roughness: 1, metalness: 0,
  }), []);
  useEffect(() => () => fadedMat.dispose(), [fadedMat]);

  if (!geoms || !showMesh) return null;
  // gate on `textured` too: when toggling to shaded, projMat clears a frame late, so without this the
  // stale projective material would render in the shaded view.
  const solidMat = projMat && textured ? projMat : shadedMat;
  return (
    <>
      {geoms.solid && <mesh geometry={geoms.solid} material={solidMat} />}
      {geoms.debris && <mesh geometry={geoms.debris} material={showDebris ? solidMat : fadedMat} />}
    </>
  );
}

// The FREE drivable-surface floor (classes 11-14, mapped to FREE so absent from the OCCUPIED mesh).
// It anchors the scene so obstacles read as objects-on-a-street. Rendered with the SAME projective
// camera shader when textured (real road/lane pixels), else a dim recessed grey so a confidently-FREE
// surface reads visibly distinct from OCCUPIED obstacles. Where Occ3D has no ground label (under/around
// obstacles) the tile is absent, so the floor keeps honest holes — obstacles stay honestly detached.
function GroundMesh({ ground, idx, cams, textured }: {
  ground: Ground | null; idx: OccIndex | null; cams: Cams | null; textured: boolean;
}) {
  const geom = useMemo(() => {
    if (!ground || !idx) return null;
    const { pos, idx: ind, color } = buildGroundGeometry(ground, idx);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setIndex(new THREE.BufferAttribute(ind, 1));
    g.setAttribute("color", new THREE.BufferAttribute(color, 3)); // projective-shader fallback color
    g.computeVertexNormals(); // flat tiles -> all +z, so the cameras (above) face it
    return g;
  }, [ground, idx]);
  useEffect(() => () => geom?.dispose(), [geom]);

  const [projMat, setProjMat] = useState<THREE.ShaderMaterial | null>(null);
  useEffect(() => {
    if (!textured || !cams?.cameras?.length) { setProjMat(null); return; }
    let alive = true; let texs: THREE.Texture[] = [];
    loadCamTextures(cams.cameras).then((t) => {
      if (!alive) { t.forEach((x) => x.dispose()); return; }
      texs = t;
      setProjMat(buildProjectiveMaterial(cams.cameras, t));
    });
    return () => { alive = false; texs.forEach((x) => x.dispose()); };
  }, [cams, textured]);
  useEffect(() => () => projMat?.dispose(), [projMat]);

  const dimMat = useMemo(() => new THREE.MeshStandardMaterial({
    color: new THREE.Color("#3a4048"), roughness: 1, metalness: 0, side: THREE.DoubleSide,
  }), []);
  useEffect(() => () => dimMat.dispose(), [dimMat]);

  if (!geom) return null;
  return <mesh geometry={geom} material={projMat && textured ? projMat : dimMat} />;
}

function Observed({ fs }: { fs: FreeSpace | null }) {
  const geom = useMemo(() => {
    if (!fs?.observed?.length) return null;
    const p = new Float32Array(fs.observed.length * 3);
    fs.observed.forEach((o, i) => { p[i * 3] = o[0]; p[i * 3 + 1] = o[1]; p[i * 3 + 2] = o[2]; });
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(p, 3));
    return g;
  }, [fs]);
  useEffect(() => () => geom?.dispose(), [geom]);
  if (!geom) return null;
  return (
    <points geometry={geom}>
      <pointsMaterial color="#22d3ee" size={0.12} sizeAttenuation />
    </points>
  );
}

function Corridor({ fs }: { fs: FreeSpace | null }) {
  if (!fs) return null;
  const hw = fs.corridor.half_width;
  return (
    <>
      {fs.corridor.states.map((s, i) => (
        <mesh key={i} position={[s.x, 0, 0.06]}>
          <boxGeometry args={[0.36, hw * 2, 0.12]} />
          <meshBasicMaterial color={CORRIDOR_COLOR[s.state]} transparent
            opacity={s.state === "free" ? 0.5 : 0.75} />
        </mesh>
      ))}
    </>
  );
}

export function FreeSpaceScene({ mesh, colors, debris, cams, ground, idx, fs, showMesh, textured, showGround, showDebris, occlude, showSplat, scene }: {
  mesh: Meshed | null; colors: Float32Array | null; debris: Uint8Array | null; cams: Cams | null;
  ground: Ground | null; idx: OccIndex | null; fs: FreeSpace | null;
  showMesh: boolean; textured: boolean; showGround: boolean; showDebris: boolean; occlude: boolean; showSplat: boolean; scene: string;
}) {
  return (
    <Canvas dpr={[1, 2]} style={{ background: "#0b0d12" }}>
      <PerspectiveCamera makeDefault position={[-22, -20, 16]} up={[0, 0, 1]} fov={55} far={2000} />
      <OrbitControls enableDamping makeDefault target={[6, 0, 0]} />
      {/* z-up form rig (only affects the shaded meshStandardMaterial — projective/splat/overlays are
          unlit): low cool ambient + high key + cool low fill + back rim, so roofs read bright,
          near walls lit, far walls fall to shadow = solid 3D form instead of flat white. */}
      <ambientLight intensity={0.25} color="#8b97ad" />
      <directionalLight intensity={1.15} position={[9, 7, 14]} />
      <directionalLight intensity={0.45} position={[-11, -6, 3]} color="#7f8ba3" />
      <directionalLight intensity={0.4} position={[-4, 11, 7]} />
      <Grid args={[120, 120]} rotation={[Math.PI / 2, 0, 0]} cellColor="#161922" sectionColor="#222b3a"
        fadeDistance={90} infiniteGrid />
      {/* ego vehicle at the origin */}
      <mesh position={[0, 0, 0.8]}>
        <boxGeometry args={[4.5, 2, 1.6]} />
        <meshBasicMaterial color="#00e676" wireframe />
      </mesh>
      <axesHelper args={[3]} />
      {showGround && <GroundMesh ground={ground} idx={idx} cams={cams} textured={textured} />}
      <MeshObject mesh={mesh} colors={colors} debris={debris} cams={cams} showMesh={showMesh} textured={textured} showDebris={showDebris} occlude={occlude} />
      {/* image-based gsplat reconstruction; global->ego0 alignment baked into the .splat bytes so it
          renders at identity, sharing the makeDefault camera + OrbitControls. Availability (asset
          present) is gated by the caller via showSplat (meta-detected), so no 404 crash. */}
      {showSplat && <Splat src={`/gsplat/${scene}/gsplat.splat`} alphaTest={0.1} />}
      <Observed fs={fs} />
      <Corridor fs={fs} />
    </Canvas>
  );
}
