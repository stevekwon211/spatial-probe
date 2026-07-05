"use client";

import { useEffect, useMemo, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { Grid, OrbitControls, PerspectiveCamera, Splat } from "@react-three/drei";
import * as THREE from "three";
import { toCreasedNormals } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import type { Meshed } from "./mdc-client";
import type { FreeSpace, Cams } from "./data";
import { buildProjectiveMaterial, loadCamTextures } from "./projective";

// nuScenes ego frame: x-forward, y-left, z-up. We keep those axes and set the camera up to +z, so
// the free-space plane reads as the ground (matches the occquery viewer's ego convention).
const CORRIDOR_COLOR = { free: 0x22c55e, unknown: 0xf59e0b, blocked: 0xef4444 } as const;

function MeshObject({ mesh, colors, cams, showMesh, textured }: {
  mesh: Meshed | null; colors: Float32Array | null; cams: Cams | null; showMesh: boolean; textured: boolean;
}) {
  const useColor = textured && !!colors && colors.length === (mesh?.pos.length ?? -1);
  const geom = useMemo(() => {
    if (!mesh) return null;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(mesh.pos, 3));
    g.setIndex(new THREE.BufferAttribute(mesh.idx, 1));
    // always carry a per-vertex color: the voxel color if we have it, else the shaded grey. The
    // standard material only reads it when useColor; the projective shader uses it as the
    // uncovered-fragment fallback (so no flat holes where the cameras didn't see).
    let vcol = colors;
    if (!vcol || vcol.length !== mesh.pos.length) {
      vcol = new Float32Array(mesh.pos.length);
      for (let i = 0; i < vcol.length; i += 3) { vcol[i] = 0.42; vcol[i + 1] = 0.5; vcol[i + 2] = 0.82; }
    }
    g.setAttribute("color", new THREE.BufferAttribute(vcol, 3));
    // Crease-aware normals: the mesher's analytic SDF-gradient normals are a smooth low-res field
    // that rounds every corner and shades flat walls like clay (the "blobby" look). Recomputing
    // normals with a 30deg crease keeps walls flat + makes building corners hard, so the 0.4m voxel
    // shape reads crisply — no mesher/data change. (De-indexes; copies the color attribute too.)
    return toCreasedNormals(g, THREE.MathUtils.degToRad(30));
  }, [mesh, colors]);
  useEffect(() => () => geom?.dispose(), [geom]);

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

  if (!geom || !showMesh) return null;
  // gate on `textured` too: when toggling to shaded, projMat clears a frame late, so without this the
  // stale projective material would render in the shaded view.
  return <mesh geometry={geom} material={projMat && textured ? projMat : shadedMat} />;
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

export function FreeSpaceScene({ mesh, colors, cams, fs, showMesh, textured, showSplat, scene }: {
  mesh: Meshed | null; colors: Float32Array | null; cams: Cams | null; fs: FreeSpace | null;
  showMesh: boolean; textured: boolean; showSplat: boolean; scene: string;
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
      <MeshObject mesh={mesh} colors={colors} cams={cams} showMesh={showMesh} textured={textured} />
      {/* image-based gsplat reconstruction; global->ego0 alignment baked into the .splat bytes so it
          renders at identity, sharing the makeDefault camera + OrbitControls. Availability (asset
          present) is gated by the caller via showSplat (meta-detected), so no 404 crash. */}
      {showSplat && <Splat src={`/gsplat/${scene}/gsplat.splat`} alphaTest={0.1} />}
      <Observed fs={fs} />
      <Corridor fs={fs} />
    </Canvas>
  );
}
