"use client";

import { useEffect, useMemo, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { Grid, OrbitControls, PerspectiveCamera, Splat } from "@react-three/drei";
import * as THREE from "three";
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
    if (mesh.normals && mesh.normals.length === mesh.pos.length) {
      g.setAttribute("normal", new THREE.BufferAttribute(mesh.normals, 3)); // analytic SDF-gradient normals
    } else {
      g.computeVertexNormals();
    }
    // always carry a per-vertex color: the voxel color if we have it, else the shaded blue. The
    // standard material only reads it when useColor; the projective shader uses it as the
    // uncovered-fragment fallback (so no flat-blue holes where the cameras didn't see).
    let vcol = colors;
    if (!vcol || vcol.length !== mesh.pos.length) {
      vcol = new Float32Array(mesh.pos.length);
      for (let i = 0; i < vcol.length; i += 3) { vcol[i] = 0.42; vcol[i + 1] = 0.5; vcol[i + 2] = 0.82; }
    }
    g.setAttribute("color", new THREE.BufferAttribute(vcol, 3));
    return g;
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

  if (!geom || !showMesh) return null;
  if (projMat) return <mesh geometry={geom} material={projMat} />; // projected camera color, full-res
  return (
    <mesh geometry={geom}>
      <meshStandardMaterial vertexColors={useColor} color={useColor ? "#ffffff" : "#6b7fd0"}
        transparent opacity={textured ? 0.95 : 0.55} side={THREE.DoubleSide}
        flatShading={!mesh?.normals} roughness={0.9} metalness={0} />
    </mesh>
  );
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
      <ambientLight intensity={textured ? 1.6 : 0.7} color={textured ? "#ffffff" : "#8899bb"} />
      <directionalLight intensity={textured ? 0.9 : 1.2} position={[0.5, -0.7, 1]} />
      <Grid args={[120, 120]} rotation={[Math.PI / 2, 0, 0]} cellColor="#161922" sectionColor="#222b3a"
        fadeDistance={90} infiniteGrid />
      {/* ego vehicle at the origin */}
      <mesh position={[0, 0, 0.8]}>
        <boxGeometry args={[4.5, 2, 1.6]} />
        <meshBasicMaterial color="#00e676" wireframe />
      </mesh>
      <axesHelper args={[3]} />
      <MeshObject mesh={mesh} colors={colors} cams={cams} showMesh={showMesh} textured={textured} />
      {/* image-based gsplat reconstruction (644k Gaussians); global->ego0 alignment baked into the
          .splat bytes so it renders at identity, sharing the makeDefault camera + OrbitControls */}
      {showSplat && scene === "scene-0061" && <Splat src={`/gsplat/${scene}/gsplat.splat`} alphaTest={0.1} />}
      <Observed fs={fs} />
      <Corridor fs={fs} />
    </Canvas>
  );
}
