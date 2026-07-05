"use client";

import { useEffect, useMemo, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { Grid, OrbitControls, PerspectiveCamera } from "@react-three/drei";
import * as THREE from "three";
import type { Meshed } from "./mdc-client";
import type { FreeSpace } from "./data";

// nuScenes ego frame: x-forward, y-left, z-up. We keep those axes and set the camera up to +z, so
// the free-space plane reads as the ground (matches the occquery viewer's ego convention).
const CORRIDOR_COLOR = { free: 0x22c55e, unknown: 0xf59e0b, blocked: 0xef4444 } as const;

function MeshObject({ mesh, showMesh }: { mesh: Meshed | null; showMesh: boolean }) {
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
    return g;
  }, [mesh]);
  useEffect(() => () => geom?.dispose(), [geom]);
  if (!geom || !showMesh) return null;
  return (
    <mesh geometry={geom}>
      <meshStandardMaterial color="#6b7fd0" transparent opacity={0.55} side={THREE.DoubleSide}
        flatShading={!mesh?.normals} />
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

export function FreeSpaceScene({ mesh, fs, showMesh }: { mesh: Meshed | null; fs: FreeSpace | null; showMesh: boolean }) {
  return (
    <Canvas dpr={[1, 2]} style={{ background: "#0b0d12" }}>
      <PerspectiveCamera makeDefault position={[-22, -20, 16]} up={[0, 0, 1]} fov={55} far={2000} />
      <OrbitControls enableDamping makeDefault target={[6, 0, 0]} />
      <ambientLight intensity={0.7} color="#8899bb" />
      <directionalLight intensity={1.2} position={[0.5, -0.7, 1]} />
      <Grid args={[120, 120]} rotation={[Math.PI / 2, 0, 0]} cellColor="#161922" sectionColor="#222b3a"
        fadeDistance={90} infiniteGrid />
      {/* ego vehicle at the origin */}
      <mesh position={[0, 0, 0.8]}>
        <boxGeometry args={[4.5, 2, 1.6]} />
        <meshBasicMaterial color="#00e676" wireframe />
      </mesh>
      <axesHelper args={[3]} />
      <MeshObject mesh={mesh} showMesh={showMesh} />
      <Observed fs={fs} />
      <Corridor fs={fs} />
    </Canvas>
  );
}
