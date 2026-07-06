"use client";

import { useEffect, useState } from "react";
import {
  fetchIndex, fetchFreespace, fetchOccGrid, fetchLidarOcc, fetchColor, fetchCams, fetchGround, fetchSplatMeta,
  vertexColors, debrisFlags, type FreeSpace, type OccIndex, type Cams, type Ground, type SplatMeta,
} from "./data";
import { meshOccupancy, type Algo, type Meshed } from "./mdc-client";
import type { LayersProps } from "./freespace-scene";

export type Source = "occ3d" | "lidar";

// The free-space geometry (mesh + texture + ground + reconstruction) state + loading, extracted so BOTH
// the standalone view and the occquery Explorer render the SAME layers off one hook. `scene` is
// controlled by the caller (occquery drives its own scene selection).
export function useFreeSpaceGeometry(scene: string) {
  const [idx, setIdx] = useState<OccIndex | null>(null);
  const [algo, setAlgo] = useState<Algo>("qef");
  const [source, setSource] = useState<Source>("occ3d");
  const [showMesh, setShowMesh] = useState(true);
  const [textured, setTextured] = useState(true);
  const [showGround, setShowGround] = useState(true);
  const [showDebris, setShowDebris] = useState(false);
  const [occlude, setOcclude] = useState(false);
  const [showSplat, setShowSplat] = useState(false);
  const [mesh, setMesh] = useState<Meshed | null>(null);
  const [colors, setColors] = useState<Float32Array | null>(null);
  const [debris, setDebris] = useState<Uint8Array | null>(null);
  const [cams, setCams] = useState<Cams | null>(null);
  const [ground, setGround] = useState<Ground | null>(null);
  const [splatMeta, setSplatMeta] = useState<SplatMeta | null>(null);
  const [fs, setFs] = useState<FreeSpace | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => { fetchIndex().then(setIdx).catch(() => {}); }, []);

  useEffect(() => {
    if (!scene || !idx) return;
    let cancelled = false;
    setBusy(true); setMesh(null); setFs(null); setColors(null); setDebris(null); setCams(null); setGround(null); setSplatMeta(null);
    (async () => {
      const [f, col, cm, gr, sm] = await Promise.all([
        fetchFreespace(scene), fetchColor(scene), fetchCams(scene), fetchGround(scene, idx), fetchSplatMeta(scene),
      ]);
      const g = source === "lidar" ? await fetchLidarOcc(scene) : await fetchOccGrid(scene, idx);
      if (cancelled) return;
      setFs(f); setCams(cm); setGround(gr); setSplatMeta(sm);
      if (!g) { setBusy(false); return; }
      const m = await meshOccupancy(g, algo);
      if (cancelled) return;
      setMesh(m);
      setColors(source === "lidar" ? null : (col ? vertexColors(m.pos, col, idx) : null));
      setDebris(debrisFlags(m.pos, g));
      setBusy(false);
    })().catch(() => { if (!cancelled) setBusy(false); });
    return () => { cancelled = true; };
  }, [scene, algo, idx, source]);

  const layers: LayersProps = {
    mesh, colors, debris, cams, ground, idx, fs,
    showMesh, textured, showGround, showDebris, occlude, showSplat: showSplat && !!splatMeta, scene,
  };
  const controls = {
    algo, setAlgo, source, setSource, showMesh, setShowMesh, textured, setTextured,
    showGround, setShowGround, showDebris, setShowDebris, occlude, setOcclude, showSplat, setShowSplat,
  };
  return { idx, busy, mesh, splatMeta, fs, cams, colors, ground, layers, controls };
}

export type GeometryControls = ReturnType<typeof useFreeSpaceGeometry>["controls"];
