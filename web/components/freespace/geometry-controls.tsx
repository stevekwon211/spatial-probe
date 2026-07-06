"use client";

import { cn } from "@/lib/utils";
import type { GeometryControls as Ctrls } from "./use-freespace-geometry";
import type { Algo, Meshed } from "./mdc-client";
import type { LayersProps } from "./freespace-scene";
import type { SplatMeta } from "./data";

// The free-space GEOMETRY controls, styled to match occquery's left-panel Section/ToggleRow/Seg so it
// reads as one integrated sidebar, not a bolted-on strip. Achromatic by the repo design law.
function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <div className="px-1 text-[10px] font-medium uppercase tracking-wider text-white/40">{title}</div>
      {children}
    </div>
  );
}

function Row({ label, active, onClick, disabled, hint }: { label: string; active: boolean; onClick: () => void; disabled?: boolean; hint?: string }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={hint}
      className={cn(
        "flex w-full items-center justify-between rounded-lg px-2.5 py-1.5 text-xs transition-colors",
        disabled ? "cursor-not-allowed text-white/20" : active ? "bg-white/10 text-white" : "text-white/50 hover:bg-white/[0.06] hover:text-white/80",
      )}
    >
      <span>{label}</span>
    </button>
  );
}

function Seg<T extends string>({ value, opts, on }: { value: T; opts: { id: T; label: string; hint?: string }[]; on: (v: T) => void }) {
  return (
    <div className="flex flex-wrap gap-1 px-1">
      {opts.map((o) => (
        <button
          key={o.id}
          onClick={() => on(o.id)}
          title={o.hint}
          className={cn(
            "rounded-lg px-2.5 py-1 text-[11px] transition-colors",
            value === o.id ? "bg-white/10 text-white" : "text-white/50 hover:bg-white/[0.06] hover:text-white/80",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function GeometrySection({ showGeometry, setShowGeometry, c, layers, splatMeta, mesh }: {
  showGeometry: boolean; setShowGeometry: (v: boolean) => void;
  c: Ctrls; layers: LayersProps; splatMeta: SplatMeta | null; mesh: Meshed | null;
}) {
  const { cams, ground, debris, colors } = layers;
  const textureLabel = c.textured && cams ? "texture · projected" : c.textured && colors ? "texture · per-voxel" : "texture · off";
  return (
    <Section title="geometry">
      <Row label="Surface geometry" active={showGeometry} onClick={() => setShowGeometry(!showGeometry)} hint="mesh the occupancy into a surface, overlaid on the voxels" />
      {showGeometry && (
        <div className="mt-1.5 space-y-2 rounded-lg border border-white/[0.06] p-1.5">
          <Seg value={c.algo} on={(v) => c.setAlgo(v as Algo)} opts={[
            { id: "qef", label: "qef", hint: "sharp dual-contour surface (smooth; blobs sparse data)" },
            { id: "nets", label: "nets" },
            { id: "blocky", label: "blocky", hint: "each voxel a cube — no smoothing, the honest grid" },
          ]} />
          <Seg value={c.source} on={(v) => c.setSource(v as "occ3d" | "lidar")} opts={[
            { id: "occ3d", label: "0.4m occ3d", hint: "Occ3D 0.4m occupancy GT" },
            { id: "lidar", label: "0.2m lidar", hint: "39 LiDAR sweeps accumulated, movers removed — 2x finer, all measured" },
          ]} />
          <div className="space-y-1">
            <Row label="Mesh" active={c.showMesh} onClick={() => c.setShowMesh(!c.showMesh)} />
            <Row label={textureLabel} active={c.textured} onClick={() => c.setTextured(!c.textured)} disabled={!cams && !colors} hint="project the camera images onto the surface" />
            <Row label="Ground surface" active={c.showGround} onClick={() => c.setShowGround(!c.showGround)} disabled={!ground} hint="FREE drivable surface (honest holes under obstacles)" />
            <Row label="Fade debris" active={c.showDebris} onClick={() => c.setShowDebris(!c.showDebris)} disabled={!debris} hint="tiny isolated components faded as likely noise — toggle to inspect (nothing deleted)" />
            <Row label="Honest occlusion" active={c.occlude} onClick={() => c.setOcclude(!c.occlude)} disabled={!cams || !c.textured} hint="only texture a face a camera actually sees (depth pre-pass)" />
            <Row label="Gaussian splat" active={c.showSplat} onClick={() => c.setShowSplat(!c.showSplat)} disabled={!splatMeta} hint={splatMeta ? `image-based 3DGS (${splatMeta.count.toLocaleString()})` : "no splat asset"} />
          </div>
          {mesh && (
            <div className="px-1 font-mono text-[10px] text-white/40">
              {(mesh.tris / 1000).toFixed(1)}k tris · {mesh.ms}ms{c.algo === "qef" ? ` · ${mesh.defects} defects` : ""}
            </div>
          )}
        </div>
      )}
    </Section>
  );
}
