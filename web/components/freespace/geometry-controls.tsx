"use client";

import { Button } from "@/components/ui/button";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import type { GeometryControls as Ctrls } from "./use-freespace-geometry";
import type { Algo, Meshed } from "./mdc-client";
import type { LayersProps } from "./freespace-scene";
import type { SplatMeta } from "./data";

// The free-space geometry toggle strip, shared by the standalone view and the occquery Explorer.
export function GeometryControls({ c, layers, splatMeta, mesh }: {
  c: Ctrls; layers: LayersProps; splatMeta: SplatMeta | null; mesh: Meshed | null;
}) {
  const { cams, ground, debris, colors } = layers;
  return (
    <div className="flex flex-wrap items-center gap-2">
      <ToggleGroup value={[c.algo]} onValueChange={(v: string[]) => v[0] && c.setAlgo(v[0] as Algo)} variant="outline" size="sm">
        <ToggleGroupItem value="qef" className="font-mono text-xs" title="QEF Manifold Dual Contouring — sharp corners, smooth (blobs sparse data)">qef</ToggleGroupItem>
        <ToggleGroupItem value="nets" className="font-mono text-xs">nets</ToggleGroupItem>
        <ToggleGroupItem value="blocky" className="font-mono text-xs" title="each voxel as a cube — no smoothing, the honest grid">blocky</ToggleGroupItem>
      </ToggleGroup>
      <ToggleGroup value={[c.source]} onValueChange={(v: string[]) => v[0] && c.setSource(v[0] as "occ3d" | "lidar")} variant="outline" size="sm">
        <ToggleGroupItem value="occ3d" className="font-mono text-xs" title="Occ3D 0.4m occupancy GT">0.4m</ToggleGroupItem>
        <ToggleGroupItem value="lidar" className="font-mono text-xs" title="0.2m — 39 LiDAR sweeps accumulated, movers removed (2x finer, all measured)">0.2m lidar</ToggleGroupItem>
      </ToggleGroup>
      <Button variant={c.showMesh ? "secondary" : "ghost"} size="sm" onClick={() => c.setShowMesh((v) => !v)}>mesh</Button>
      <Button variant={c.textured ? "secondary" : "ghost"} size="sm" onClick={() => c.setTextured((v) => !v)} disabled={!cams && !colors}
        title={cams ? "render-time projective texturing" : colors ? "per-voxel camera color" : "no camera images"}>
        {c.textured && cams ? "projected" : c.textured && colors ? "textured" : "shaded"}
      </Button>
      <Button variant={c.showGround ? "secondary" : "ghost"} size="sm" onClick={() => c.setShowGround((v) => !v)} disabled={!ground}
        title="FREE drivable surface (honest holes under obstacles)">ground</Button>
      <Button variant={c.showDebris ? "secondary" : "ghost"} size="sm" onClick={() => c.setShowDebris((v) => !v)} disabled={!debris}
        title="tiny isolated components — faded as likely noise; toggle to inspect (nothing deleted)">debris</Button>
      <Button variant={c.occlude ? "secondary" : "ghost"} size="sm" onClick={() => c.setOcclude((v) => !v)} disabled={!cams || !c.textured}
        title="honest occlusion: only texture a face a camera actually sees">occlude</Button>
      <Button variant={c.showSplat && splatMeta ? "secondary" : "ghost"} size="sm" onClick={() => c.setShowSplat((v) => !v)} disabled={!splatMeta}
        title={splatMeta ? `image-based 3D Gaussian splat (${splatMeta.count.toLocaleString()})` : "no splat asset"}>splat</Button>
      {mesh && (
        <span className="font-mono text-xs text-muted-foreground">
          {(mesh.tris / 1000).toFixed(1)}k tris · {mesh.ms}ms{c.algo === "qef" ? ` · ${mesh.defects} defects` : ""}
        </span>
      )}
    </div>
  );
}
