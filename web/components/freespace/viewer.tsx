"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { FreeSpaceScene } from "./freespace-scene";
import { fetchIndex, fetchFreespace, fetchOccGrid, type FreeSpace, type OccIndex } from "./data";
import { meshOccupancy, modelSdf, type Algo, type Meshed } from "./mdc-client";

// Free-space review: the "honest occupancy" view. Aggregated Occ3D -> a Rust/WASM QEF-MDC surface
// (sharp features) + a single-sweep corridor classified confirmed-free / fog / blocked, so a
// reviewer sees the confident surface AND how little one sweep verifies (the occlusion the mesh
// would otherwise hide). UI stays achromatic; color belongs to the data (corridor + confirmed pts).
export function FreeSpaceViewer() {
  const [idx, setIdx] = useState<OccIndex | null>(null);
  const [scene, setScene] = useState("");
  const [algo, setAlgo] = useState<Algo>("qef");
  const [showMesh, setShowMesh] = useState(true);
  const [mesh, setMesh] = useState<Meshed | null>(null);
  const [fs, setFs] = useState<FreeSpace | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => { fetchIndex().then((i) => { setIdx(i); if (i.scenes[0]) setScene(i.scenes[0]); }); }, []);

  useEffect(() => {
    if (!scene || !idx) return;
    let cancelled = false;
    setBusy(true); setMesh(null); setFs(null);
    (async () => {
      const [f, g] = await Promise.all([fetchFreespace(scene), fetchOccGrid(scene, idx)]);
      if (cancelled) return;
      setFs(f);
      const m = await meshOccupancy(g, algo);
      if (cancelled) return;
      setMesh(m); setBusy(false);
    })().catch(() => { if (!cancelled) setBusy(false); });
    return () => { cancelled = true; };
  }, [scene, algo, idx]);

  async function exportSdf() {
    if (!scene || !idx) return;
    const g = await fetchOccGrid(scene, idx);
    const sdf = await modelSdf(g);
    const blob = new Blob([sdf.buffer as ArrayBuffer], { type: "application/octet-stream" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${scene}_sdf_${idx.nx}x${idx.ny}x${idx.nz}_f32.bin`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  const c = fs?.corridor;
  return (
    <div className="flex h-[calc(100dvh-3.5rem)] flex-col gap-3 p-4">
      {/* controls */}
      <div className="flex flex-wrap items-center gap-3">
        <Select value={scene} onValueChange={(v: string | null) => v && setScene(v)}>
          <SelectTrigger className="w-40 font-mono text-xs"><SelectValue placeholder="scene" /></SelectTrigger>
          <SelectContent>
            {idx?.scenes.map((s) => <SelectItem key={s} value={s} className="font-mono text-xs">{s}</SelectItem>)}
          </SelectContent>
        </Select>

        <ToggleGroup value={[algo]} onValueChange={(v: string[]) => { if (v[0]) setAlgo(v[0] as Algo); }} variant="outline" size="sm">
          <ToggleGroupItem value="qef" className="font-mono text-xs">qef-MDC (sharp)</ToggleGroupItem>
          <ToggleGroupItem value="nets" className="font-mono text-xs">surface-nets</ToggleGroupItem>
        </ToggleGroup>

        <Button variant={showMesh ? "secondary" : "ghost"} size="sm" onClick={() => setShowMesh((v) => !v)}>mesh</Button>
        <Button variant="outline" size="sm" onClick={exportSdf} className="font-mono text-xs">↓ model SDF (.f32)</Button>

        {mesh && (
          <span className="font-mono text-xs text-muted-foreground">
            {(mesh.tris / 1000).toFixed(1)}k tris · {mesh.ms}ms
            {algo === "qef" && ` · ${mesh.fallback} planar-fallback · `}
            {algo === "qef" && <span className={mesh.defects === 0 ? "text-emerald-400" : "text-amber-400"}>{mesh.defects} defects</span>}
          </span>
        )}
        {busy && <Badge variant="outline" className="font-mono text-xs">meshing…</Badge>}

        <div className="ml-auto flex items-center gap-3 font-mono text-[11px] text-muted-foreground">
          <Legend c="#22d3ee" t="confirmed (1 sweep)" />
          <Legend c="#22c55e" t="free" />
          <Legend c="#f59e0b" t="fog" />
          <Legend c="#ef4444" t="blocked" />
        </div>
      </div>

      {/* scene + honest-gap */}
      <div className="grid min-h-0 flex-1 grid-cols-1 gap-3 lg:grid-cols-[1fr_20rem]">
        <div className="min-h-0 overflow-hidden rounded-xl border">
          <FreeSpaceScene mesh={mesh} fs={fs} showMesh={showMesh} />
        </div>
        <div className="flex flex-col gap-3">
          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-sm">the honest gap</CardTitle></CardHeader>
            <CardContent className="space-y-1.5 text-sm">
              <Row k="aggregated mesh says clear to" v={c?.aggregated_clearance != null ? `${c.aggregated_clearance} m` : "—"} />
              <Row k="1 sweep confirms free to" v={c?.confirmed_free_to != null ? `${c.confirmed_free_to} m` : "—"} amber />
              <Row k="single-sweep fog (unknown)" v={fs ? `${Math.round(fs.unknown_frac_single_sweep * 100)}%` : "—"} />
              <Row k="mesh triangles" v={mesh ? `${mesh.tris.toLocaleString()}` : "—"} />
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2"><CardTitle className="text-sm">reading</CardTitle></CardHeader>
            <CardContent className="text-xs leading-relaxed text-muted-foreground">
              solid 표면(파랑)은 여러 스윕을 합친 <b className="text-foreground">자신만만한 추측</b>. 청록 점은 <b className="text-foreground">한 번에 실제 확인</b>한 것.
              통로가 초록(free)이 아니라 대부분 <b className="text-foreground">amber(fog)</b>면 = 이 프레임에선 확인 못 한 곳 → 검수/자동라벨이 여기를 의심해야 함.
              qef-MDC는 EDT→SDF→QEF로 각진 구조를 코너에 얹고(surface-nets는 뭉갬), <b className="text-foreground">defects</b>는 아직 non-manifold인 셀 수(정직한 QA 신호).
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

function Row({ k, v, amber }: { k: string; v: string; amber?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <span className="text-muted-foreground">{k}</span>
      <span className={`font-mono tabular-nums ${amber ? "text-amber-400" : ""}`}>{v}</span>
    </div>
  );
}

function Legend({ c, t }: { c: string; t: string }) {
  return <span className="inline-flex items-center gap-1.5"><i className="size-2.5 rounded-sm" style={{ background: c }} />{t}</span>;
}
