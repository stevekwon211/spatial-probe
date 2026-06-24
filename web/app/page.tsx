import Link from "next/link";
import { ArrowRight, GitBranch, ScanLine, Terminal } from "lucide-react";
import { GlassPanel } from "@/components/occquery/glass";
import { StatusChip } from "@/components/ui/status-chip";
import { ALL_MODULES, IN_PROGRESS, SHIPPED, THESIS, TOTAL } from "@/lib/pipeline";

export const metadata = {
  title: "spatial-probe — the spatial data engine for AV and robotics",
  description:
    "Query, measure, and trust 3D scene state by what it physically means — free-space, clearance, corridor width, dynamics — with every number you can re-run.",
};

function Logo({ className = "size-6" }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={`${className} shrink-0`} fill="none" aria-hidden="true">
      <g stroke="currentColor" strokeWidth={2}>
        <rect x="3" y="3" width="10" height="10" />
        <rect x="3" y="19" width="10" height="10" />
        <rect x="19" y="19" width="10" height="10" />
      </g>
      <rect x="18" y="2" width="12" height="12" fill="#2BB0A4" />
    </svg>
  );
}

const lifecycle = (s: string) => (s === "in-progress" ? "live" : s === "done" ? "shipped" : "planned");

// The signature output, drawn top-down (BEV): occupancy structure as voxel cells, the ego, and the
// free-space measurement (teal) that the box-only baseline (faint, to the nearest object box) is blind
// to. Deterministic SVG -- no Math.random, so server and client render identically.
const RAMP = ["#4b6b63", "#5f7a52", "#7d8348", "#94795a", "#8a6450"]; // muted occupancy depth ramp
function Structure({ x, y, cols, rows }: { x: number; y: number; cols: number; rows: number }) {
  const cells = [];
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) {
      // eroded edge for an organic massing (drop a few corner cells deterministically)
      if ((r + c) % 7 === 0 && (r === 0 || r === rows - 1 || c === 0 || c === cols - 1)) continue;
      cells.push(<rect key={`${r}-${c}`} x={x + c * 11} y={y + r * 11} width={9.5} height={9.5} rx={1} fill={RAMP[r % RAMP.length]} opacity={0.92} />);
    }
  return <g>{cells}</g>;
}

function OccupancyDiagram() {
  return (
    <svg viewBox="0 0 800 380" className="block w-full" role="img" aria-label="Top-down occupancy: free-space measured where boxes are blind">
      <rect width="800" height="380" fill="#0b0b0b" />
      {/* faint ground grid */}
      <g stroke="rgba(255,255,255,0.05)" strokeWidth="1">
        {Array.from({ length: 9 }, (_, i) => <line key={`v${i}`} x1={i * 100} y1={0} x2={i * 100} y2={380} />)}
        {Array.from({ length: 5 }, (_, i) => <line key={`h${i}`} x1={0} y1={i * 95} x2={800} y2={i * 95} />)}
      </g>
      {/* occupancy structures left + right, free corridor down the middle */}
      <Structure x={84} y={150} cols={8} rows={10} />
      <Structure x={56} y={44} cols={7} rows={7} />
      <Structure x={604} y={132} cols={8} rows={11} />
      <Structure x={664} y={52} cols={6} rows={6} />
      {/* the nearest object BOX (what box-only sees) -- a vehicle ahead, drawn as an outline */}
      <rect x={300} y={70} width={30} height={46} rx={3} fill="none" stroke="rgba(255,255,255,0.4)" strokeWidth="1.5" />
      <text x={315} y={62} textAnchor="middle" className="font-mono" fontSize="10" fill="rgba(255,255,255,0.35)">vehicle</text>
      {/* ego */}
      <rect x={388} y={300} width={16} height={28} rx={2} fill="#1a1a1a" stroke="#2BB0A4" strokeWidth="1.5" />
      <line x1={396} y1={300} x2={396} y2={284} stroke="#2BB0A4" strokeWidth="1.5" />
      {/* box-only distance: faint dashed to the box center (9.49 m) */}
      <line x1={396} y1={300} x2={315} y2={93} stroke="rgba(255,255,255,0.28)" strokeWidth="1.25" strokeDasharray="4 4" />
      <text x={372} y={196} textAnchor="end" className="font-mono" fontSize="11" fill="rgba(255,255,255,0.4)">9.49 m · box</text>
      {/* occupancy clearance: teal, to the nearest STRUCTURE edge (5.08 m) -- box-blind */}
      <g stroke="#2BB0A4" strokeWidth="1.75">
        <line x1={388} y1={250} x2={172} y2={250} />
        <line x1={388} y1={244} x2={388} y2={256} />
        <line x1={172} y1={244} x2={172} y2={256} />
      </g>
      <text x={280} y={241} textAnchor="middle" className="font-mono" fontSize="12" fill="#42c5b8">5.08 m · free space</text>
    </svg>
  );
}

export default function Landing() {
  return (
    <div className="min-h-screen bg-[#080808] text-white antialiased">
      {/* nav */}
      <header className="sticky top-0 z-30 border-b border-white/[0.06] bg-[#080808]/70 backdrop-blur-xl">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-3.5">
          <div className="flex items-center gap-2 text-sm font-medium tracking-tight">
            <Logo className="size-5" />
            spatial-probe
          </div>
          <nav className="flex items-center gap-1 text-[13px]">
            <Link href="/overview" className="rounded-md px-3 py-1.5 text-white/55 transition-colors hover:text-white">Pipeline</Link>
            <Link href="/evidence" className="rounded-md px-3 py-1.5 text-white/55 transition-colors hover:text-white">Evidence</Link>
            <Link href="/occquery" className="ml-1 inline-flex items-center gap-1.5 rounded-lg bg-white px-3.5 py-1.5 font-medium text-black transition-opacity hover:opacity-90">
              Open Explorer <ArrowRight className="size-3.5" />
            </Link>
          </nav>
        </div>
      </header>

      {/* hero */}
      <section className="relative overflow-hidden">
        {/* faint voxel-grid backdrop */}
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 opacity-[0.18] [mask-image:radial-gradient(ellipse_at_top,black,transparent_70%)]"
          style={{ backgroundImage: "linear-gradient(rgba(255,255,255,.06) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.06) 1px,transparent 1px)", backgroundSize: "28px 28px" }}
        />
        <div className="relative mx-auto max-w-3xl px-6 pt-24 pb-12 text-center sm:pt-32">
          <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-3 py-1 font-mono text-[11px] uppercase tracking-[0.18em] text-white/45">
            <span className="size-1.5 rounded-full" style={{ backgroundColor: "#2BB0A4" }} />
            state, not render
          </div>
          <h1 className="text-balance text-[2.1rem] font-medium leading-[1.08] tracking-tight sm:text-5xl">
            Query, measure, and trust 3D scene state by what it physically means.
          </h1>
          <p className="mx-auto mt-6 max-w-xl text-balance text-[15px] leading-relaxed text-white/55">
            Free-space, clearance, corridor width, dynamics — geometry an object-box pipeline cannot
            express, with every number you can re-run. The spatial data engine for AV and robotics.
          </p>
          <div className="mt-9 flex items-center justify-center gap-3">
            <Link href="/occquery" className="inline-flex items-center gap-1.5 rounded-lg bg-white px-5 py-2.5 text-sm font-medium text-black transition-opacity hover:opacity-90">
              Open Explorer <ArrowRight className="size-4" />
            </Link>
            <Link href="/evidence" className="rounded-lg border border-white/15 px-5 py-2.5 text-sm text-white/80 transition-colors hover:border-white/40 hover:text-white">
              See the evidence
            </Link>
          </div>
        </div>

        {/* product window — the signature output: free-space measured where boxes are blind */}
        <div className="relative mx-auto max-w-4xl px-6 pb-20">
          <Link href="/occquery" className="group block overflow-hidden rounded-2xl border border-white/10 bg-neutral-950 shadow-2xl shadow-black/60 transition-colors hover:border-white/20">
            <div className="flex items-center gap-2 border-b border-white/[0.07] bg-white/[0.02] px-4 py-2.5">
              <span className="size-2.5 rounded-full bg-white/15" />
              <span className="size-2.5 rounded-full bg-white/15" />
              <span className="size-2.5 rounded-full bg-white/15" />
              <span className="ml-3 font-mono text-[11px] text-white/35">explorer · scene-0061 · top-down</span>
              <span className="ml-auto inline-flex items-center gap-1 font-mono text-[11px] text-white/40 opacity-0 transition-opacity group-hover:opacity-100">open live <ArrowRight className="size-3" /></span>
            </div>
            <OccupancyDiagram />
          </Link>
        </div>
      </section>

      {/* the gap — the moat, shown */}
      <section className="mx-auto max-w-4xl px-6 py-12">
        <div className="grid items-center gap-8 sm:grid-cols-2">
          <div>
            <div className="mb-3 font-mono text-[11px] uppercase tracking-wider text-white/35">the gap</div>
            <p className="text-balance text-lg leading-relaxed text-white/80">
              Object-box pipelines store boxes. Planners drive through free space. spatial-probe measures
              the free space — a wall 5.08&nbsp;m beside the ego that box-only never sees, because its
              nearest object box reads 9.49&nbsp;m away.
            </p>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-xl border border-white/12 bg-white/[0.03] p-5 text-center">
              <div className="font-mono text-3xl tabular-nums text-white">5.08<span className="ml-0.5 text-base text-white/40">m</span></div>
              <div className="mt-1.5 text-[11px] text-white/50">occupancy</div>
              <div className="mt-0.5 font-mono text-[10px] text-white/30">lateral_clearance</div>
            </div>
            <div className="rounded-xl border border-white/[0.07] bg-white/[0.015] p-5 text-center">
              <div className="font-mono text-3xl tabular-nums text-white/50">9.49<span className="ml-0.5 text-base text-white/25">m</span></div>
              <div className="mt-1.5 text-[11px] text-white/40">box-only</div>
              <div className="mt-0.5 font-mono text-[10px] text-white/25">box_distance</div>
            </div>
            <div className="col-span-2 text-center font-mono text-[10px] text-white/25">scene-0061 · frame 10 · re-runnable via probe_scene()</div>
          </div>
        </div>
      </section>

      {/* capability map */}
      <section className="mx-auto max-w-5xl px-6 py-12">
        <div className="mb-1.5 flex items-baseline justify-between">
          <h2 className="text-xl font-medium tracking-tight">The pipeline, as capabilities</h2>
          <Link href="/overview" className="font-mono text-[11px] text-white/40 transition-colors hover:text-white/70">{SHIPPED}/{TOTAL} shipped · {IN_PROGRESS} live →</Link>
        </div>
        <p className="mb-6 max-w-xl text-sm text-white/40">Six axes of the data pipeline. Two are live. The rest are planned, and the rail says so.</p>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {ALL_MODULES.map((m) => {
            const live = m.status !== "planned";
            const card = (
              <div className={`flex h-full flex-col rounded-xl border bg-white/[0.02] p-4 transition-colors ${live ? "border-white/12 hover:border-white/30 hover:bg-white/[0.04]" : "border-white/[0.06] opacity-60"}`}>
                <div className="flex items-center justify-between gap-2">
                  <span className="text-sm font-medium">{m.title}</span>
                  <span className={`font-mono text-[9px] uppercase tracking-wider ${live ? "text-white/45" : "text-white/30"}`}>{lifecycle(m.status)}</span>
                </div>
                <div className="mt-0.5 font-mono text-[10px] uppercase tracking-wide text-white/30">{m.axis}</div>
                <p className="mt-2.5 text-[12px] leading-relaxed text-white/45">{m.oneLine}</p>
              </div>
            );
            return m.href ? <Link key={m.id} href={m.href} className="block">{card}</Link> : <div key={m.id}>{card}</div>;
          })}
        </div>
      </section>

      {/* auditable measurement */}
      <section className="mx-auto max-w-5xl px-6 py-12">
        <h2 className="mb-1.5 text-xl font-medium tracking-tight">Measurement you can audit</h2>
        <p className="mb-6 max-w-xl text-sm text-white/40">A vendor whose own audit kills its own results is the one to trust.</p>
        <div className="grid gap-3 sm:grid-cols-3">
          {[
            { icon: Terminal, title: "Reproducible", body: "Every number carries its call. probe_scene(scene, frame, predicate) regenerates it bit-for-bit." },
            { icon: GitBranch, title: "Pre-registered", body: "Hypothesis, analysis path, and kill criterion are committed before the data. HARKing shows in the diff." },
            { icon: ScanLine, title: "Independently graded", body: "H1 holds oracle-free. H3 is inconclusive — audited twice and retracted in public.", href: "/evidence" },
          ].map((c) => {
            const inner = (
              <GlassPanel className="flex h-full flex-col p-5 transition-colors hover:border-white/20">
                <c.icon className="size-4 text-white/40" />
                <div className="mt-3.5 text-[15px] font-medium">{c.title}</div>
                <p className="mt-1.5 flex-1 text-[13px] leading-relaxed text-white/50">{c.body}</p>
                {c.href && <div className="mt-3 inline-flex items-center gap-1 font-mono text-[10px] text-white/35">see Evidence <ArrowRight className="size-3" /></div>}
              </GlassPanel>
            );
            return c.href ? <Link key={c.title} href={c.href}>{inner}</Link> : <div key={c.title}>{inner}</div>;
          })}
        </div>
        <div className="mt-5 flex flex-wrap items-center gap-x-4 gap-y-2 text-[12px] text-white/40">
          <span className="font-mono text-[10px] uppercase tracking-wider text-white/30">verdicts on file</span>
          <span className="inline-flex items-center gap-2"><StatusChip verdict="HOLDS" /> H1 expressivity</span>
          <span className="inline-flex items-center gap-2"><StatusChip verdict="INCONCLUSIVE" /> H3 denotation</span>
        </div>
      </section>

      {/* access band */}
      <section className="mx-auto max-w-5xl px-6 py-12">
        <GlassPanel className="flex flex-col items-center gap-5 p-8 text-center sm:flex-row sm:justify-between sm:text-left">
          <div>
            <div className="text-lg font-medium tracking-tight">One engine, three front doors.</div>
            <div className="mt-1 text-sm text-white/50">Explorer for the eye. A typed predicate for the query. MCP for the agent.</div>
          </div>
          <Link href="/occquery" className="inline-flex shrink-0 items-center gap-1.5 rounded-lg bg-white px-5 py-2.5 text-sm font-medium text-black transition-opacity hover:opacity-90">
            Open Explorer <ArrowRight className="size-4" />
          </Link>
        </GlassPanel>
      </section>

      {/* footer */}
      <footer className="mx-auto max-w-5xl px-6 pb-12 pt-6">
        <div className="flex flex-col gap-3 border-t border-white/[0.08] pt-6 text-[12px] leading-relaxed text-white/35 sm:flex-row sm:items-center sm:justify-between">
          <p className="max-w-2xl">Runs on Occ3D-nuScenes and a deterministic predicate core. No validated F1 is claimed where no independent oracle exists. Negatives are headlines here, not footnotes.</p>
          <div className="flex shrink-0 items-center gap-2 text-white/30" title={THESIS}>
            <Logo className="size-4" /> state, not render
          </div>
        </div>
      </footer>
    </div>
  );
}
