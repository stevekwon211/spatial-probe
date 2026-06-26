"use client";

import { useState } from "react";
import { X } from "lucide-react";
import { useTranslations } from "next-intl";
import { FINDINGS, VERDICT_COUNTS, type Finding } from "@/lib/findings";
import { DataTable, type Column } from "@/components/ui/data-table";
import { StatusChip } from "@/components/ui/status-chip";
import { GlassPanel } from "@/components/occquery/glass";
import { cn } from "@/lib/utils";

// The claim/detail/axis/verdict/gradedBy/dossier text is verbatim research content from
// experiments/*/results/summary.md (a primary source), so it stays as data — only the page
// chrome (headings, column labels, intro) is translated.
export function EvidenceLedger() {
  const t = useTranslations();
  const [open, setOpen] = useState<Finding | null>(null);

  const COLUMNS: Column<Finding>[] = [
    { key: "claim", header: t("evidence.columns.claim"), sortable: true, sortValue: (f) => f.claim, render: (f) => (
      <div>
        <div className="text-foreground">{f.claim}</div>
        <div className="mt-0.5 text-[11px] text-muted-foreground/70">{f.detail}</div>
      </div>
    ) },
    { key: "axis", header: t("evidence.columns.axis"), sortable: true, render: (f) => (
      <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground/60">{f.axis}</span>
    ) },
    { key: "verdict", header: t("evidence.columns.verdict"), sortable: true, sortValue: (f) => f.verdict, render: (f) => <StatusChip verdict={f.verdict} /> },
    { key: "gradedBy", header: t("evidence.columns.gradedBy"), render: (f) => (
      <span className="font-mono text-[11px] text-muted-foreground/70">{f.gradedBy}</span>
    ) },
  ];

  return (
    <div className="mx-auto w-full max-w-5xl px-6 py-10 text-foreground">
      <header className="mb-1 flex items-baseline justify-between gap-4">
        <h1 className="text-lg font-medium tracking-tight">{t("evidence.heading")}</h1>
        <div className="flex gap-3 font-mono text-[11px] text-muted-foreground/70 tabular-nums">
          {(Object.entries(VERDICT_COUNTS) as [string, number][])
            .filter(([, n]) => n > 0)
            .map(([v, n]) => <span key={v}>{t("evidence.countSuffix", { count: n, verdict: v.toLowerCase() })}</span>)}
        </div>
      </header>
      <p className="mb-6 max-w-2xl text-sm text-muted-foreground">
        {t("evidence.intro")}
      </p>

      <GlassPanel className="overflow-hidden">
        <DataTable columns={COLUMNS} rows={FINDINGS} onRowClick={setOpen} />
      </GlassPanel>

      {open && (
        <>
          <div className="fixed inset-0 z-40 bg-black/40" onClick={() => setOpen(null)} />
          <aside className="fixed top-0 right-0 z-50 flex h-full w-full max-w-md flex-col border-l border-white/10 bg-neutral-900/80 backdrop-blur-2xl">
            <div className="flex items-start justify-between gap-3 border-b border-white/10 px-5 py-4">
              <div>
                <div className="mb-1.5"><StatusChip verdict={open.verdict} /></div>
                <h2 className="text-sm font-medium leading-snug">{open.claim}</h2>
                <div className="mt-1 font-mono text-[11px] text-muted-foreground/60">{open.axis} · {open.gradedBy}</div>
              </div>
              <button onClick={() => setOpen(null)} className="text-muted-foreground/60 hover:text-foreground">
                <X className="size-4" />
              </button>
            </div>
            <div className="flex-1 space-y-3.5 overflow-y-auto px-5 py-5 text-[13px] leading-relaxed text-muted-foreground">
              {open.dossier.map((para, i) => (
                <p key={i} className={cn(i === 0 && "text-foreground/90")}>{para}</p>
              ))}
            </div>
          </aside>
        </>
      )}
    </div>
  );
}
