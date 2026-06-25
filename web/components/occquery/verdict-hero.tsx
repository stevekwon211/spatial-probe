"use client";

import { useState } from "react";
import { FINDINGS } from "@/lib/findings";

type Props = {
  lateralClearance: number | null;
  boxDistance: number | null;
  freePathBlocked: boolean;
};

const h1 = FINDINGS.find((f) => f.id === "h1")!;

function buildSentence(
  lc: number | null,
  bd: number | null,
  blocked: boolean,
): string {
  if (lc === null || bd === null) {
    return "No measurement available for this frame — occupancy predicate returned null.";
  }

  const gap = (bd - lc).toFixed(2);
  const lcStr = lc.toFixed(2);
  const bdStr = bd.toFixed(2);

  if (bd > lc) {
    // Common case in this data: box ranges farther than occupancy clearance.
    // Occupancy sees a surface closer than the box pipeline can name.
    return `A solid surface ${lcStr} m off the ego that the box pipeline ranges at ${bdStr} m — a ${gap} m gap no box query can name.`;
  }

  if (lc > bd) {
    // Occupancy clearance is looser than box distance.
    // Be honest: do NOT claim occupancy is globally more conservative.
    return `Box ranges the nearest object at ${bdStr} m; occupancy reads ${lcStr} m of reachable clearance here.`;
  }

  // bd === lc (exact match, rare)
  return `Box and occupancy agree at ${bdStr} m lateral — no gap between the two measurements for this frame.`;
}

function Metric({ v }: { v: string }) {
  return (
    <span
      className="metric"
      style={{ color: "var(--data-text)", fontFamily: "var(--font-mono)" }}
    >
      {v}
    </span>
  );
}

function renderSentence(raw: string) {
  // Inline-highlight every number that ends in " m" in teal mono.
  // Pattern: digits with optional decimal, followed by " m"
  const parts = raw.split(/(\d+\.\d+\s*m\b)/g);
  return parts.map((part, i) =>
    /^\d+\.\d+\s*m\b/.test(part) ? (
      <Metric key={i} v={part} />
    ) : (
      <span key={i}>{part}</span>
    ),
  );
}

export function VerdictHero({ lateralClearance, boxDistance, freePathBlocked }: Props) {
  const [showDossier, setShowDossier] = useState(false);

  const sentence = buildSentence(lateralClearance, boxDistance, freePathBlocked);

  return (
    <div className="rounded-xl border border-white/[0.07] bg-white/[0.03] p-3">
      {/* claim sentence */}
      <p
        className="text-sm leading-snug text-white/85"
        style={{ fontFamily: "var(--font-geist-sans)" }}
      >
        {renderSentence(sentence)}
      </p>

      {/* [why] toggle */}
      <button
        onClick={() => setShowDossier((v) => !v)}
        className="mt-2 font-mono text-[10px] tracking-wide text-white/30 transition-colors hover:text-white/60"
        aria-expanded={showDossier}
      >
        [{showDossier ? "hide" : "why"}]
      </button>

      {/* H1 dossier verbatim */}
      {showDossier && (
        <div className="mt-2 space-y-1.5 border-t border-white/[0.06] pt-2">
          <div className="font-mono text-[9px] uppercase tracking-wider text-white/25">
            H1 dossier — verbatim
          </div>
          {h1.dossier.map((para, i) => (
            <p key={i} className="text-[11px] leading-relaxed text-white/45">
              {para}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}
