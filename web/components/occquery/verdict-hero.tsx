"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { FINDINGS } from "@/lib/findings";

type Props = {
  lateralClearance: number | null;
  boxDistance: number | null;
  freePathBlocked: boolean;
};

const h1 = FINDINGS.find((f) => f.id === "h1")!;

// The per-frame claim sentence. The numbers are live measurements (kept as values); the surrounding
// prose is authored, so it resolves from messages with the formatted numbers as placeholders.
function useSentence() {
  const t = useTranslations();
  return (lc: number | null, bd: number | null): string => {
    if (lc === null || bd === null) {
      return t("occquery.verdictHero.noMeasurement");
    }
    const gap = `${(bd - lc).toFixed(2)} m`;
    const lcStr = `${lc.toFixed(2)} m`;
    const bdStr = `${bd.toFixed(2)} m`;

    if (bd > lc) {
      // Common case in this data: box ranges farther than occupancy clearance.
      return t("occquery.verdictHero.surfaceGap", { clearance: lcStr, boxDistance: bdStr, gap });
    }
    if (lc > bd) {
      // Occupancy clearance is looser than box distance. Be honest: do NOT claim it is globally tighter.
      return t("occquery.verdictHero.boxCloser", { boxDistance: bdStr, clearance: lcStr });
    }
    // bd === lc (exact match, rare)
    return t("occquery.verdictHero.agree", { boxDistance: bdStr });
  };
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

export function VerdictHero({ lateralClearance, boxDistance }: Props) {
  const t = useTranslations();
  const buildSentence = useSentence();
  const [showDossier, setShowDossier] = useState(false);

  const sentence = buildSentence(lateralClearance, boxDistance);

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
        [{showDossier ? t("occquery.verdictHero.hide") : t("occquery.verdictHero.why")}]
      </button>

      {/* H1 dossier verbatim */}
      {showDossier && (
        <div className="mt-2 space-y-1.5 border-t border-white/[0.06] pt-2">
          <div className="font-mono text-[9px] uppercase tracking-wider text-white/25">
            {t("occquery.verdictHero.dossierTitle")}
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
