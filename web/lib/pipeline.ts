// The research program, placed in order on the data pipeline it serves.
// Six falsifiable diagnostics, one per stage; status reflects the Python repo.

import { Database, RefreshCw, Search, ShieldCheck, type LucideIcon } from "lucide-react";

export type Status = "done" | "in-progress" | "planned";

export interface ModuleItem {
  id: string;
  i18nKey: string; // message key under pipeline.modules.* (title/axis/oneLine/statusLabel)
  title: string; // canonical English; display strings resolve from messages by i18nKey
  axis: string;
  oneLine: string;
  status: Status;
  statusLabel: string;
  href?: string; // set only when a page exists; otherwise the sidebar item is disabled
}

export interface Stage {
  id: string;
  i18nKey: string; // message key under pipeline.stages.* (name/blurb)
  name: string; // canonical English; display strings resolve from messages by i18nKey
  blurb: string;
  icon: LucideIcon;
  modules: ModuleItem[];
}

export const THESIS =
  "3D is queryable, updatable state, not the render.";

export const PIPELINE: Stage[] = [
  {
    id: "ingest",
    i18nKey: "ingest",
    name: "Ingest",
    blurb: "Raw multi-sensor logs become 3D state: objects, HD map, occupancy.",
    icon: Database,
    modules: [
      {
        id: "asof",
        i18nKey: "asof",
        title: "ASOF",
        axis: "render to state",
        oneLine:
          "Does the converted state preserve action-relevant signal, or only look right? An acceptance gate at the door.",
        status: "planned",
        statusLabel: "Planned",
      },
    ],
  },
  {
    id: "search",
    i18nKey: "search",
    name: "Search",
    blurb:
      "Find scenes by measured physical quantities, reproducibly. No hallucination.",
    icon: Search,
    modules: [
      {
        id: "occquery",
        i18nKey: "occquery",
        title: "OccQuery",
        axis: "geometry / occupancy",
        oneLine:
          "Occupancy-native predicates measure box-blind free-space geometry (clearance, free-width) that object-box query languages cannot express.",
        status: "in-progress",
        statusLabel: "H1 expressivity holds · H3 inconclusive",
        href: "/occquery",
      },
    ],
  },
  {
    id: "qa",
    i18nKey: "qa",
    name: "QA / Review",
    blurb: "Decide where auto-labels are safe and where a human must look.",
    icon: ShieldCheck,
    modules: [
      {
        id: "gt-distrust",
        i18nKey: "gtDistrust",
        title: "GT-distrust",
        axis: "visibility",
        oneLine:
          "Occlusion geometry predicts which occupancy labels are untrustworthy. A geometric review trigger.",
        status: "planned",
        statusLabel: "Planned",
      },
      {
        id: "calibration",
        i18nKey: "calibration",
        title: "Visibility Calibration",
        axis: "uncertainty",
        oneLine:
          "Is the model's confidence honest where the sensor cannot see? Catch the silent overconfidence.",
        status: "planned",
        statusLabel: "Planned",
      },
    ],
  },
  {
    id: "loop",
    i18nKey: "loop",
    name: "Curation / Loop",
    blurb: "Pick what to fix and what to label so the model actually improves.",
    icon: RefreshCw,
    modules: [
      {
        id: "value",
        i18nKey: "value",
        title: "Value-of-Correction",
        axis: "valuation",
        oneLine:
          "Rank scenes by how much fixing a label actually moves the model, not how many are wrong.",
        status: "planned",
        statusLabel: "Planned",
      },
      {
        id: "dynfield",
        i18nKey: "dynfield",
        title: "DynField-Necessity",
        axis: "dynamics / time",
        oneLine:
          "Which stored motion fields a planner truly needs, and in which moments (cut-in, hard brake, low TTC).",
        status: "in-progress",
        statusLabel: "Tier-1: velocity redundant when safe",
      },
    ],
  },
];

export const ALL_MODULES = PIPELINE.flatMap((s) => s.modules);
export const TOTAL = ALL_MODULES.length;
export const SHIPPED = ALL_MODULES.filter((m) => m.status === "done").length;
export const IN_PROGRESS = ALL_MODULES.filter(
  (m) => m.status === "in-progress",
).length;
export const PIPELINE_RUNNABLE = SHIPPED === TOTAL;
