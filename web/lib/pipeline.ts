// The research program, placed in order on the data pipeline it serves.
// Six falsifiable diagnostics, one per stage; status reflects the Python repo.

import { Database, RefreshCw, Search, ShieldCheck, type LucideIcon } from "lucide-react";

export type Status = "done" | "in-progress" | "planned";

export interface ModuleItem {
  id: string;
  title: string;
  axis: string;
  oneLine: string;
  status: Status;
  statusLabel: string;
}

export interface Stage {
  id: string;
  name: string;
  blurb: string;
  icon: LucideIcon;
  modules: ModuleItem[];
}

export const THESIS =
  "3D's essence is queryable, updatable state — not the render.";

export const PIPELINE: Stage[] = [
  {
    id: "ingest",
    name: "Ingest",
    blurb: "Raw multi-sensor logs become 3D state: objects, HD map, occupancy.",
    icon: Database,
    modules: [
      {
        id: "asof",
        title: "ASOF",
        axis: "render → state",
        oneLine:
          "Does the converted state preserve action-relevant signal, or only look right? An acceptance gate at the door.",
        status: "planned",
        statusLabel: "Planned",
      },
    ],
  },
  {
    id: "search",
    name: "Search",
    blurb:
      "Find scenes by measured physical quantities — reproducibly, no hallucination.",
    icon: Search,
    modules: [
      {
        id: "occquery",
        title: "OccQuery",
        axis: "geometry / occupancy",
        oneLine:
          "Occupancy-native predicates (clearance, free-path) retrieve scenes that object-box query languages cannot express.",
        status: "in-progress",
        statusLabel: "M1 · core primitive",
      },
    ],
  },
  {
    id: "qa",
    name: "QA / Review",
    blurb: "Decide where auto-labels are safe and where a human must look.",
    icon: ShieldCheck,
    modules: [
      {
        id: "gt-distrust",
        title: "GT-distrust",
        axis: "visibility",
        oneLine:
          "Occlusion geometry predicts which occupancy labels are untrustworthy — a geometric review trigger.",
        status: "planned",
        statusLabel: "Planned",
      },
      {
        id: "calibration",
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
    name: "Curation / Loop",
    blurb: "Pick what to fix and what to label so the model actually improves.",
    icon: RefreshCw,
    modules: [
      {
        id: "value",
        title: "Value-of-Correction",
        axis: "valuation",
        oneLine:
          "Rank scenes by how much fixing a label actually moves the model — not by how many are wrong.",
        status: "planned",
        statusLabel: "Planned",
      },
      {
        id: "dynfield",
        title: "DynField-Necessity",
        axis: "dynamics / time",
        oneLine:
          "Which stored motion fields a planner truly needs — and in which moments (cut-in, hard brake, low TTC).",
        status: "planned",
        statusLabel: "Planned",
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
