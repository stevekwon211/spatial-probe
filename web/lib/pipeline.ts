// The research program, placed on the data-OS pipeline it serves.
// Each module is one falsifiable diagnostic; status reflects the Python repo.

export type Status = "in-progress" | "planned";

export interface ResearchModule {
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
  modules: ResearchModule[];
}

export const THESIS =
  "3D's essence is queryable, updatable state — not the render.";

export const PIPELINE: Stage[] = [
  {
    id: "ingest",
    name: "Ingest / Data Engine",
    blurb: "Raw multi-sensor logs become 3D state: objects, HD map, occupancy.",
    modules: [
      {
        id: "asof",
        title: "ASOF",
        axis: "render → state",
        oneLine:
          "Does the converted state preserve action-relevant signal, or only look right? An acceptance gate at the door.",
        status: "planned",
        statusLabel: "planned",
      },
      {
        id: "gt-distrust-ingest",
        title: "GT-distrust",
        axis: "visibility",
        oneLine:
          "Occlusion geometry predicts which occupancy labels are untrustworthy — before they pollute training.",
        status: "planned",
        statusLabel: "planned",
      },
    ],
  },
  {
    id: "search",
    name: "Search",
    blurb:
      "Find scenes by measured physical quantities — reproducibly, no hallucination.",
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
    modules: [
      {
        id: "gt-distrust-qa",
        title: "GT-distrust",
        axis: "visibility",
        oneLine:
          "A geometric review trigger — point the checkers at the labels most likely wrong.",
        status: "planned",
        statusLabel: "planned",
      },
      {
        id: "calibration",
        title: "Visibility Calibration",
        axis: "uncertainty",
        oneLine:
          "Is the model's confidence honest where the sensor cannot see? Catch the silent overconfidence.",
        status: "planned",
        statusLabel: "planned",
      },
    ],
  },
  {
    id: "loop",
    name: "Curation / Loop",
    blurb: "Pick what to fix and what to label so the model actually improves.",
    modules: [
      {
        id: "value",
        title: "Value-of-Correction",
        axis: "valuation",
        oneLine:
          "Rank scenes by how much fixing a label actually moves the model — not by how many are wrong.",
        status: "planned",
        statusLabel: "planned",
      },
      {
        id: "dynfield",
        title: "DynField-Necessity",
        axis: "dynamics / time",
        oneLine:
          "Which stored motion fields a planner truly needs — and in which moments (cut-in, hard brake, low TTC).",
        status: "planned",
        statusLabel: "planned",
      },
    ],
  },
];
