"use client";

import Link from "next/link";
import { Plus } from "lucide-react";
import { useTranslations } from "next-intl";
import { ReactFlow, Background, BackgroundVariant, Handle, Position, type Edge, type Node, type NodeProps } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { IN_PROGRESS, PIPELINE, SHIPPED, THESIS, TOTAL } from "@/lib/pipeline";
import { GlassPanel } from "@/components/occquery/glass";
import { LocaleToggle } from "@/components/locale-toggle";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

type ExpData = { i18nKey: string; status: string; href?: string };
type StageData = { no: string; i18nKey: string };

const TOOLTIP = "max-w-[230px] rounded-lg border border-white/10 bg-neutral-900/95 leading-relaxed text-white/80 backdrop-blur-xl";

function statusLabel(status: string) {
  if (status === "in-progress") return "live";
  if (status === "done") return "shipped";
  return "planned";
}

function StageLabelNode({ data }: NodeProps) {
  const t = useTranslations();
  const d = data as unknown as StageData;
  return (
    <div className="pointer-events-none flex items-center gap-2 whitespace-nowrap">
      <span className="font-mono text-[11px] text-white/25">{d.no}</span>
      <span className="text-[11px] font-medium uppercase tracking-[0.2em] text-white/45">{t(`pipeline.stages.${d.i18nKey}.name`)}</span>
    </div>
  );
}

function ExperimentNode({ data }: NodeProps) {
  const t = useTranslations();
  const d = data as unknown as ExpData;
  const active = d.status === "in-progress" || d.status === "done";
  const card = (
    <div
      className={cn(
        "w-52 rounded-2xl border bg-neutral-900/60 p-3.5 backdrop-blur-xl transition-colors",
        active ? "border-white/15 hover:border-white/40" : "border-white/[0.07] opacity-55",
      )}
    >
      <Handle type="target" position={Position.Left} className="!h-1.5 !w-1.5 !border-0 !bg-white/25" />
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm font-medium text-white">{t(`pipeline.modules.${d.i18nKey}.title`)}</span>
        <span
          className={cn(
            "shrink-0 rounded-full px-2 py-0.5 text-[9px] uppercase tracking-wider",
            active ? "bg-white/15 text-white/80" : "bg-white/[0.06] text-white/40",
          )}
        >
          {t(`overview.status.${statusLabel(d.status)}`)}
        </span>
      </div>
      <div className="mt-1 text-[10px] uppercase tracking-wide text-white/35">{t(`pipeline.modules.${d.i18nKey}.axis`)}</div>
      <Handle type="source" position={Position.Right} className="!h-1.5 !w-1.5 !border-0 !bg-white/25" />
    </div>
  );
  const inner = d.href ? (
    <Link href={d.href} className="block">
      {card}
    </Link>
  ) : (
    card
  );
  return (
    <Tooltip>
      <TooltipTrigger render={inner} />
      <TooltipContent side="bottom" sideOffset={8} className={TOOLTIP}>
        {t(`pipeline.modules.${d.i18nKey}.oneLine`)}
      </TooltipContent>
    </Tooltip>
  );
}

const nodeTypes = { experiment: ExperimentNode, stageLabel: StageLabelNode };

const COL = 340;
const ROW = 170;
const LABEL_Y = -125;

function buildGraph(): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = [];
  const edges: Edge[] = [];
  PIPELINE.forEach((stage, si) => {
    nodes.push({
      id: `stage-${stage.id}`,
      type: "stageLabel",
      position: { x: si * COL + 8, y: LABEL_Y },
      data: { no: String(si + 1).padStart(2, "0"), i18nKey: stage.i18nKey },
      selectable: false,
    });
    const mods = stage.modules;
    mods.forEach((m, mi) => {
      nodes.push({
        id: m.id,
        type: "experiment",
        position: { x: si * COL, y: mi * ROW - ((mods.length - 1) * ROW) / 2 },
        data: { i18nKey: m.i18nKey, status: m.status, href: m.href },
      });
    });
    if (si < PIPELINE.length - 1) {
      const next = PIPELINE[si + 1];
      mods.forEach((m) => {
        next.modules.forEach((nm) => {
          edges.push({
            id: `${m.id}-${nm.id}`,
            source: m.id,
            target: nm.id,
            animated: m.status === "in-progress",
            style: { stroke: "rgba(255,255,255,0.12)", strokeWidth: 1 },
          });
        });
      });
    }
  });
  return { nodes, edges };
}

const { nodes, edges } = buildGraph();

export default function Home() {
  const t = useTranslations();
  return (
    <div className="relative h-screen w-full bg-[#080808]">
      <GlassPanel className="absolute top-4 left-4 z-10 flex w-56 flex-col text-white">
        <div className="flex items-center justify-between px-3 pt-3 pb-2.5">
          <Tooltip>
            <TooltipTrigger
              render={<div className="w-fit cursor-default text-sm font-medium tracking-tight">{t("brand.name")}</div>}
            />
            <TooltipContent side="bottom" align="start" className={TOOLTIP}>
              {THESIS}
            </TooltipContent>
          </Tooltip>
          <LocaleToggle />
        </div>
        <div className="border-t border-white/10 px-3 py-2.5">
          <div className="mb-1.5 px-0.5 text-[10px] uppercase tracking-wider text-white/35">{t("overview.projects")}</div>
          <button className="flex w-full items-center justify-between rounded-lg bg-white/10 px-2.5 py-1.5 text-xs text-white">
            <span>{t("overview.pipelineLabel")}</span>
            <span className="text-[9px] text-white/40">{t("overview.pipelineNodes", { count: TOTAL })}</span>
          </button>
          <Link
            href="/evidence"
            className="mt-1 flex w-full items-center justify-between rounded-lg px-2.5 py-1.5 text-xs text-white/60 transition-colors hover:bg-white/[0.06] hover:text-white"
          >
            <span>{t("overview.evidence")}</span>
            <span className="text-[9px] text-white/40">{t("overview.evidenceMeta")}</span>
          </Link>
          <button
            disabled
            className="mt-1 flex w-full cursor-not-allowed items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs text-white/30"
          >
            <Plus className="size-3" />
            {t("overview.newProject")}
            <span className="ml-auto text-[9px] uppercase tracking-wider">{t("overview.soon")}</span>
          </button>
        </div>
      </GlassPanel>

      <div className="pointer-events-none absolute top-6 right-6 z-10 text-right text-xs">
        <div className="font-mono text-white/55">
          {t("overview.shippedSummary", { shipped: SHIPPED, total: TOTAL, live: IN_PROGRESS })}
        </div>
        <div className="mt-1 text-white/30">{t("brand.tagline")}</div>
      </div>

      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.4 }}
        nodesDraggable={false}
        nodesConnectable={false}
        proOptions={{ hideAttribution: true }}
        minZoom={0.4}
        maxZoom={1.5}
      >
        <Background variant={BackgroundVariant.Dots} gap={28} size={1} color="rgba(255,255,255,0.06)" />
      </ReactFlow>
    </div>
  );
}
